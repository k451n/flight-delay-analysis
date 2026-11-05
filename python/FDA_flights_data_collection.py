# --- FDA_flights_data_collection.py ---
"""
Сбор рейсов Пулково (ULLI) и Шереметьево (UUEE) через AeroDataBox (RapidAPI) -> MongoDB.

Запуск:
    export RAPIDAPI_KEY=...                             # ключ не хранить в коде и не коммитить
    python FDA_flights_data_collection.py --dry-run     # только план и оценка units
    python FDA_flights_data_collection.py               # скачать недостающее + импорт в MongoDB
    python FDA_flights_data_collection.py --import-only # только пересобрать MongoDB из кэша
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import DuplicateKeyError, OperationFailure

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "aerodatabox.p.rapidapi.com"
API_URL_TEMPLATE = (
    "https://aerodatabox.p.rapidapi.com/flights/airports/icao/{icao}/{from_time}/{to_time}"
)

AIRPORTS = [
    {"icao": "ULLI", "name": "pulkovo"},
    {"icao": "UUEE", "name": "sheremetyevo"},
]

# Период исследования (согласован: 180 дней).
START_DATE = date(2026, 5, 1)
PERIOD_END = date(2026, 10, 28)  # последний день, до которого собираем данные
HISTORY_DAYS = 180

MSK = timezone(timedelta(hours=3))

RAW_DIR = Path("raw_aerodatabox")  # кэш сырых JSON (в Git не коммитить)
PAUSE_SEC = 1.0                    # лимит - 2 запроса/сек, берём с запасом

UNITS_PER_CALL = 2
UNIT_BUDGET = int(os.environ.get("FLIGHTS_UNIT_BUDGET", "2000"))  # предохранитель за один запуск
MAX_CONSECUTIVE_FAILS = 5

MONGO_DB = "aviation_flights"      # та же БД, что и у погоды
MONGO_COLLECTION = "flights"


def get_mongo_uri():
    """MONGO_URI целиком, либо MONGO_HOST (имя хоста, как в main.py), либо localhost."""
    uri = os.environ.get("MONGO_URI")
    if uri:
        return uri
    host = os.environ.get("MONGO_HOST", "localhost")
    return host if "://" in host else f"mongodb://{host}:27017/"


# ==============================================================================
# 2. СБОР (с кэшем и защитой units)
# ==============================================================================
def day_windows(d):
    """Два окна по <12 ч. Формат AeroDataBox - YYYY-MM-DDTHH:mm (без секунд)."""
    return [(f"{d.isoformat()}T00:00", f"{d.isoformat()}T11:59"),
            (f"{d.isoformat()}T12:00", f"{d.isoformat()}T23:59")]


def cache_is_valid(path):
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  [Кэш] {path} повреждён, перезапросим")
        return False
    return isinstance(cached, dict) and ("departures" in cached or "arrivals" in cached)


def fetch_window(session, icao, from_str, to_str, max_retries=5):
    """Возвращает (status, data): ok | empty | fail | fatal."""
    url = API_URL_TEMPLATE.format(icao=icao, from_time=from_str, to_time=to_str)
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {"withLeg": "true", "withCancelled": "true", "direction": "Both"}

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  [сеть] {e}; попытка {attempt}/{max_retries}, ждём 15 с")
            time.sleep(15)
            continue

        code = r.status_code
        if code == 200:
            try:
                return "ok", r.json()
            except ValueError as e:
                print(f"  [JSON] не удалось разобрать ответ: {e}")
                time.sleep(5 * attempt)
                continue
        if code == 204:
            return "empty", {"departures": [], "arrivals": []}
        if code in (401, 403):
            print(f"  [{code}] нет доступа: проверьте ключ и подписку RapidAPI")
            return "fatal", None
        if code == 429:
            if "quota" in r.text.lower():
                print(f"  [429] исчерпана квота units: {r.text[:200]}")
                return "fatal", None
            print("  [429] превышена частота запросов, ждём 60 с")
            time.sleep(60)
            continue
        if code in (400, 404):
            # Повтор бессмысленен (неверное окно, дата вне 180 дней, неверный ICAO).
            print(f"  [{code}] запрос отклонён: {r.text[:200]}")
            return "fail", None
        print(f"  [HTTP {code}] попытка {attempt}/{max_retries}")
        time.sleep(5 * attempt)
    return "fail", None


def collect_all_data(start, end, dry_run=False):
    """
    Идём от старых дат к новым (старые первыми выпадают из окна 180 дней),
    внутри дня - по обоим аэропортам, чтобы они набирались синхронно.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo, total = [], 0
    d = start
    while d <= end:
        for ap in AIRPORTS:
            (RAW_DIR / ap["icao"]).mkdir(exist_ok=True)
            for idx, (f, t) in enumerate(day_windows(d), start=1):
                total += 1
                path = RAW_DIR / ap["icao"] / f"{ap['icao']}_{d.isoformat()}_part{idx}.json"
                if not cache_is_valid(path):
                    todo.append((ap["icao"], f, t, path))
        d += timedelta(days=1)

    est_units = len(todo) * UNITS_PER_CALL
    print(f"[PLAN] Период {start} .. {end}: всего окон {total}, уже в кэше {total - len(todo)}, "
          f"осталось {len(todo)} вызовов (~{est_units} units, предохранитель {UNIT_BUDGET}).")
    if dry_run or not todo:
        return True
    if not RAPIDAPI_KEY:
        print("[ОШИБКА] Не задана переменная окружения RAPIDAPI_KEY.")
        return False

    session = requests.Session()
    units_used, fails_in_row, failed = 0, 0, []
    for i, (icao, f, t, path) in enumerate(todo, start=1):
        if units_used + UNITS_PER_CALL > UNIT_BUDGET:
            print(f"[СТОП] Достигнут предохранитель {UNIT_BUDGET} units. "
                  f"Повысьте FLIGHTS_UNIT_BUDGET, если квота позволяет.")
            return False

        print(f"[API {i}/{len(todo)}] {icao} {f} -> {t}")
        status, data = fetch_window(session, icao, f, t)

        if status in ("ok", "empty"):
            units_used += UNITS_PER_CALL
            fails_in_row = 0
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            tmp.replace(path)
        elif status == "fatal":
            print("[КРИТИЧЕСКАЯ ОШИБКА] Сбор остановлен, прогресс сохранён в кэше.")
            return False
        else:
            failed.append((icao, f, t))
            fails_in_row += 1
            if fails_in_row >= MAX_CONSECUTIVE_FAILS:
                print(f"[СТОП] {MAX_CONSECUTIVE_FAILS} неудач подряд - вероятно, проблема "
                      f"в параметрах или доступе. Прогресс сохранён.")
                return False

        time.sleep(PAUSE_SEC)

    print(f"[ИТОГ СБОРА] Израсходовано ~{units_used} units, неудачных окон: {len(failed)}")
    for item in failed:
        print(f"  пропущено: {item[0]} {item[1]} -> {item[2]} (запустите скрипт ещё раз)")
    return not failed


# ==============================================================================
# 3. НОРМАЛИЗАЦИЯ И ИМПОРТ В MONGODB
# ==============================================================================
def parse_utc(value):
    """'2026-05-01 10:00Z' / ISO-строка -> наивный datetime в UTC (удобно для BSON)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def movement_time(mv, kind):
    """kind: scheduled | revised | runway. Поддерживает {'utc':..} и устаревший '<kind>TimeUtc'."""
    t = mv.get(f"{kind}Time")
    value = t.get("utc") if isinstance(t, dict) else t
    return parse_utc(value or mv.get(f"{kind}TimeUtc"))


def parse_flight(fl, base_icao, base_name, direction):
    """Один рейс из ответа FIDS -> документ MongoDB (или None, если нет времени)."""
    dep, arr = fl.get("departure") or {}, fl.get("arrival") or {}
    if direction == "departure":
        base_mv, other_mv = dep or fl.get("movement") or {}, arr
    else:
        base_mv, other_mv = arr or fl.get("movement") or {}, dep

    scheduled = movement_time(base_mv, "scheduled")
    revised = movement_time(base_mv, "revised")
    runway = movement_time(base_mv, "runway")
    if scheduled is None:
        return None

    status = fl.get("status")
    is_cancelled = status == "Canceled"
    delay_min = runway_delay_min = None
    if not is_cancelled:
        if revised is not None:
            delay_min = round((revised - scheduled).total_seconds() / 60.0, 1)
        if runway is not None:
            runway_delay_min = round((runway - scheduled).total_seconds() / 60.0, 1)

    number = (fl.get("number") or "unknown").replace(" ", "")
    other_ap = other_mv.get("airport") or {}
    aircraft = fl.get("aircraft") or {}
    airline = fl.get("airline") or {}

    return {
        "flight_uid": f"{base_icao}_{direction}_{number}_{scheduled:%Y%m%dT%H%M}",
        "flight_number": number,
        "base_airport": base_icao,
        "airport_name": base_name,
        "direction": direction,
        "other_airport_icao": other_ap.get("icao"),
        "other_airport_iata": other_ap.get("iata"),
        "other_airport_name": other_ap.get("name"),
        "airline_iata": airline.get("iata"),
        "airline_icao": airline.get("icao"),
        "airline_name": airline.get("name"),
        "aircraft_reg": aircraft.get("reg"),
        "aircraft_mode_s": aircraft.get("modeS"),
        "aircraft_model": aircraft.get("model"),
        "status": status,                       # Departed / Arrived / Canceled / ...
        "is_cancelled": is_cancelled,
        "codeshare_status": fl.get("codeshareStatus"),  # для анализа брать IsOperator
        "is_cargo": fl.get("isCargo"),
        "scheduled_time_utc": scheduled,
        "revised_time_utc": revised,
        "runway_time_utc": runway,
        "delay_min": delay_min,                 # revised - scheduled (у гейта), мин
        "runway_delay_min": runway_delay_min,   # runway - scheduled (взлёт/посадка), мин
        "hour_utc": scheduled.replace(minute=0, second=0, microsecond=0),  # ключ join с погодой
        "scheduled_time_msk": scheduled + timedelta(hours=3),
        "last_updated_utc": parse_utc(fl.get("lastUpdatedUtc")),
    }


def ensure_index(collection, keys, **options):
    """
    create_index, устойчивый к конфликту с уже существующим индексом (IndexKeySpecsConflict):
    если индекс с тем же именем создан раньше с другими параметрами (например, без unique),
    он пересоздаётся.
    """
    try:
        collection.create_index(keys, **options)
    except OperationFailure as e:
        if e.code not in (85, 86):  # IndexOptionsConflict / IndexKeySpecsConflict
            raise
        name = "_".join(f"{k}_{v}" for k, v in keys)
        print(f"  [INDEX] Индекс {name} уже есть с другими параметрами - пересоздаю")
        collection.drop_index(name)
        collection.create_index(keys, **options)


def normalize_and_save_to_mongo(reset=False):
    client = MongoClient(get_mongo_uri())
    collection = client[MONGO_DB][MONGO_COLLECTION]

    if reset:
        collection.drop()
        print(f"[MONGO] Коллекция {MONGO_DB}.{MONGO_COLLECTION} очищена (--reset-flights)")
    else:
        legacy = collection.count_documents({"scheduled_time_utc": {"$exists": False}})
        if legacy:
            print(f"[ВНИМАНИЕ] В {MONGO_COLLECTION} есть {legacy} документов старого формата "
                  f"(без scheduled_time_utc), например от тестового прогона. "
                  f"Запустите с --reset-flights, чтобы очистить коллекцию.")

    # Индексы создаём до вставки: upsert по flight_uid идемпотентен, повторный запуск безопасен.
    try:
        ensure_index(collection, [("flight_uid", 1)], unique=True)
    except DuplicateKeyError:
        print("[ОШИБКА] В коллекции уже есть дубли/пустые flight_uid (старые данные). "
              "Запустите с --reset-flights.")
        client.close()
        raise SystemExit(1)
    ensure_index(collection, [("base_airport", 1), ("direction", 1), ("scheduled_time_utc", 1)])
    ensure_index(collection, [("hour_utc", 1)])
    ensure_index(collection, [("status", 1)])

    def flush(buffer):
        if buffer:
            collection.bulk_write(
                [ReplaceOne({"flight_uid": uid}, doc, upsert=True) for uid, doc in buffer.items()],
                ordered=False,
            )
            buffer.clear()

    for airport in AIRPORTS:
        icao, name = airport["icao"], airport["name"]
        files = sorted((RAW_DIR / icao).glob("*.json"))
        print(f"\n[PARSER] {icao}: файлов кэша {len(files)}")
        stats, buffer = Counter(), {}

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [Пропуск] {path}: {e}")
                continue
            for key, direction in (("departures", "departure"), ("arrivals", "arrival")):
                for fl in content.get(key, []) or []:
                    doc = parse_flight(fl, icao, name, direction)
                    if doc is None:
                        stats["без_времени"] += 1
                        continue
                    buffer[doc["flight_uid"]] = doc
                    stats["разобрано"] += 1
                    stats[f"статус:{doc['status']}"] += 1
            if len(buffer) >= 5000:
                flush(buffer)
        flush(buffer)

        print(f"[PARSER] {icao}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        if stats["без_времени"] and stats["без_времени"] >= stats["разобрано"]:
            print("  [ВНИМАНИЕ] Времена не распознаны - проверьте структуру сырого JSON "
                  "(поля departure/arrival.scheduledTime.utc).")

        in_db = collection.count_documents({"base_airport": icao})
        cancelled = collection.count_documents({"base_airport": icao, "is_cancelled": True})
        with_delay = collection.count_documents({"base_airport": icao, "delay_min": {"$ne": None}})
        print(f"[MONGO] {icao}: в базе {in_db}, отменено {cancelled}, с рассчитанной задержкой {with_delay}")

    client.close()


# ==============================================================================
# 4. ТОЧКА ВХОДА
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Сбор рейсов AeroDataBox -> MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="только план и оценка units")
    parser.add_argument("--import-only", action="store_true", help="не обращаться к API, только импорт кэша")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None,
                        help="последний день сбора, напр. 2026-05-10 для пробного прогона "
                             "(по умолчанию: вчера, но не позже 2026-10-28)")
    parser.add_argument("--reset-flights", action="store_true",
                        help="очистить коллекцию flights перед импортом")
    args = parser.parse_args()

    today = datetime.now(MSK).date()
    earliest = today - timedelta(days=HISTORY_DAYS)
    start = START_DATE
    end = args.end_date or min(PERIOD_END, today - timedelta(days=1))  # сегодняшний день не завершён

    if start < earliest:
        print(f"[ВНИМАНИЕ] {start} уже вне окна истории (доступно с {earliest}); старт сдвинут.")
        start = earliest
    expires = START_DATE + timedelta(days=HISTORY_DAYS)
    print(f"[INFO] Сегодня {today}. {START_DATE} будет доступен до {expires} "
          f"(осталось дней: {(expires - today).days}).")
    if end < start:
        print("[INFO] Нечего собирать: конец периода раньше начала.")
        return 0

    if not args.import_only:
        ok = collect_all_data(start, end, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        if not ok:
            print("[ВНИМАНИЕ] Сбор завершён не полностью. Импортируем то, что есть в кэше.")

    print("[MONGO] Нормализация и загрузка в базу...")
    normalize_and_save_to_mongo(reset=args.reset_flights)
    print("=== ГОТОВО ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
