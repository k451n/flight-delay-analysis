# --- FDA_weather_data_collection.py ---
"""
Сбор погоды для Пулково (ULLI) и Шереметьево (UUEE) -> MongoDB.

Запуск:
    python FDA_weather_data_collection.py --dry-run                 # только план
    python FDA_weather_data_collection.py                           # скачать недостающее + импорт
    python FDA_weather_data_collection.py --import-only             # только пересобрать MongoDB из кэша
    python FDA_weather_data_collection.py --reset-weather           # очистить hourly_weather/metar_weather
    python FDA_weather_data_collection.py --export-csv dataset      # дополнительно выгрузить CSV
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import DuplicateKeyError, OperationFailure

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
AIRPORTS = [
    {"icao": "ULLI", "name": "Pulkovo", "lat": 59.8003, "lon": 30.2625},
    {"icao": "UUEE", "name": "Sheremetyevo", "lat": 55.9728, "lon": 37.4147},
]

# Период исследования
PERIOD_START = date(2026, 5, 1)
PERIOD_END = date(2026, 10, 28)
START_BUFFER_DAYS = 1

RAW_DIR = Path("raw_weather")  # кэш сырых ответов (в Git не коммитить)
PAUSE_SEC = 1.0
MAX_CONSECUTIVE_FAILS = 3
HEADERS = {"User-Agent": "FDA-course-project/1.0 (student research, weather collector)"}

# --- Open-Meteo ---
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_LAG_DAYS = 2
OPENMETEO_FINAL_DAYS = 7
HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall",
    "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "weather_code",
]

# --- METAR (Iowa Mesonet) ---
METAR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
METAR_FINAL_DAYS = 1
METAR_NUMERIC = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby",
                 "skyl1", "skyl2", "skyl3", "p01i"]
METAR_TEXT = ["skyc1", "skyc2", "skyc3", "wxcodes", "metar"]

# --- MongoDB ---
MONGO_DB = os.environ.get("MONGO_DB_WEATHER", "aviation_weather")
HOURLY_COLLECTION = "hourly_weather"
METAR_COLLECTION = "metar_weather"


def get_mongo_uri():
    """MONGO_URI целиком, либо MONGO_HOST (имя хоста, как в main.py), либо localhost."""
    uri = os.environ.get("MONGO_URI")
    if uri:
        return uri
    host = os.environ.get("MONGO_HOST", "localhost")
    return host if "://" in host else f"mongodb://{host}:27017/"


# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    """'2026-05-01T00:00' / '2026-05-01 00:00' / '... +00:00' -> naive datetime в UTC."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def to_float(value):
    """Число или None. Пустые значения, 'M' (missing в METAR) и NaN -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        s = str(value).strip()
        if s in ("", "M", "null", "None", "nan", "NaN"):
            return None
        try:
            x = float(s)
        except ValueError:
            return None
    return None if x != x else x


def month_chunks(start, end):
    """Месячные куски [first, last] (обе даты включительно) в пределах [start, end]."""
    d = date(start.year, start.month, 1)
    while d <= end:
        nxt = date(d.year + d.month // 12, d.month % 12 + 1, 1)
        first, last = max(start, d), min(end, nxt - timedelta(days=1))
        if first <= last:
            yield d.strftime("%Y-%m"), first, last
        d = nxt


def write_atomic(path, text):
    """Атомарная запись: без битых файлов при Ctrl+C."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    tmp.replace(path)


def meta_dict(first, last):
    return {"first": first.isoformat(), "last": last.isoformat(),
            "fetched_at_utc": utc_now().isoformat()}


def read_meta(path, source):
    """
    Метаданные кэша (какой период запрошен и когда скачан) или None, если файла нет/он битый.
    JSON: ключ "_meta"; CSV: первая строка "# first=... last=... fetched_at_utc=...".
    """
    try:
        if source == "openmeteo":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = data["_meta"]
            if "hourly" not in data:
                return None
        else:
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
            if not line.startswith("#"):
                return None
            m = dict(p.split("=", 1) for p in line.lstrip("# ").split())
        return {"first": date.fromisoformat(m["first"]),
                "last": date.fromisoformat(m["last"]),
                "fetched": datetime.fromisoformat(m["fetched_at_utc"])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def cache_is_final(meta, first, last, final_days):
    """
    Кэш не нужно перекачивать, если он покрывает нужный период И был скачан
    не раньше чем через final_days после его конца (то есть данные уже не меняются).
    Иначе (свежий хвост периода) кусок перекачивается при каждом запуске.
    """
    if not meta or meta["first"] > first or meta["last"] < last:
        return False
    return meta["fetched"].date() >= last + timedelta(days=final_days)


# ==============================================================================
# 3. СБОР (кэш по месяцам, повторные запросы)
# ==============================================================================
def build_tasks(start, period_end, today):
    """Список кусков (аэропорт x источник x месяц) с признаком 'уже в кэше и окончателен'."""
    openmeteo_end = min(period_end, today - timedelta(days=OPENMETEO_LAG_DAYS))
    metar_end = min(period_end, today)
    tasks = []
    for ap in AIRPORTS:
        for source, end, ext, final_days in (
                ("openmeteo", openmeteo_end, "json", OPENMETEO_FINAL_DAYS),
                ("metar", metar_end, "csv", METAR_FINAL_DAYS)):
            for key, first, last in month_chunks(start, end):
                path = RAW_DIR / source / ap["icao"] / f"{ap['icao']}_{key}.{ext}"
                tasks.append({
                    "source": source, "ap": ap, "key": key, "first": first, "last": last,
                    "path": path,
                    "cached": cache_is_final(read_meta(path, source), first, last, final_days),
                })
    return tasks, {"openmeteo_end": openmeteo_end, "metar_end": metar_end}


def fetch_openmeteo(session, ap, first, last, max_retries=5):
    """Возвращает (status, data): ok | fail | fatal."""
    params = {
        "latitude": ap["lat"], "longitude": ap["lon"],
        "start_date": first.isoformat(), "end_date": last.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",  # времена приходят в UTC
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(OPENMETEO_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"  [сеть] {e}; попытка {attempt}/{max_retries}, ждём 15 с")
            time.sleep(15)
            continue

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError as e:
                print(f"  [JSON] не удалось разобрать ответ: {e}")
                time.sleep(5 * attempt)
                continue
            if not isinstance(data.get("hourly"), dict) or "time" not in data["hourly"]:
                print(f"  [Open-Meteo] неожиданная структура ответа: {str(data)[:200]}")
                return "fail", None
            return "ok", data
        if r.status_code == 429:
            if "daily" in r.text.lower():
                print(f"  [429] исчерпан суточный лимит Open-Meteo: {r.text[:200]}")
                return "fatal", None
            print("  [429] превышена частота запросов, ждём 60 с")
            time.sleep(60)
            continue
        if r.status_code in (400, 404):
            print(f"  [{r.status_code}] запрос отклонён: {r.text[:300]}")
            return "fail", None
        print(f"  [HTTP {r.status_code}] попытка {attempt}/{max_retries}")
        time.sleep(5 * attempt)
    return "fail", None


def fetch_metar(session, ap, first, last, max_retries=5):
    """Возвращает (status, csv_text): ok | fail. Конец периода в IEM не включается."""
    end_excl = last + timedelta(days=1)
    params = [
        ("station", ap["icao"]), ("data", "all"),
        ("year1", first.year), ("month1", first.month), ("day1", first.day),
        ("year2", end_excl.year), ("month2", end_excl.month), ("day2", end_excl.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),
    ]
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(METAR_URL, params=params, timeout=180)
        except requests.exceptions.RequestException as e:
            print(f"  [сеть] {e}; попытка {attempt}/{max_retries}, ждём 15 с")
            time.sleep(15)
            continue

        if r.status_code == 200:
            lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
            if not lines or "valid" not in lines[0].split(","):
                print(f"  [METAR] в ответе нет заголовка CSV: {r.text[:200]!r}")
                return "fail", None
            return "ok", r.text
        if r.status_code in (400, 404):
            print(f"  [{r.status_code}] запрос отклонён: {r.text[:300]}")
            return "fail", None
        print(f"  [HTTP {r.status_code}] попытка {attempt}/{max_retries}")
        time.sleep(5 * attempt)
    return "fail", None


def collect_all(tasks, dry_run=False):
    todo = [t for t in tasks if not t["cached"]]
    print(f"[PLAN] Кусков всего {len(tasks)}, уже в кэше {len(tasks) - len(todo)}, "
          f"осталось запросов {len(todo)}.")
    if dry_run:
        for t in todo:
            print(f"  скачать: {t['source']:9s} {t['ap']['icao']} {t['first']} .. {t['last']}")
    if dry_run or not todo:
        return True

    session = requests.Session()
    session.headers.update(HEADERS)
    fails_in_row, failed = 0, []
    for i, t in enumerate(todo, start=1):
        icao, first, last = t["ap"]["icao"], t["first"], t["last"]
        print(f"[API {i}/{len(todo)}] {t['source']} {icao} {first} .. {last}")
        if t["source"] == "openmeteo":
            status, payload = fetch_openmeteo(session, t["ap"], first, last)
        else:
            status, payload = fetch_metar(session, t["ap"], first, last)

        if status == "ok":
            fails_in_row = 0
            meta = meta_dict(first, last)
            if t["source"] == "openmeteo":
                payload["_meta"] = meta
                write_atomic(t["path"], json.dumps(payload, ensure_ascii=False))
            else:
                header = "# first={first} last={last} fetched_at_utc={fetched_at_utc}\n".format(**meta)
                write_atomic(t["path"], header + payload)
        elif status == "fatal":
            print("[КРИТИЧЕСКАЯ ОШИБКА] Сбор остановлен, прогресс сохранён в кэше.")
            return False
        else:
            failed.append(t)
            fails_in_row += 1
            if fails_in_row >= MAX_CONSECUTIVE_FAILS:
                print(f"[СТОП] {MAX_CONSECUTIVE_FAILS} неудачи подряд - проверьте сеть и параметры. "
                      f"Прогресс сохранён.")
                return False
        time.sleep(PAUSE_SEC)

    print(f"[ИТОГ СБОРА] Неудачных кусков: {len(failed)}")
    for t in failed:
        print(f"  пропущено: {t['source']} {t['ap']['icao']} {t['first']} .. {t['last']} "
              f"(запустите скрипт ещё раз)")
    return not failed


# ==============================================================================
# 4. НОРМАЛИЗАЦИЯ
# ==============================================================================
def parse_openmeteo(data, ap):
    """JSON Open-Meteo -> документы hourly_weather. Часы без единого значения (хвост архива) пропускаются."""
    hourly = data["hourly"]
    times = hourly["time"]
    series = {v: hourly.get(v) or [None] * len(times) for v in HOURLY_VARIABLES}
    rows = []
    for i, t in enumerate(times):
        hour = parse_utc(t)
        values = {v: to_float(series[v][i]) for v in HOURLY_VARIABLES}
        if hour is None or all(x is None for x in values.values()):
            continue
        rows.append({"icao": ap["icao"], "airport": ap["name"], "hour_utc": hour,
                     **values, "source": "open-meteo-archive"})
    return rows


def parse_metar_csv(text, icao):
    """CSV Iowa Mesonet -> документы metar_weather (дубликаты по (station, valid) схлопываются)."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    docs = {}
    for rec in csv.DictReader(lines):
        valid = parse_utc(rec.get("valid"))
        if valid is None:
            continue
        doc = {"station": icao, "valid": valid,
               "hour_utc": valid.replace(minute=0, second=0, microsecond=0)}
        for col in METAR_NUMERIC:
            raw = rec.get(col)
            if col == "p01i" and raw is not None and raw.strip() == "T":
                doc[col] = 0.0001
            else:
                doc[col] = to_float(raw)
        for col in METAR_TEXT:
            raw = rec.get(col)
            doc[col] = None if raw is None or raw.strip() in ("", "M") else raw.strip()
        docs[valid] = doc
    return list(docs.values())


def load_rows(icao, source):
    """Читает весь кэш аэропорта и возвращает документы, отсортированные по времени."""
    ap = next(a for a in AIRPORTS if a["icao"] == icao)
    folder = RAW_DIR / source / icao
    ext = "json" if source == "openmeteo" else "csv"
    key = "hour_utc" if source == "openmeteo" else "valid"
    merged = {}
    for path in sorted(folder.glob(f"*.{ext}")):
        try:
            if source == "openmeteo":
                with open(path, "r", encoding="utf-8") as f:
                    rows = parse_openmeteo(json.load(f), ap)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    rows = parse_metar_csv(f.read(), icao)
        except (OSError, ValueError, KeyError) as e:
            print(f"  [Пропуск] {path}: {e}")
            continue
        for row in rows:
            merged[row[key]] = row  # более поздний файл перекрывает более ранний
    return [merged[k] for k in sorted(merged)]


def report_hourly(icao, rows):
    if not rows:
        print(f"[CHECK] Open-Meteo {icao}: данных нет")
        return
    first, last = rows[0]["hour_utc"], rows[-1]["hour_utc"]
    expected = int((last - first).total_seconds() // 3600) + 1
    nulls = {v: sum(1 for r in rows if r[v] is None) for v in HOURLY_VARIABLES}
    nulls = {k: n for k, n in nulls.items() if n}
    print(f"[CHECK] Open-Meteo {icao}: {len(rows)} ч, {first} .. {last} UTC, "
          f"пропущено часов внутри диапазона: {expected - len(rows)}, "
          f"пустые значения: {nulls or 'нет'}")


def report_metar(icao, rows):
    if not rows:
        print(f"[CHECK] METAR {icao}: данных нет")
        return
    first, last = rows[0]["valid"], rows[-1]["valid"]
    hours = {r["hour_utc"] for r in rows}
    expected = int((last.replace(minute=0) - first.replace(minute=0)).total_seconds() // 3600) + 1
    no_vis = sum(1 for r in rows if r["vsby"] is None)
    print(f"[CHECK] METAR {icao}: {len(rows)} наблюдений, {first} .. {last} UTC, "
          f"часов с наблюдениями {len(hours)} из {expected}, без видимости: {no_vis}")


def export_csv(folder, name, rows, fields):
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / name
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r[c].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r[c], datetime)
                        else ("" if r[c] is None else r[c]) for c in fields])
    print(f"  CSV: {path} ({len(rows)} строк)")


# ==============================================================================
# 5. ИМПОРТ В MONGODB
# ==============================================================================
def ensure_index(collection, keys, **options):
    """create_index, устойчивый к конфликту с ранее созданным индексом другой конфигурации."""
    try:
        collection.create_index(keys, **options)
    except OperationFailure as e:
        if e.code not in (85, 86):  # IndexOptionsConflict / IndexKeySpecsConflict
            raise
        name = "_".join(f"{k}_{v}" for k, v in keys)
        print(f"  [INDEX] Индекс {name} уже есть с другими параметрами - пересоздаю")
        collection.drop_index(name)
        collection.create_index(keys, **options)


def upsert(collection, docs, key_fields, batch=5000):
    ops = [ReplaceOne({k: d[k] for k in key_fields}, d, upsert=True) for d in docs]
    for i in range(0, len(ops), batch):
        collection.bulk_write(ops[i:i + batch], ordered=False)


def save_to_mongo(hourly_by_icao, metar_by_icao, reset=False):
    client = MongoClient(get_mongo_uri())
    db = client[MONGO_DB]
    hourly, metar = db[HOURLY_COLLECTION], db[METAR_COLLECTION]

    if reset:
        hourly.drop()
        metar.drop()
        print(f"[MONGO] Коллекции {HOURLY_COLLECTION} и {METAR_COLLECTION} очищены (--reset-weather)")
    else:
        for coll in (hourly, metar):
            legacy = coll.count_documents({"hour_utc": {"$exists": False}})
            if legacy:
                print(f"[ВНИМАНИЕ] В {coll.name} есть {legacy} документов старого формата "
                      f"(без hour_utc, например данные за 2025 год). "
                      f"Запустите с --reset-weather, чтобы очистить коллекции.")

    try:
        ensure_index(hourly, [("icao", 1), ("hour_utc", 1)], unique=True)
        ensure_index(metar, [("station", 1), ("valid", 1)], unique=True)
    except DuplicateKeyError:
        print("[ОШИБКА] В коллекциях есть дубли/старые документы, уникальный индекс не создаётся. "
              "Запустите с --reset-weather.")
        client.close()
        raise SystemExit(1)
    ensure_index(hourly, [("hour_utc", 1)])
    ensure_index(metar, [("hour_utc", 1)])

    for ap in AIRPORTS:
        icao = ap["icao"]
        upsert(hourly, hourly_by_icao[icao], ("icao", "hour_utc"))
        upsert(metar, metar_by_icao[icao], ("station", "valid"))
        print(f"[MONGO] {icao}: {HOURLY_COLLECTION} = {hourly.count_documents({'icao': icao})}, "
              f"{METAR_COLLECTION} = {metar.count_documents({'station': icao})}")
    client.close()


# ==============================================================================
# 6. ТОЧКА ВХОДА
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Сбор погоды (Open-Meteo + METAR) -> MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="только план запросов")
    parser.add_argument("--import-only", action="store_true", help="не обращаться к API, только импорт кэша")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None,
                        help="последний день периода (по умолчанию 2026-10-28)")
    parser.add_argument("--reset-weather", action="store_true",
                        help="очистить коллекции hourly_weather и metar_weather перед импортом")
    parser.add_argument("--export-csv", metavar="DIR", default=None,
                        help="дополнительно сохранить CSV в указанную папку")
    args = parser.parse_args()

    today = utc_now().date()
    start = PERIOD_START - timedelta(days=START_BUFFER_DAYS)
    period_end = args.end_date or PERIOD_END
    tasks, plan = build_tasks(start, period_end, today)
    print(f"[INFO] Сегодня {today} (UTC). Период {start} .. {period_end}. "
          f"Open-Meteo доступен до {plan['openmeteo_end']}, METAR до {plan['metar_end']}.")
    if plan["openmeteo_end"] < period_end:
        print(f"[INFO] Архив Open-Meteo отстаёт: часы после {plan['openmeteo_end']} "
              f"догрузятся повторным запуском (погода не «протухает», в отличие от рейсов).")

    if not args.import_only:
        ok = collect_all(tasks, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        if not ok:
            print("[ВНИМАНИЕ] Сбор завершён не полностью. Импортируем то, что есть в кэше.")

    hourly_by_icao, metar_by_icao = {}, {}
    for ap in AIRPORTS:
        icao = ap["icao"]
        hourly_by_icao[icao] = load_rows(icao, "openmeteo")
        metar_by_icao[icao] = load_rows(icao, "metar")
        report_hourly(icao, hourly_by_icao[icao])
        report_metar(icao, metar_by_icao[icao])

    if not any(hourly_by_icao.values()) and not any(metar_by_icao.values()):
        print("[ОШИБКА] В кэше нет данных, импортировать нечего.")
        return 1

    print("[MONGO] Загрузка в базу...")
    save_to_mongo(hourly_by_icao, metar_by_icao, reset=args.reset_weather)

    if args.export_csv:
        print("[CSV] Выгрузка...")
        for ap in AIRPORTS:
            icao = ap["icao"]
            export_csv(args.export_csv, f"weather_{ap['name'].lower()}_2026.csv", hourly_by_icao[icao],
                       ["icao", "airport", "hour_utc"] + HOURLY_VARIABLES)
            export_csv(args.export_csv, f"metar_{icao.lower()}_2026.csv", metar_by_icao[icao],
                       ["station", "valid", "hour_utc"] + METAR_NUMERIC + METAR_TEXT)
    print("=== ГОТОВО ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
