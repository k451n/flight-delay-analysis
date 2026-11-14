# --- FDA_import.py ---
"""
Импорт СВОИХ данных из CSV в почасовой датасет, с которым работает FDA_main_analyzer.py.

Поддерживаются два способа подготовки данных.

А) Готовый почасовой CSV (один файл, строка = час):
     hour_utc                 время начала часа (UTC; допускаются time / datetime / timestamp)
     n_flights, n_completed, n_cancelled, n_delayed15, mean_delay_min       - рейсы за час
     temperature_2m, relative_humidity_2m, precipitation, snowfall,
     wind_speed_10m, wind_gusts_10m, cloud_cover                            - погода (км/ч, °C, мм, %)
   необязательно: n_diverted, median_delay_min, vsby_mi (или visibility_km), ceiling_ft, n_metar.
   Это формат файлов hourly_dataset_*.csv, которые создаёт main.py.

Б) Исходные таблицы (как в коллекциях сборщиков FDA_*):
     weather: время + признаки Open-Meteo (формат hourly_weather или выгрузки Open-Meteo)
     flights: по строке на рейс: время (scheduled_time_utc / hour_utc), delay_min,
              is_cancelled или status; необязательно direction, codeshare_status, is_cargo, base_airport
     metar (необязательно): выгрузка Iowa Mesonet - valid, vsby, skyc1-3, skyl1-3 ('M' = нет данных)

Общие правила чтения: разделитель , ; или табуляция определяется автоматически (для ; десятичная запятая),
строки-комментарии '#' в начале пропускаются, 'M', 'NA', 'null' и пустые ячейки = пропуск, кодировка UTF-8 или CP1251.
Время с часовым поясом (+03:00) переводится в UTC; время без пояса считается UTC (naive_tz="UTC") или
московским (naive_tz="MSK").
"""
import io
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import FDA_main_analyzer as M

MSK_HOURS = 3
TIME_ALIASES = ["hour_utc", "time", "datetime", "timestamp", "date_time", "time_utc", "valid", "scheduled_time_utc"]
NA_VALUES = ["", "M", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "-"]
HOURLY_REQUIRED = ["n_flights", "n_completed", "n_cancelled", "n_delayed15", "mean_delay_min"]
TRUE_VALUES = {"true", "1", "yes", "y", "да", "t"}


class ImportProblem(ValueError):
    """Ошибка в данных пользователя: сообщение показывается как есть."""


@dataclass
class DataSpec:
    """Что и откуда загружать."""
    kind: str = "hourly"                 # "hourly" | "raw"
    hourly: str = ""                     # путь к почасовому CSV
    weather: str = ""                    # пути для kind="raw"
    flights: str = ""
    metar: str = ""
    naive_tz: str = "UTC"                # как понимать время без часового пояса: "UTC" | "MSK"
    airport: str = ""                    # ICAO: если в файлах несколько аэропортов
    delay_field: str = "delay_min"
    direction: str = "both"              # both | departure | arrival
    include_codeshare: bool = False
    exclude_cargo: bool = False
    start: date = None                   # необязательное ограничение периода (даты UTC, включительно)
    end: date = None


@dataclass
class ImportResult:
    ds: pd.DataFrame
    notes: list = field(default_factory=list)       # что найдено (информация)
    warnings: list = field(default_factory=list)    # на что обратить внимание


# ==============================================================================
# Чтение файлов
# ==============================================================================
def _read_text(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ImportProblem(f"{Path(path).name}: не удалось определить кодировку (нужна UTF-8 или CP1251)")


def read_table(path):
    """CSV -> DataFrame (строки-комментарии '#' пропускаются, разделитель определяется по заголовку)."""
    p = Path(path)
    if not p.is_file():
        raise ImportProblem(f"Файл не найден: {path}")
    lines = _read_text(p).splitlines()
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ImportProblem(f"{p.name}: нет данных (нужны заголовок и хотя бы одна строка)")
    header = lines[0]
    sep = max((",", ";", "\t"), key=header.count)
    if header.count(sep) == 0:
        raise ImportProblem(f"{p.name}: не найден разделитель колонок (запятая, точка с запятой или табуляция)")
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, decimal="," if sep == ";" else ".",
                         na_values=NA_VALUES, keep_default_na=False, low_memory=False)
    except Exception as e:  # noqa: BLE001 - любая ошибка разбора -> понятное сообщение
        raise ImportProblem(f"{p.name}: не удалось разобрать CSV ({e})") from e
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def parse_time(series, naive_tz="UTC", name="время"):
    """Время -> naive UTC, округлённое вниз до часа. Нечитаемые значения -> NaT."""
    s = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if naive_tz == "MSK":
        # значения без пояса мы прочитали как UTC; их надо понимать как МСК. Значения с поясом уже верны,
        # но отличить их после to_datetime нельзя, поэтому смотрим на исходный текст
        has_tz = series.astype(str).str.contains(r"(?:[+-]\d\d:?\d\d|Z)$", regex=True)
        s = s.where(has_tz, s - pd.Timedelta(hours=MSK_HOURS))
    return s.dt.tz_convert(None).dt.floor("h")


def _find_time_column(df, preferred=None, what="файл"):
    names = ([preferred] if preferred else []) + TIME_ALIASES
    for n in names:
        if n in df.columns:
            return n
    raise ImportProblem(f"{what}: не найдена колонка времени (ожидается одна из: {', '.join(TIME_ALIASES)}). "
                        f"Есть колонки: {', '.join(df.columns[:12])}")


def _with_hour(df, spec, what, preferred=None):
    col = _find_time_column(df, preferred, what)
    hour = parse_time(df[col], spec.naive_tz)
    bad = int(hour.isna().sum())
    if bad == len(df):
        raise ImportProblem(f"{what}: колонка {col} не содержит распознаваемых дат (пример: {df[col].iloc[0]!r})")
    out = df.assign(hour_utc=hour).dropna(subset=["hour_utc"])
    return out, col, bad


def _filter_airport(df, spec, what, columns=("base_airport", "icao", "station", "airport"), soft=("airport",)):
    """
    Оставляет строки выбранного аэропорта. Колонки из `soft` могут содержать название (Pulkovo), а не ICAO:
    если ICAO в них не находится и значение одно - файл берётся целиком.
    """
    for c in columns:
        if c not in df.columns:
            continue
        values = df[c].dropna().astype(str).unique()
        matches = df[c].astype(str).str.upper() == spec.airport.upper() if spec.airport else None
        if spec.airport and matches.any():
            return df[matches]
        if len(values) <= 1:
            continue
        if spec.airport and c not in soft:
            raise ImportProblem(f"{what}: аэропорта {spec.airport} нет в колонке {c} (там: {', '.join(values[:8])})")
        raise ImportProblem(f"{what}: в колонке {c} несколько значений ({', '.join(values[:8])}). "
                            f"Укажите аэропорт (ICAO), который нужно анализировать.")
    return df


# ==============================================================================
# Б) Исходные таблицы
# ==============================================================================
def load_raw_tables(spec, result):
    """weather + flights (+ metar) -> три почасовые таблицы."""
    if not spec.weather or not spec.flights:
        raise ImportProblem("Для исходных таблиц нужны файл погоды и файл рейсов (METAR - по желанию).")

    w = read_table(spec.weather)
    w = _filter_airport(w, spec, "Погода", columns=("icao", "airport", "base_airport"))
    w, tcol, bad = _with_hour(w, spec, "Погода")
    if bad:
        result.warnings.append(f"Погода: {bad} строк с нечитаемым временем пропущено.")
    present = [c for c in M.OPENMETEO_FEATURES if c in w.columns]
    if not present:
        raise ImportProblem("Погода: не найдено ни одной погодной колонки (" + ", ".join(M.OPENMETEO_FEATURES) + ")")
    missing = [c for c in M.OPENMETEO_FEATURES if c not in w.columns]
    if missing:
        result.warnings.append("Погода: нет колонок " + ", ".join(missing) + " - эти признаки не анализируются.")
    dup = int(w.duplicated("hour_utc").sum())
    if dup:
        result.warnings.append(f"Погода: {dup} дублей часов, оставлена первая строка.")
    weather_h = M.weather_frame_to_hourly(w)
    result.notes.append(f"Погода: {len(weather_h)} часов ({weather_h.index.min()} .. {weather_h.index.max()} UTC).")

    # --- METAR: отдельный файл либо колонки vsby / ceiling_ft в файле погоды
    if spec.metar:
        m = read_table(spec.metar)
        m = _filter_airport(m, spec, "METAR", columns=("station", "icao", "airport"))
        m, _, bad_m = _with_hour(m, spec, "METAR", preferred="valid")
        if "vsby" not in m.columns and "vsby_mi" in m.columns:
            m["vsby"] = m["vsby_mi"]
        if "vsby" not in m.columns:
            raise ImportProblem("METAR: нет колонки vsby (видимость в милях, как в выгрузке Iowa Mesonet).")
        metar_h = M.metar_frame_to_hourly(m)
        result.notes.append(f"METAR: {int(metar_h['n_metar'].sum())} сводок за {len(metar_h)} ч.")
    elif ("vsby_mi" in w.columns or "vsby" in w.columns) or "ceiling_ft" in w.columns:
        vs = w.get("vsby_mi", w.get("vsby", pd.Series(np.nan, index=w.index)))
        mt = pd.DataFrame({"vsby_mi": pd.to_numeric(vs, errors="coerce"),
                           "ceiling_ft": pd.to_numeric(w.get("ceiling_ft", np.nan), errors="coerce"),
                           "hour_utc": w["hour_utc"]}).drop_duplicates("hour_utc").set_index("hour_utc").sort_index()
        mt["n_metar"] = mt["vsby_mi"].notna().astype(int)
        metar_h = mt[["vsby_mi", "ceiling_ft", "n_metar"]]
    else:
        metar_h = M.metar_frame_to_hourly(None)
        result.warnings.append("METAR не указан: видимость и нижняя граница облаков не анализируются.")

    # --- рейсы
    f = read_table(spec.flights)
    f = _filter_airport(f, spec, "Рейсы", columns=("base_airport", "icao", "airport"))
    f, tcol_f, bad_f = _with_hour(f, spec, "Рейсы", preferred="scheduled_time_utc" if "scheduled_time_utc" in f.columns else None)
    if bad_f:
        result.warnings.append(f"Рейсы: {bad_f} строк с нечитаемым временем пропущено.")
    field_ = spec.delay_field
    if field_ not in f.columns:
        # задержку можно вычислить из плановой и фактической/расчётной времён
        sched = "scheduled_time_utc"
        actual = "revised_time_utc" if field_ == "delay_min" else "runway_time_utc"
        if sched in f.columns and actual in f.columns:
            t_s = pd.to_datetime(f[sched], errors="coerce", utc=True, format="mixed")
            t_a = pd.to_datetime(f[actual], errors="coerce", utc=True, format="mixed")
            f[field_] = (t_a - t_s).dt.total_seconds() / 60.0
            result.notes.append(f"Рейсы: задержка вычислена как {actual} − {sched}.")
        else:
            raise ImportProblem(f"Рейсы: нет колонки {field_} (задержка в минутах). Нужна она либо пара "
                                f"{sched} и {actual}. Есть колонки: {', '.join(f.columns[:14])}")
    if "is_cancelled" in f.columns:
        f["is_cancelled"] = f["is_cancelled"].map(
            lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() in TRUE_VALUES)
    elif "status" not in f.columns:
        result.warnings.append("Рейсы: нет ни is_cancelled, ни status - отмены определить нельзя (считаются нулём).")
        f["is_cancelled"] = False
    if spec.direction != "both" and "direction" in f.columns:
        f = f[f["direction"].astype(str).str.lower().str.startswith(spec.direction[:3])]
    if not spec.include_codeshare and "codeshare_status" in f.columns:
        f = f[f["codeshare_status"].astype(str) != "IsCodeshared"]
    if spec.exclude_cargo and "is_cargo" in f.columns:
        f = f[~f["is_cargo"].astype(str).str.lower().isin(TRUE_VALUES)]
    try:
        flights_h, statuses = M.flights_frame_to_hourly(f, field_)
    except ValueError as e:
        raise ImportProblem(f"Рейсы: {e}") from e
    result.notes.append(f"Рейсы: {len(f)} строк, из них отменено "
                        f"{int(flights_h['n_cancelled'].sum()) if len(flights_h) else 0}.")
    if statuses:
        result.notes.append("Статусы: " + ", ".join(f"{k}={v}" for k, v in statuses.most_common(6)))
    return weather_h, metar_h, flights_h


# ==============================================================================
# А) Готовый почасовой CSV
# ==============================================================================
def load_hourly_tables(spec, result):
    df = read_table(spec.hourly)
    df, tcol, bad = _with_hour(df, spec, "Почасовой CSV")
    if bad:
        result.warnings.append(f"Нечитаемое время в {bad} строках - они пропущены.")
    missing = [c for c in HOURLY_REQUIRED if c not in df.columns]
    if missing:
        raise ImportProblem("Почасовой CSV: не хватает колонок " + ", ".join(missing) + ". "
                            "Обязательные: " + ", ".join(HOURLY_REQUIRED) + " и хотя бы одна погодная колонка.")
    if not any(c in df.columns for c in M.OPENMETEO_FEATURES):
        raise ImportProblem("Почасовой CSV: нет погодных колонок (" + ", ".join(M.OPENMETEO_FEATURES) + ")")
    dup = int(df.duplicated("hour_utc").sum())
    if dup:
        result.warnings.append(f"{dup} дублей часов, оставлена первая строка.")
        df = df.drop_duplicates("hour_utc")
    df = df.set_index("hour_utc").sort_index()
    for c in set(HOURLY_REQUIRED + M.OPENMETEO_FEATURES + ["vsby_mi", "visibility_km", "ceiling_ft", "n_diverted",
                                                          "median_delay_min", "n_metar"]) & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "vsby_mi" not in df.columns and "vsby" in df.columns:
        df["vsby_mi"] = pd.to_numeric(df["vsby"], errors="coerce")
    if "vsby_mi" not in df.columns and "visibility_km" in df.columns:
        df["vsby_mi"] = df["visibility_km"] / M.MILES_TO_KM
    weather_h = df.reindex(columns=M.OPENMETEO_FEATURES).astype(float)
    metar_h = pd.DataFrame({"vsby_mi": df["vsby_mi"] if "vsby_mi" in df else np.nan,
                            "ceiling_ft": df["ceiling_ft"] if "ceiling_ft" in df else np.nan}, index=df.index)
    metar_h["n_metar"] = df["n_metar"] if "n_metar" in df else metar_h["vsby_mi"].notna().astype(int)
    if metar_h["n_metar"].sum() == 0:
        result.warnings.append("Нет видимости/потолка облаков: эти признаки не анализируются.")
    fl = pd.DataFrame(index=df.index)
    fl["n_flights"] = df["n_flights"]
    fl["n_completed"] = df["n_completed"]
    fl["n_cancelled"] = df["n_cancelled"]
    fl["n_diverted"] = df["n_diverted"] if "n_diverted" in df else 0
    fl["n_delayed15"] = df["n_delayed15"]
    fl["mean_delay_min"] = df["mean_delay_min"]
    fl["median_delay_min"] = df["median_delay_min"] if "median_delay_min" in df else df["mean_delay_min"]
    fl["cancel_rate"] = fl["n_cancelled"] / fl["n_flights"].where(fl["n_flights"] > 0)
    fl["delayed15_share"] = fl["n_delayed15"] / fl["n_completed"].where(fl["n_completed"] > 0)
    result.notes.append(f"Почасовой CSV: {len(df)} строк, {df.index.min()} .. {df.index.max()} UTC.")
    return weather_h, metar_h, fl


# ==============================================================================
# Сборка и проверка
# ==============================================================================
def build_dataset(spec, backend):
    """DataSpec -> ImportResult(ds, notes, warnings). backend нужен для очистки видимости (C++ или Python)."""
    result = ImportResult(ds=None)
    if spec.kind == "hourly":
        if not spec.hourly:
            raise ImportProblem("Укажите файл с почасовым датасетом.")
        weather_h, metar_h, flights_h = load_hourly_tables(spec, result)
    elif spec.kind == "raw":
        weather_h, metar_h, flights_h = load_raw_tables(spec, result)
    else:
        raise ImportProblem(f"Неизвестный тип источника: {spec.kind}")

    indexes = [t.index for t in (weather_h, flights_h) if len(t)]
    if not indexes:
        raise ImportProblem("В файлах нет ни одной строки с распознанным временем.")
    lo = min(i.min() for i in indexes)
    hi = max(i.max() for i in indexes)
    if spec.start:
        lo = max(lo, pd.Timestamp(spec.start))
    if spec.end:
        hi = min(hi, pd.Timestamp(spec.end) + timedelta(hours=23))
    if hi < lo:
        raise ImportProblem("Выбранный период не пересекается с данными.")
    t0, t1 = lo.to_pydatetime(), (hi + pd.Timedelta(hours=1)).to_pydatetime()
    ds = M.assemble_dataset(t0, t1, weather_h.loc[(weather_h.index >= lo) & (weather_h.index <= hi)],
                            metar_h.loc[(metar_h.index >= lo) & (metar_h.index <= hi)],
                            flights_h.loc[(flights_h.index >= lo) & (flights_h.index <= hi)], backend)
    result.ds = ds
    result.warnings += check_dataset(ds)
    return result


def check_dataset(ds, min_flights=5):
    """Проверка качества готового датасета: список предупреждений."""
    w = []
    n = len(ds)
    with_flights = int((ds["n_flights"] > 0).sum())
    if with_flights == 0:
        w.append("Нет ни одного часа с рейсами: проверьте время в файле рейсов и его часовой пояс.")
        return w
    if n > 2 * with_flights and n > 48:
        w.append(f"Рейсы есть только в {with_flights} из {n} часов: погода и рейсы могут относиться к разным периодам.")
    for f in M.OPENMETEO_FEATURES:
        share = float(ds[f].notna().mean()) if f in ds else 0.0
        if f in ds and share < 0.9:
            w.append(f"Признак {f} заполнен только в {100 * share:.0f}% часов.")
    no_weather = int(((ds["n_flights"] > 0) & ~ds[M.OPENMETEO_FEATURES].notna().any(axis=1)).sum())
    if no_weather:
        w.append(f"{no_weather} часов с рейсами без погоды - в анализ не попадут.")
    feats = M.available_features(ds)
    active = int(((ds["n_flights"] >= min_flights) & ds[feats].notna().all(axis=1) & ds["mean_delay_min"].notna()).sum()) if feats else 0
    if active < M.MIN_SAMPLE_HOURS:
        w.append(f"Подходящих часов для анализа всего {active} (нужно не менее {M.MIN_SAMPLE_HOURS}). "
                 f"Уменьшите «минимум рейсов в часе» или проверьте данные.")
    if ds["n_flights"].max() > 0 and (ds["n_completed"] > ds["n_flights"]).any():
        w.append("В некоторых часах выполнено больше рейсов, чем всего (n_completed > n_flights).")
    return w


def describe(res):
    """Текстовый отчёт о проверке данных."""
    ds = res.ds
    lines = list(res.notes)
    lines.append(f"Итог: {len(ds)} часов ({ds.index.min()} .. {ds.index.max()} UTC), с рейсами {int((ds['n_flights'] > 0).sum())}, "
                 f"рейсов {int(ds['n_flights'].sum())}, отменено {int(ds['n_cancelled'].sum())}.")
    feats = M.available_features(ds)
    lines.append("Признаки в анализе: " + (", ".join(feats) if feats else "нет"))
    for wmsg in res.warnings:
        lines.append("⚠ " + wmsg)
    return lines


# ==============================================================================
# Шаблоны и учебный пример (синтетические данные - помечены как таковые)
# ==============================================================================
def synthetic_example(days=45, start=date(2026, 6, 1), seed=7):
    """
    Правдоподобный синтетический пример в «исходных» форматах (погода, METAR, рейсы) с заложенной связью:
    порывы ветра и низкая видимость увеличивают задержки. НЕ реальные данные - только чтобы попробовать программу.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(datetime(start.year, start.month, start.day), periods=24 * days, freq="h", name="hour_utc")
    n = len(idx)
    hm = (idx.hour + 3) % 24
    gust = np.clip(rng.gamma(2.0, 8.0, n) + 4 * (np.sin(2 * np.pi * hm / 24) + 1), 0, 90)
    vis_mi = np.clip(rng.gamma(6, 1.0, n), 0.2, 6.21)
    precip = np.where(rng.random(n) < 0.12, rng.gamma(1.2, 1.0, n), 0.0)
    weather = pd.DataFrame({
        "time": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "temperature_2m": 14 + 6 * np.sin(2 * np.pi * np.arange(n) / 24 / 30) + rng.normal(0, 2, n),
        "relative_humidity_2m": rng.uniform(40, 100, n).round(), "precipitation": precip.round(2),
        "snowfall": 0.0, "wind_speed_10m": (gust * 0.55).round(1), "wind_gusts_10m": gust.round(1),
        "cloud_cover": rng.uniform(0, 100, n).round(), "airport": "Example"})
    weather["temperature_2m"] = weather["temperature_2m"].round(1)
    metar_rows = []
    for i, t in enumerate(idx):
        for minute in (0, 30):
            v = vis_mi[i] if rng.random() > 0.05 else None
            metar_rows.append({"station": "EXMP", "valid": (t + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M"),
                               "vsby": "M" if v is None else f"{v:.2f}", "skyc1": "OVC" if rng.random() < 0.3 else "M",
                               "skyl1": f"{rng.choice([500, 1500, 3000]):.2f}", "gust": "M"})
    metar = pd.DataFrame(metar_rows)
    metar.loc[metar["skyc1"] == "M", "skyl1"] = "M"
    load = np.clip(np.round(14 + 12 * np.sin(np.pi * np.clip(hm - 5, 0, 18) / 18) + rng.normal(0, 2, n)), 0, None).astype(int)
    stress = 0.35 * np.clip(gust - 25, 0, None) + 5 * np.clip(4 - vis_mi * 1.6, 0, None) + 3 * precip
    base_delay = 4 + 0.25 * load + stress
    rows = []
    for i, t in enumerate(idx):
        for _ in range(int(load[i])):
            sched = t + pd.Timedelta(minutes=int(rng.integers(0, 60)))
            canc = rng.random() < min(0.4, 0.004 + 0.002 * stress[i])
            delay = float(rng.normal(base_delay[i], 9))
            rows.append({"flight_uid": f"EX{len(rows)}", "base_airport": "EXMP",
                         "direction": "departure" if rng.random() < 0.5 else "arrival",
                         "status": "Canceled" if canc else "Departed",
                         "is_cancelled": canc, "codeshare_status": "IsOperator", "is_cargo": False,
                         "scheduled_time_utc": sched.strftime("%Y-%m-%d %H:%M:%S"),
                         "delay_min": "" if canc else round(delay, 1)})
    return weather, metar, pd.DataFrame(rows)


def write_examples(folder, days=45):
    """Создаёт в папке учебные CSV (все три исходные таблицы + почасовой) и подробную справку о форматах."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    weather, metar, flights = synthetic_example(days)
    weather.to_csv(folder / "example_weather.csv", index=False)
    metar.to_csv(folder / "example_metar.csv", index=False)
    flights.to_csv(folder / "example_flights.csv", index=False)
    spec = DataSpec(kind="raw", weather=str(folder / "example_weather.csv"), metar=str(folder / "example_metar.csv"),
                    flights=str(folder / "example_flights.csv"))
    res = build_dataset(spec, M.PythonBackend())
    cols = (M.OPENMETEO_FEATURES + ["vsby_mi", "ceiling_ft", "n_metar"] + M.FLIGHT_COUNT_COLUMNS +
            ["mean_delay_min", "median_delay_min"])
    res.ds[cols].to_csv(folder / "example_hourly.csv", float_format="%.6g")
    (folder / "ПРОЧТИ_МЕНЯ.txt").write_text(
        "Эти файлы - СИНТЕТИЧЕСКИЙ учебный пример (не реальные данные): с их помощью можно попробовать режим «Анализ».\n\n"
        "Вариант А (проще): example_hourly.csv - один файл, строка = час.\n"
        "Вариант Б: example_weather.csv + example_flights.csv (+ example_metar.csv) - исходные таблицы.\n\n"
        + (__doc__ or ""), encoding="utf-8")
    return sorted(p.name for p in folder.iterdir())
