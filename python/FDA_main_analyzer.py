# --- FDA_main_analyzer.py ---
"""
Головная программа анализа: влияние погоды на задержки и отмены рейсов
в Пулково (ULLI) и Шереметьево (UUEE).

Запуск:
    python FDA_main_analyzer.py                                  # оба аэропорта, окна 24 и 168 часов
    python FDA_main_analyzer.py --airport ULLI --windows 24,72   # конкретный аэропорт, свои окна
    python FDA_main_analyzer.py --backend python                 # без C++ библиотеки
    python FDA_main_analyzer.py --verify                         # сверить C++ и Python-эталон
"""
import argparse
import ctypes
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
AIRPORTS = {"ULLI": "Pulkovo", "UUEE": "Sheremetyevo"}

# Период исследования
PERIOD_START = date(2026, 5, 1)
PERIOD_END = date(2026, 10, 28)
MSK = timezone(timedelta(hours=3))

MONGO_DB_WEATHER = os.environ.get("MONGO_DB_WEATHER", "aviation_weather")
MONGO_DB_FLIGHTS = os.environ.get("MONGO_DB_FLIGHTS", "aviation_flights")
HOURLY_COLLECTION = "hourly_weather"
METAR_COLLECTION = "metar_weather"
FLIGHTS_COLLECTION = "flights"

OPENMETEO_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall",
    "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
]
# visibility_km - из METAR (очищается в C++), ceiling_ft - нижняя граница облаков
FEATURES = OPENMETEO_FEATURES + ["visibility_km", "ceiling_ft"]
TARGETS = ["mean_delay_min", "cancel_rate", "delayed15_share"]
FLIGHT_COUNT_COLUMNS = ["n_flights", "n_completed", "n_cancelled", "n_diverted", "n_delayed15"]

DELAY_CLIP = (-60.0, 360.0)
DELAY_THRESHOLD_MIN = 15.0
CEILING_COVER = ("BKN", "OVC", "VV")
CEILING_UNLIMITED_FT = 10000.0

MISSING_MARKER = -999.0
MILES_TO_KM = 1.60934
DEFAULT_VSBY_MI = 6.21
MIN_SAMPLE_HOURS = 30
PY_BLOCK_ELEMENTS = 2_000_000


def get_mongo_uri():
    """MONGO_URI целиком, либо MONGO_HOST (имя хоста), либо localhost - как в сборщиках."""
    uri = os.environ.get("MONGO_URI")
    if uri:
        return uri
    host = os.environ.get("MONGO_HOST", "localhost")
    return host if "://" in host else f"mongodb://{host}:27017/"


# ==============================================================================
# 2. C++ БИБЛИОТЕКА (СиВВ) И PYTHON-ЭТАЛОН
# ==============================================================================
class WeatherRecord(ctypes.Structure):
    """Точное описание struct WeatherRecord из aviation_analysis.h (6 x double, 48 байт)."""
    _fields_ = [
        ("temperature", ctypes.c_double),
        ("humidity", ctypes.c_double),
        ("precipitation", ctypes.c_double),
        ("wind_speed", ctypes.c_double),
        ("cloud_cover", ctypes.c_double),
        ("visibility", ctypes.c_double),
    ]


WEATHER_DTYPE = np.dtype([(name, "<f8") for name, _ in WeatherRecord._fields_])
assert WEATHER_DTYPE.itemsize == ctypes.sizeof(WeatherRecord), "раскладка WeatherRecord разошлась с C++"


def aligned_zeros(n, boundary=64, dtype=np.float64):
    """Массив из n нулей, адрес данных которого кратен boundary байт (для SIMD)."""
    itemsize = np.dtype(dtype).itemsize
    buf = np.zeros(n + boundary // itemsize, dtype=dtype)
    offset = (-buf.ctypes.data) % boundary
    assert offset % itemsize == 0
    start = offset // itemsize
    arr = buf[start:start + n]
    assert n == 0 or arr.ctypes.data % boundary == 0
    return arr


class LibraryError(RuntimeError):
    pass


def library_candidates(explicit=None):
    names = ["libaviation_analysis.so", "aviation_analysis.dll", "libaviation_analysis.dylib"]
    here = Path(__file__).resolve().parent
    roots = [here, here.parent.parent / "cpp" / "build", here.parent / "cpp" / "build",
             Path.cwd() / "cpp" / "build", Path("/app/cpp/build")]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        roots = [exe_dir, exe_dir / "lib", Path(getattr(sys, "_MEIPASS", exe_dir))] + roots
    roots.append(here / "lib")
    paths = []
    if explicit or os.environ.get("HPC_LIB"):
        paths.append(Path(explicit or os.environ["HPC_LIB"]))
    for root in roots:
        for sub in ("", "Release", "Debug"):
            for name in names:
                paths.append(root / sub / name)
    return paths


class HpcBackend:
    """Вызовы C++ библиотеки. Весь знание об ABI (структуры, типы, указатели) - только здесь."""
    name = "hpc"

    def __init__(self, lib_path=None):
        self.lib, self.path = None, None
        tried = library_candidates(lib_path)
        for path in tried:
            if not path.exists():
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(path.parent.resolve()))
                self.lib, self.path = ctypes.CDLL(str(path)), path
                break
            except OSError as e:
                print(f"[WARNING] Не удалось загрузить {path}: {e}")
        if self.lib is None:
            raise LibraryError("C++ библиотека не найдена (искали aviation_analysis в cpp/build и рядом). "
                               "Соберите: cd cpp && cmake -B build && cmake --build build "
                               "или укажите путь: --lib PATH / переменная HPC_LIB.")
        dp = ctypes.POINTER(ctypes.c_double)
        self.lib.preprocess_weather_vectors.argtypes = [dp, dp, ctypes.c_int, ctypes.c_double]
        self.lib.preprocess_weather_vectors.restype = None
        self.lib.compute_rolling_correlation.argtypes = [
            ctypes.POINTER(WeatherRecord), dp, ctypes.c_int, ctypes.c_int, dp]
        self.lib.compute_rolling_correlation.restype = None
        self.seconds, self.calls = 0.0, 0

    def preprocess_visibility(self, vsby_mi):
        """Мили -> км с заполнением пропусков (SIMD-функция библиотеки)."""
        values = np.asarray(vsby_mi, dtype=np.float64)
        n = len(values)
        raw, out = aligned_zeros(n), aligned_zeros(n)
        raw[:] = np.where(np.isfinite(values), values, MISSING_MARKER)
        dp = ctypes.POINTER(ctypes.c_double)
        t = time.perf_counter()
        self.lib.preprocess_weather_vectors(raw.ctypes.data_as(dp), out.ctypes.data_as(dp), n, MISSING_MARKER)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return np.asarray(out, dtype=np.float64)

    def rolling_correlation(self, x, y, window):
        """
        Скользящий коэффициент Пирсона corr(x, y) в окнах по window точек.
        Возвращает n - window + 1 значений (значение i относится к окну x[i:i+window]).

        ABI v1: C++ функция берёт признак из поля WeatherRecord.temperature, поэтому любой
        признак упаковывается в него. При переработке библиотеки (SoA, много признаков)
        изменится только этот метод.
        """
        xa = np.ascontiguousarray(x, dtype=np.float64)
        ya = np.ascontiguousarray(y, dtype=np.float64)
        n = len(xa)
        if n < window:
            return np.empty(0)
        if not (np.isfinite(xa).all() and np.isfinite(ya).all()):
            raise ValueError("NaN/Inf во входных данных: C++ собран с -ffast-math, пропуски надо убрать заранее")
        records = np.zeros(n, dtype=WEATHER_DTYPE)
        records["temperature"] = xa
        out = aligned_zeros(n - window + 1)
        dp = ctypes.POINTER(ctypes.c_double)
        t = time.perf_counter()
        self.lib.compute_rolling_correlation(records.ctypes.data_as(ctypes.POINTER(WeatherRecord)),
                                             ya.ctypes.data_as(dp), n, window, out.ctypes.data_as(dp))
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return np.asarray(out, dtype=np.float64)


class PythonBackend:
    """Эталон на numpy: те же формулы и пороги, что в C++ (для --backend python и --verify)."""
    name = "python"
    path = None

    def __init__(self):
        self.seconds, self.calls = 0.0, 0

    def preprocess_visibility(self, vsby_mi):
        t = time.perf_counter()
        values = np.asarray(vsby_mi, dtype=np.float64)
        n = len(values)
        missing = ~np.isfinite(values) | (values == MISSING_MARKER)
        clean = np.where(missing, -1.0, values * MILES_TO_KM)
        next_valid = np.full(n, np.nan)
        nxt = np.nan
        for i in range(n - 1, -1, -1):
            next_valid[i] = nxt
            if not missing[i]:
                nxt = values[i] * MILES_TO_KM
        for i in range(n):
            if clean[i] < 0.0:
                prev = clean[i - 1] if i > 0 else DEFAULT_VSBY_MI * MILES_TO_KM
                nxt_i = next_valid[i] if np.isfinite(next_valid[i]) else prev
                clean[i] = (prev + nxt_i) * 0.5
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return clean

    def rolling_correlation(self, x, y, window):
        xa = np.ascontiguousarray(x, dtype=np.float64)
        ya = np.ascontiguousarray(y, dtype=np.float64)
        n = len(xa)
        if n < window:
            return np.empty(0)
        t = time.perf_counter()
        sw = np.lib.stride_tricks.sliding_window_view
        n_win = n - window + 1
        r = np.empty(n_win)
        block = max(1, PY_BLOCK_ELEMENTS // window)
        for start in range(0, n_win, block):
            stop = min(start + block, n_win)
            wx, wy = sw(xa[start:stop + window - 1], window), sw(ya[start:stop + window - 1], window)
            sx, sy = wx.sum(axis=1), wy.sum(axis=1)
            sxx, syy, sxy = (wx * wx).sum(axis=1), (wy * wy).sum(axis=1), (wx * wy).sum(axis=1)
            num = window * sxy - sx * sy
            den_sq = (window * sxx - sx * sx) * (window * syy - sy * sy)
            flat = den_sq <= 1e-9
            r[start:stop] = np.where(flat, 0.0, num / np.sqrt(np.where(flat, 1.0, den_sq)))
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return r


def select_backends(args):
    """Возвращает (основной бэкенд, эталон для --verify или None)."""
    if args.backend == "python":
        if args.verify:
            print("[WARNING] --verify при --backend python нечего сверять, флаг игнорируется")
        return PythonBackend(), None
    try:
        hpc = HpcBackend(args.lib)
    except LibraryError as e:
        if args.backend == "hpc":
            print(f"[ERROR] {e}")
            raise SystemExit(1)
        print(f"[WARNING] {e}\n          Продолжаю на Python-реализации (--backend python).")
        return PythonBackend(), None
    print(f"[INFO] Загружена HPC-библиотека: {hpc.path}")
    return hpc, (PythonBackend() if args.verify else None)


# ==============================================================================
# 3. ЗАГРУЗКА ДАННЫХ ИЗ MONGODB И ПОЧАСОВОЙ ДАТАСЕТ
# ==============================================================================
def utc_bounds(start, end):
    """Период по Москве (даты включительно) -> [t0, t1) в naive UTC (как хранится в MongoDB)."""
    def to_utc(d):
        return datetime.combine(d, datetime.min.time(), tzinfo=MSK).astimezone(timezone.utc).replace(tzinfo=None)
    return to_utc(start), to_utc(end + timedelta(days=1))


def _hour_index(df, column="hour_utc"):
    df[column] = pd.to_datetime(df[column])
    return df.set_index(column).sort_index()


def _empty(columns):
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="hour_utc"), dtype=float)


def weather_frame_to_hourly(df):
    """Open-Meteo (DataFrame с колонкой hour_utc и признаками) -> почасовые значения (индекс hour_utc)."""
    if df is None or len(df) == 0:
        return _empty(OPENMETEO_FEATURES)
    df = _hour_index(df.drop_duplicates("hour_utc").copy())
    df = df.reindex(columns=OPENMETEO_FEATURES)
    return df.apply(pd.to_numeric, errors="coerce")


def load_weather_hourly(db, icao, t0, t1):
    """Open-Meteo из MongoDB: почасовые значения (индекс hour_utc)."""
    projection = {"_id": 0, "hour_utc": 1, **{c: 1 for c in OPENMETEO_FEATURES}}
    docs = list(db[HOURLY_COLLECTION].find({"icao": icao, "hour_utc": {"$gte": t0, "$lt": t1}}, projection))
    return weather_frame_to_hourly(pd.DataFrame(docs))


def metar_frame_to_hourly(df):
    """
    METAR (DataFrame: hour_utc, vsby, skyc1-3, skyl1-3) -> почасовые агрегаты. В час бывает несколько
    наблюдений (METAR + SPECI), берём наихудшее для полётов: минимальную видимость и самый низкий потолок.
    """
    columns = ["vsby_mi", "ceiling_ft", "n_metar"]
    layers = [f"sky{k}{i}" for i in (1, 2, 3) for k in ("c", "l")]
    if df is None or len(df) == 0:
        return _empty(columns)
    df = df.reindex(columns=["hour_utc", "vsby"] + layers)
    df["vsby"] = pd.to_numeric(df["vsby"], errors="coerce")
    for i in (1, 2, 3):
        height = pd.to_numeric(df[f"skyl{i}"], errors="coerce")
        df[f"_ceil{i}"] = height.where(df[f"skyc{i}"].isin(CEILING_COVER))
    df["_ceiling"] = df[["_ceil1", "_ceil2", "_ceil3"]].min(axis=1)
    df["hour_utc"] = pd.to_datetime(df["hour_utc"])
    g = df.groupby("hour_utc")
    out = pd.DataFrame({"vsby_mi": g["vsby"].min(), "ceiling_ft": g["_ceiling"].min(), "n_metar": g.size()})
    out["ceiling_ft"] = out["ceiling_ft"].fillna(CEILING_UNLIMITED_FT)
    return out[columns]


def load_metar_hourly(db, icao, t0, t1):
    """METAR из MongoDB -> почасовые агрегаты."""
    layers = [f"sky{k}{i}" for i in (1, 2, 3) for k in ("c", "l")]
    projection = {"_id": 0, "hour_utc": 1, "vsby": 1, **{c: 1 for c in layers}}
    docs = list(db[METAR_COLLECTION].find({"station": icao, "hour_utc": {"$gte": t0, "$lt": t1}}, projection))
    return metar_frame_to_hourly(pd.DataFrame(docs))


def flights_frame_to_hourly(df, delay_field="delay_min"):
    """
    Рейсы (DataFrame: hour_utc, status, is_cancelled, <delay_field>) -> почасовые счётчики и задержки.
    Возвращает (DataFrame, Counter статусов). Ошибка в данных -> ValueError.
    """
    columns = FLIGHT_COUNT_COLUMNS + ["mean_delay_min", "median_delay_min", "cancel_rate", "delayed15_share"]
    if df is None or len(df) == 0:
        return _empty(columns), Counter()
    if delay_field not in df.columns:
        raise ValueError(f"В данных о рейсах нет поля {delay_field}")
    df = df.copy()
    if "status" not in df.columns:
        df["status"] = None
    if "is_cancelled" not in df.columns:
        df["is_cancelled"] = df["status"].eq("Canceled")
    statuses = Counter(df["status"].fillna("None"))

    cancelled = df["is_cancelled"].fillna(False).astype(bool)
    diverted = df["status"].eq("Diverted")
    delay = pd.to_numeric(df[delay_field], errors="coerce")
    completed = delay.notna() & ~cancelled & ~diverted
    delay = delay.where(completed).clip(*DELAY_CLIP)
    df = pd.DataFrame({
        "hour_utc": pd.to_datetime(df["hour_utc"]),
        "_cancelled": cancelled.astype(int), "_diverted": diverted.astype(int),
        "_completed": completed.astype(int), "_delay": delay,
        "_delayed15": (completed & (delay > DELAY_THRESHOLD_MIN)).astype(int),
    })
    g = df.groupby("hour_utc")
    out = pd.DataFrame({
        "n_flights": g.size(), "n_completed": g["_completed"].sum(),
        "n_cancelled": g["_cancelled"].sum(), "n_diverted": g["_diverted"].sum(),
        "n_delayed15": g["_delayed15"].sum(),
        "mean_delay_min": g["_delay"].mean(), "median_delay_min": g["_delay"].median(),
    })
    out["cancel_rate"] = out["n_cancelled"] / out["n_flights"]
    out["delayed15_share"] = out["n_delayed15"] / out["n_completed"].where(out["n_completed"] > 0)
    return out[columns], statuses


def load_flights_hourly(db, icao, t0, t1, args):
    """Рейсы из MongoDB -> почасовые счётчики и задержки. Возвращает (DataFrame, Counter статусов)."""
    columns = FLIGHT_COUNT_COLUMNS + ["mean_delay_min", "median_delay_min", "cancel_rate", "delayed15_share"]
    query = {"base_airport": icao, "scheduled_time_utc": {"$gte": t0, "$lt": t1}}
    if args.direction != "both":
        query["direction"] = args.direction
    if not args.include_codeshare:
        query["codeshare_status"] = {"$ne": "IsCodeshared"}
    if args.exclude_cargo:
        query["is_cargo"] = {"$ne": True}
    projection = {"_id": 0, "hour_utc": 1, "status": 1, "is_cancelled": 1, args.delay_field: 1}
    docs = list(db[FLIGHTS_COLLECTION].find(query, projection))
    if not docs:
        return _empty(columns), Counter()
    try:
        return flights_frame_to_hourly(pd.DataFrame(docs), args.delay_field)
    except ValueError:
        print(f"[ERROR] В коллекции flights нет поля {args.delay_field}. Пересоберите рейсы из кэша: "
              f"python FDA_flights_data_collection.py --import-only --reset-flights")
        raise SystemExit(1)


def build_dataset(client, icao, t0, t1, args, backend):
    """Почасовой датасет: погода + METAR + рейсы на сетке всех часов периода (UTC)."""
    weather_db, flights_db = client[MONGO_DB_WEATHER], client[MONGO_DB_FLIGHTS]
    flights, statuses = load_flights_hourly(flights_db, icao, t0, t1, args)
    ds = assemble_dataset(t0, t1, load_weather_hourly(weather_db, icao, t0, t1),
                          load_metar_hourly(weather_db, icao, t0, t1), flights, backend)
    return ds, statuses


def assemble_dataset(t0, t1, weather, metar, flights, backend):
    """
    Склейка почасовых таблиц погоды, METAR и рейсов на сетке всех часов [t0, t1) (UTC).
    Одинаково для MongoDB и для CSV-файлов пользователя.
    """
    grid = pd.date_range(t0, t1 - timedelta(hours=1), freq=pd.Timedelta(hours=1), name="hour_utc")
    ds = pd.DataFrame(index=grid).join(weather).join(metar).join(flights)
    for col in FLIGHT_COUNT_COLUMNS + ["n_metar"]:
        ds[col] = ds[col].fillna(0).astype(int)

    if (ds["n_metar"] > 0).any():
        # Видимость: мили -> км и заполнение пропусков в C++ (по всей сетке часов).
        ds["visibility_km"] = backend.preprocess_visibility(ds["vsby_mi"].to_numpy(dtype=np.float64))
        ds["ceiling_ft"] = ds["ceiling_ft"].ffill().bfill()  # часы без METAR: ближайшее известное значение
    else:
        # METAR нет совсем: заполнять нечем, признаки видимости и потолка исключаются из анализа
        ds["visibility_km"] = np.nan
        ds["ceiling_ft"] = np.nan
    ds["metar_imputed"] = ds["vsby_mi"].isna()
    return ds


def report_dataset(icao, ds, statuses, t0, t1):
    n = len(ds)
    with_weather = int(ds[OPENMETEO_FEATURES].notna().all(axis=1).sum())
    with_metar = int((ds["n_metar"] > 0).sum())
    with_flights = int((ds["n_flights"] > 0).sum())
    total = int(ds["n_flights"].sum())
    cancelled, diverted = int(ds["n_cancelled"].sum()), int(ds["n_diverted"].sum())
    completed = int(ds["n_completed"].sum())
    delayed = int(ds["n_delayed15"].sum())
    print(f"[DATA] {icao}: {t0} .. {t1} UTC, часов {n}: с Open-Meteo {with_weather}, "
          f"с METAR {with_metar}, с рейсами {with_flights}")
    if total:
        print(f"[DATA] {icao}: рейсов {total}, выполнено с задержкой в данных {completed}, "
              f"отменено {cancelled} ({100 * cancelled / total:.1f}%), снято с маршрута {diverted}, "
              f"задержка >{DELAY_THRESHOLD_MIN:.0f} мин: {100 * delayed / max(completed, 1):.1f}%")
        print(f"[DATA] {icao}: статусы: " + ", ".join(f"{k}={v}" for k, v in statuses.most_common()))
    no_weather = int(((ds["n_flights"] > 0) & ~ds[OPENMETEO_FEATURES].notna().all(axis=1)).sum())
    if no_weather:
        print(f"[CHECK] {icao}: {no_weather} ч с рейсами без погоды Open-Meteo (хвост архива) - "
              f"в анализ не попадут; догрузите погоду позже")
    if with_flights and ds["metar_imputed"].sum():
        print(f"[CHECK] {icao}: часов без видимости METAR {int(ds['metar_imputed'].sum())} - "
              f"заполнены в C++ (колонка metar_imputed)")


# ==============================================================================
# 4. АНАЛИЗ: СТАТИЧЕСКИЕ И СКОЛЬЗЯЩИЕ КОРРЕЛЯЦИИ
# ==============================================================================
def available_features(ds):
    """Признаки, по которым в датасете вообще есть данные (нет METAR -> без видимости и потолка)."""
    return [f for f in FEATURES if f in ds.columns and ds[f].notna().any()]


def select_rows(ds, target, min_flights, features=None):
    """Активные часы: достаточно рейсов, определены цель и все признаки."""
    features = list(features) if features is not None else FEATURES
    mask = (ds["n_flights"] >= min_flights) & ds[target].notna() & ds[features].notna().all(axis=1)
    return ds.loc[mask]


def static_correlations(sel, target, features=None):
    rows = []
    y = sel[target]
    for feature in (features if features is not None else FEATURES):
        x = sel[feature]
        row = {"target": target, "feature": feature, "n": len(sel),
               "pearson": np.nan, "spearman": np.nan, "note": ""}
        if x.nunique() < 2 or y.nunique() < 2:
            row["note"] = "константа"
        else:
            row["pearson"] = x.corr(y)
            row["spearman"] = x.rank().corr(y.rank())
        rows.append(row)
    return rows


def analyze_airport(icao, ds, backend, reference, args, outdir):
    static_rows, summary_rows = [], []
    verify = {"calls": 0, "max_abs_diff": 0.0, "mismatch": 0}
    used_features = available_features(ds)
    skipped = [f for f in FEATURES if f not in used_features]
    if skipped:
        print(f"[CHECK] {icao}: нет данных по признакам {', '.join(skipped)} - они исключены из анализа")
    for target in TARGETS:
        sel = select_rows(ds, target, args.min_flights, used_features)
        total_hours = int((ds["n_flights"] >= args.min_flights).sum())
        print(f"[HPC] {icao} {target}: часов в выборке {len(sel)} из {total_hours} активных")
        if len(sel) < MIN_SAMPLE_HOURS:
            print(f"[WARNING] {icao} {target}: меньше {MIN_SAMPLE_HOURS} часов, пропуск")
            continue
        static_rows += static_correlations(sel, target, used_features)
        features = [f for f in used_features if sel[f].nunique() > 1]
        constant = [f for f in used_features if f not in features]
        if constant:
            print(f"[CHECK] {icao} {target}: признаки-константы пропущены: {', '.join(constant)}")
        y = sel[target].to_numpy(dtype=np.float64)

        for window in args.windows:
            if len(sel) < window + 5:
                print(f"[WARNING] {icao} {target}: окно {window} ч длиннее выборки, пропуск")
                continue
            series = {}
            for feature in features:
                x = sel[feature].to_numpy(dtype=np.float64)
                r = backend.rolling_correlation(x, y, window)
                if reference is not None:
                    diff = np.abs(r - reference.rolling_correlation(x, y, window))
                    verify["calls"] += 1
                    verify["max_abs_diff"] = max(verify["max_abs_diff"], float(diff.max()))
                    verify["mismatch"] += int((diff > 1e-6).sum())
                series[feature] = r.tolist()
                summary_rows.append({
                    "target": target, "window_h": window, "feature": feature, "windows": len(r),
                    "mean_r": r.mean(), "median_r": np.median(r), "std_r": r.std(),
                    "min_r": r.min(), "max_r": r.max(),
                    "share_abs_r_ge_0_3": float((np.abs(r) >= 0.3).mean()),
                })
            wide = pd.DataFrame(series, index=sel.index[window - 1:])
            wide.index.name = "window_end_utc"
            wide.to_csv(outdir / f"rolling_{icao}_w{window}_{target}.csv", float_format="%.6f")

    static_df, summary_df = pd.DataFrame(static_rows), pd.DataFrame(summary_rows)
    static_df.to_csv(outdir / f"correlations_static_{icao}.csv", index=False, float_format="%.6f")
    summary_df.to_csv(outdir / f"rolling_summary_{icao}.csv", index=False, float_format="%.6f")
    return static_df, summary_df, verify


def print_results(icao, static_df, summary_df, windows):
    if static_df.empty:
        return
    fmt = lambda v: f"{v:+.3f}"
    print(f"\n=== {icao}: корреляция Пирсона за весь период (по активным часам) ===")
    print(static_df.pivot(index="feature", columns="target", values="pearson")
          .reindex(FEATURES)[[t for t in TARGETS if t in static_df["target"].values]]
          .to_string(float_format=fmt, na_rep="  -  "))
    for window in windows:
        part = summary_df[summary_df["window_h"] == window] if not summary_df.empty else summary_df
        if part.empty:
            continue
        print(f"\n=== {icao}: средняя скользящая корреляция, окно {window} ч ===")
        print(part.pivot(index="feature", columns="target", values="mean_r")
              .reindex(FEATURES)[[t for t in TARGETS if t in part["target"].values]]
              .to_string(float_format=fmt, na_rep="  -  "))


# ==============================================================================
# 5. ТОЧКА ВХОДА
# ==============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Влияние погоды на задержки и отмены рейсов (MongoDB + C++/OpenMP)")
    p.add_argument("--airport", choices=["all"] + list(AIRPORTS), default="all")
    p.add_argument("--start", type=date.fromisoformat, default=PERIOD_START, help="первый день (MSK)")
    p.add_argument("--end", type=date.fromisoformat, default=PERIOD_END, help="последний день (MSK)")
    p.add_argument("--windows", default="24,168", help="размеры скользящего окна в часах через запятую")
    p.add_argument("--min-flights", type=int, default=5, help="минимум рейсов в часе, чтобы час участвовал")
    p.add_argument("--direction", choices=["both", "departure", "arrival"], default="both")
    p.add_argument("--delay-field", choices=["delay_min", "runway_delay_min"], default="delay_min",
                   help="delay_min: revised-scheduled (у гейта); runway_delay_min: взлёт/посадка")
    p.add_argument("--include-codeshare", action="store_true", help="не исключать codeshare-рейсы")
    p.add_argument("--exclude-cargo", action="store_true", help="исключить грузовые рейсы")
    p.add_argument("--backend", choices=["auto", "hpc", "python"], default="auto")
    p.add_argument("--lib", default=None, help="путь к libaviation_analysis.so / aviation_analysis.dll")
    p.add_argument("--verify", action="store_true", help="сверить C++ с Python-эталоном")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args(argv)
    try:
        args.windows = sorted({int(w) for w in args.windows.split(",") if w.strip()})
    except ValueError:
        p.error("--windows: ожидается список целых чисел, например 24,168")
    if not args.windows or min(args.windows) < 2:
        p.error("--windows: окно должно быть не меньше 2 часов")
    if args.end < args.start:
        p.error("--end раньше --start")
    return args


def main(argv=None):
    args = parse_args(argv)
    print("=" * 70)
    print(" АНАЛИЗ ПОГОДЫ И ЗАДЕРЖЕК РЕЙСОВ (MongoDB + C++ HPC через ctypes)")
    print("=" * 70)

    backend, reference = select_backends(args)
    try:
        from pymongo import MongoClient  # нужен только для режима MongoDB
    except ImportError:
        print("[ERROR] Для чтения из MongoDB нужен пакет pymongo (pip install pymongo). "
              "Для своих CSV-файлов используйте fda_app.py (режим analyze).")
        return 1
    client = MongoClient(get_mongo_uri(), serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as e:  # noqa: BLE001 - любая ошибка подключения должна давать понятное сообщение
        print(f"[ERROR] MongoDB недоступна ({get_mongo_uri()}): {e}")
        return 1

    t0, t1 = utc_bounds(args.start, args.end)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    airports = list(AIRPORTS) if args.airport == "all" else [args.airport]
    info = {"started_utc": datetime.now(timezone.utc).isoformat(), "backend": backend.name,
            "library": str(backend.path) if backend.path else None,
            "args": {k: (str(v) if isinstance(v, date) else v) for k, v in vars(args).items()},
            "period_utc": [str(t0), str(t1)], "airports": {}}
    verify_total = {"calls": 0, "max_abs_diff": 0.0, "mismatch": 0}
    done = 0

    for icao in airports:
        print(f"\n--- {icao} ({AIRPORTS[icao]}) ---")
        ds, statuses = build_dataset(client, icao, t0, t1, args, backend)
        report_dataset(icao, ds, statuses, t0, t1)
        if int((ds["n_flights"] >= args.min_flights).sum()) < MIN_SAMPLE_HOURS:
            print(f"[ERROR] {icao}: нет данных о рейсах за период. Запустите FDA_flights_data_collection.py "
                  f"(и FDA_weather_data_collection.py для погоды).")
            continue
        columns = (OPENMETEO_FEATURES + ["vsby_mi", "visibility_km", "ceiling_ft", "n_metar", "metar_imputed"]
                   + FLIGHT_COUNT_COLUMNS + ["mean_delay_min", "median_delay_min", "delayed15_share", "cancel_rate"])
        ds[columns].to_csv(outdir / f"hourly_dataset_{icao}.csv", float_format="%.6g")

        static_df, summary_df, verify = analyze_airport(icao, ds, backend, reference, args, outdir)
        print_results(icao, static_df, summary_df, args.windows)
        for key in ("calls", "mismatch"):
            verify_total[key] += verify[key]
        verify_total["max_abs_diff"] = max(verify_total["max_abs_diff"], verify["max_abs_diff"])
        info["airports"][icao] = {"hours": len(ds), "flights": int(ds["n_flights"].sum()),
                                  "cancelled": int(ds["n_cancelled"].sum()),
                                  "used_hours": {t: len(select_rows(ds, t, args.min_flights, available_features(ds))) for t in TARGETS}}
        done += 1

    if reference is not None and verify_total["calls"]:
        ok = verify_total["mismatch"] == 0
        print(f"\n[VERIFY] C++ vs Python: {verify_total['calls']} расчётов, макс. расхождение "
              f"{verify_total['max_abs_diff']:.2e}, значений >1e-6: {verify_total['mismatch']} -> "
              f"{'OK' if ok else 'РАСХОЖДЕНИЕ'}")
    info["calls"], info["seconds"] = backend.calls, round(backend.seconds, 6)
    info["verify"] = verify_total if reference is not None else None
    with open(outdir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\n[HPC] Бэкенд {backend.name}: {backend.calls} вызовов, {backend.seconds * 1000:.1f} мс чистого счёта")
    print(f"[OK] Результаты: {outdir.resolve()}" if done else "[ERROR] Нечего анализировать")
    client.close()
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
