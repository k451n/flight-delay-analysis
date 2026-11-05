# --- FDA_results.py ---
"""
Общий слой чтения и пересчёта результатов FDA_main_analyzer.py.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# 1. Справочники
# ==============================================================================
AIRPORTS = {"ULLI": "Пулково", "UUEE": "Шереметьево"}
AIRPORT_COLORS = {"ULLI": "#2a6fb0", "UUEE": "#d9622b"}
# Для своих наборов данных (любой идентификатор): цвета по порядку появления
PALETTE = ["#2a6fb0", "#d9622b", "#3a9a5b", "#8e5bb5", "#b8860b", "#c2476b"]
_dynamic_colors = {}


def airport_name(ds_id):
    """Название набора данных: аэропорты проекта - по-русски, свои наборы - как названы."""
    return AIRPORTS.get(ds_id, ds_id)


def color(ds_id):
    """Устойчивый цвет набора данных."""
    if ds_id in AIRPORT_COLORS:
        return AIRPORT_COLORS[ds_id]
    if ds_id not in _dynamic_colors:
        _dynamic_colors[ds_id] = PALETTE[(2 + len(_dynamic_colors)) % len(PALETTE)]
    return _dynamic_colors[ds_id]

FEATURE_LABELS = {
    "temperature_2m": "Температура, °C",
    "relative_humidity_2m": "Влажность, %",
    "precipitation": "Осадки, мм/ч",
    "snowfall": "Снегопад, см/ч",
    "wind_speed_10m": "Скорость ветра, км/ч",
    "wind_gusts_10m": "Порывы ветра, км/ч",
    "cloud_cover": "Облачность, %",
    "visibility_km": "Видимость, км",
    "ceiling_ft": "Нижняя граница облаков, фут",
}
FEATURES = list(FEATURE_LABELS)

TARGET_LABELS = {
    "mean_delay_min": "Средняя задержка, мин",
    "delayed15_share": "Доля задержанных >15 мин",
    "cancel_rate": "Доля отменённых",
}
TARGETS = list(TARGET_LABELS)
# (числитель, знаменатель) для долей и колонка-вес для среднего
TARGET_WEIGHT = {"mean_delay_min": "n_completed", "delayed15_share": "n_completed", "cancel_rate": "n_flights"}
TARGET_COUNTS = {"delayed15_share": ("n_delayed15", "n_completed"), "cancel_rate": ("n_cancelled", "n_flights")}
RATE_TARGETS = tuple(TARGET_COUNTS)

# Погодные условия-флаги: имя -> (подпись, условие над почасовым датасетом)
WEATHER_FLAGS = {
    "gust_40": ("Порывы ≥ 40 км/ч", lambda d: d["wind_gusts_10m"] >= 40),
    "gust_60": ("Порывы ≥ 60 км/ч", lambda d: d["wind_gusts_10m"] >= 60),
    "vis_5": ("Видимость < 5 км", lambda d: d["visibility_km"] < 5),
    "vis_1": ("Видимость < 1 км", lambda d: d["visibility_km"] < 1),
    "ceiling_low": ("Облака ниже 600 фут", lambda d: d["ceiling_ft"] < 600),
    "precip": ("Осадки ≥ 0.1 мм/ч", lambda d: d["precipitation"] >= 0.1),
    "snow": ("Снег", lambda d: d["snowfall"] > 0),
    "frost": ("Мороз (< 0 °C)", lambda d: d["temperature_2m"] < 0),
}

MSK_OFFSET = pd.Timedelta(hours=3)
DEFAULT_MIN_FLIGHTS = 5
CORR_STRONG = 0.3


# ==============================================================================
# 2. Поиск и чтение results/
# ==============================================================================
def find_results_dir(explicit=None):
    """Явный путь -> FDA_RESULTS_DIR -> results/ рядом со скриптом -> в текущей папке -> в родительских."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("FDA_RESULTS_DIR"):
        candidates.append(Path(os.environ["FDA_RESULTS_DIR"]))
    here = Path(__file__).resolve().parent
    candidates += [here / "results", Path.cwd() / "results", here.parent / "results"]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0] if explicit else here / "results"


def airports_available(results_dir):
    """Наборы данных в папке: проектные аэропорты первыми, остальные (свои) по алфавиту."""
    d = Path(results_dir)
    if not d.is_dir():
        return []
    ids = {p.name[len("hourly_dataset_"):-len(".csv")] for p in d.glob("hourly_dataset_*.csv")}
    return [i for i in AIRPORTS if i in ids] + sorted(ids - set(AIRPORTS))


def results_signature(results_dir):
    """(имя, размер, mtime) всех csv/json: ключ кэша Streamlit - после нового запуска main.py данные перечитаются."""
    d = Path(results_dir)
    if not d.is_dir():
        return ()
    return tuple(sorted((p.name, p.stat().st_size, int(p.stat().st_mtime))
                        for p in d.iterdir() if p.suffix in (".csv", ".json")))


def load_run_info(results_dir):
    p = Path(results_dir) / "run_info.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_hourly(results_dir, icao):
    p = Path(results_dir) / f"hourly_dataset_{icao}.csv"
    if not p.is_file():
        return None
    ds = pd.read_csv(p, parse_dates=["hour_utc"], index_col="hour_utc")
    if "metar_imputed" in ds:
        ds["metar_imputed"] = ds["metar_imputed"].astype(bool)
    return ds


def load_static(results_dir, icao):
    p = Path(results_dir) / f"correlations_static_{icao}.csv"
    return pd.read_csv(p) if p.is_file() else None


def load_summary(results_dir, icao):
    p = Path(results_dir) / f"rolling_summary_{icao}.csv"
    return pd.read_csv(p) if p.is_file() else None


def load_rolling(results_dir, icao, window, target):
    p = Path(results_dir) / f"rolling_{icao}_w{window}_{target}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p, parse_dates=["window_end_utc"], index_col="window_end_utc")


def rolling_windows_available(results_dir, icao=None):
    """Размеры окон, для которых main.py записал ряды (по именам файлов)."""
    pat = re.compile(r"rolling_(.+?)_w(\d+)_(?:mean_delay_min|cancel_rate|delayed15_share)\.csv$")
    found = set()
    for p in Path(results_dir).glob("rolling_*_w*_*.csv"):
        m = pat.match(p.name)
        if m and (icao is None or m.group(1) == icao):
            found.add(int(m.group(2)))
    return sorted(found)


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


# ==============================================================================
# 3. Выборка часов и аномалии
# ==============================================================================
def to_msk(ds):
    """Копия с колонками hour_msk (0-23), dow (0=пн) и датой по МСК."""
    out = ds.copy()
    msk = out.index + MSK_OFFSET
    out["hour_msk"] = msk.hour
    out["dow"] = msk.dayofweek
    out["date_msk"] = msk.floor("D")
    return out


def select_hours(ds, target, min_flights=DEFAULT_MIN_FLIGHTS, features=FEATURES, date_from=None, date_to=None):
    """Активные часы - то же правило, что select_rows() в main.py, плюс необязательный период (даты MSK)."""
    d = to_msk(ds)
    cols = [f for f in features if f in d and d[f].notna().any()]  # признак без данных (нет METAR) не отсекает часы
    mask = (d["n_flights"] >= min_flights) & d[target].notna() & d[cols].notna().all(axis=1)
    if date_from is not None:
        mask &= d["date_msk"] >= pd.Timestamp(date_from)
    if date_to is not None:
        mask &= d["date_msk"] <= pd.Timestamp(date_to)
    return d.loc[mask]


def hour_anomaly(values, hour_msk):
    """Значение минус среднее для того же часа суток: убирает суточный цикл."""
    return values - values.groupby(hour_msk).transform("mean")


def _corr(x, y, method):
    if len(x) < 10 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    if method == "spearman":
        x, y = x.rank(), y.rank()
    return float(np.corrcoef(x.to_numpy(float), y.to_numpy(float))[0, 1])


def block_bootstrap_ci(x, y, method="pearson", n_boot=300, seed=0):
    """
    95% интервал корреляции блочным бутстрепом: пересэмплируются целые сутки (МСК), а не часы,
    чтобы учесть автокорреляцию соседних часов. Возвращает (lo, hi) или (nan, nan).
    """
    days = x.index.floor("D")
    uniq = np.unique(days.to_numpy())
    if len(x) < 48 or len(uniq) < 8:
        return np.nan, np.nan
    xv, yv = (x.rank().to_numpy(float), y.rank().to_numpy(float)) if method == "spearman" \
        else (x.to_numpy(float), y.to_numpy(float))
    groups = [np.flatnonzero(days.to_numpy() == u) for u in uniq]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([groups[k] for k in pick])
        a, b = xv[ix], yv[ix]
        if a.std() > 0 and b.std() > 0:
            stats.append(np.corrcoef(a, b)[0, 1])
    if len(stats) < 30:
        return np.nan, np.nan
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def correlation_table(ds, target, min_flights=DEFAULT_MIN_FLIGHTS, method="pearson", date_from=None, date_to=None,
                      ci=True, n_boot=300):
    """
    Для каждого признака: r по сырым значениям (как в main.py), r по аномалиям (без суточного цикла)
    и интервал для аномалий. Один вызов - один аэропорт и одна метрика.
    """
    sel = select_hours(ds, target, min_flights, date_from=date_from, date_to=date_to)
    rows = []
    for f in FEATURES:
        if f not in sel:
            continue
        x, y = sel[f], sel[target]
        xa, ya = hour_anomaly(x, sel["hour_msk"]), hour_anomaly(y, sel["hour_msk"])
        lo, hi = block_bootstrap_ci(xa, ya, method, n_boot) if ci else (np.nan, np.nan)
        rows.append({"feature": f, "n_hours": len(sel), "r_raw": _corr(x, y, method),
                     "r_anomaly": _corr(xa, ya, method), "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out


def lag_correlation(ds, feature, target, lags=range(-6, 13), min_flights=DEFAULT_MIN_FLIGHTS, method="pearson"):
    """
    r(погода в час t; метрика в час t+lag) по аномалиям, на сплошной часовой сетке.
    lag > 0: задержки реагируют с запозданием (накапливаются после плохой погоды).
    """
    d = to_msk(ds)
    ok = (d["n_flights"] >= min_flights) & d[target].notna()
    y = d[target].where(ok)
    x = d[feature]
    xa, ya = hour_anomaly(x, d["hour_msk"]), hour_anomaly(y, d["hour_msk"])
    rows = []
    for lag in lags:
        pair = pd.concat([xa, ya.shift(-lag)], axis=1).dropna()
        rows.append({"lag_h": lag, "r": _corr(pair.iloc[:, 0], pair.iloc[:, 1], method), "n": len(pair)})
    return pd.DataFrame(rows)


# ==============================================================================
# 4. Эффекты: интервалы признака, флаги, профили, сутки
# ==============================================================================
def wilson(k, n, z=1.96):
    """Интервал Уилсона для доли k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.where(n > 0, centre - half, np.nan), np.where(n > 0, centre + half, np.nan)


def aggregate_group(sub, target):
    """
    Значение метрики и 95% интервал для группы часов.
    Доли: сумма числителей / сумма знаменателей, интервал Уилсона.
    Средняя задержка: взвешенное среднее по часам, интервал - нормальное приближение по часам
    (часы автокоррелированы, поэтому интервал приблизительный и несколько узок).
    """
    if target in TARGET_COUNTS:
        num_c, den_c = TARGET_COUNTS[target]
        k, n = sub[num_c].sum(), sub[den_c].sum()
        lo, hi = wilson(k, n)
        return {"value": k / n if n else np.nan, "lo": float(lo), "hi": float(hi), "flights": int(n)}
    w = sub[TARGET_WEIGHT[target]].astype(float)
    y = sub[target]
    ok = y.notna() & (w > 0)
    if not ok.any():
        return {"value": np.nan, "lo": np.nan, "hi": np.nan, "flights": 0}
    y, w = y[ok], w[ok]
    mean = float(np.average(y, weights=w))
    if len(y) > 1:
        var = float(np.average((y - mean) ** 2, weights=w))
        eff_n = w.sum() ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(eff_n, 1.0))
    else:
        se = np.nan
    return {"value": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se, "flights": int(w.sum())}


def binned_effect(ds, feature, target, bins=8, min_flights=DEFAULT_MIN_FLIGHTS, date_from=None, date_to=None):
    """Метрика по интервалам признака (в каждом примерно одинаковое число часов)."""
    sel = select_hours(ds, target, min_flights, date_from=date_from, date_to=date_to)
    if sel.empty or sel[feature].nunique() < 3:
        return pd.DataFrame(columns=["bin", "x_mid", "value", "lo", "hi", "hours", "flights"])
    cut = pd.qcut(sel[feature], min(bins, sel[feature].nunique()), duplicates="drop")
    rows = []
    for interval, sub in sel.groupby(cut, observed=True):
        agg = aggregate_group(sub, target)
        rows.append({"bin": f"{interval.left:.3g} … {interval.right:.3g}", "x_mid": float(sub[feature].median()),
                     "hours": len(sub), **agg})
    return pd.DataFrame(rows)


def flag_comparison(ds, target, min_flights=DEFAULT_MIN_FLIGHTS, date_from=None, date_to=None):
    """Метрика при наличии погодного условия и без него (по активным часам)."""
    sel = select_hours(ds, target, min_flights, date_from=date_from, date_to=date_to)
    rows = []
    for key, (name, fn) in WEATHER_FLAGS.items():
        flag = fn(sel).fillna(False) if not sel.empty else pd.Series(dtype=bool)
        for state, sub in ((True, sel[flag]), (False, sel[~flag])):
            rows.append({"flag": key, "label": name, "active": state, "hours": len(sub),
                         **aggregate_group(sub, target)})
    return pd.DataFrame(rows)


def flag_summary(ds, target, **kwargs):
    """Одна строка на флаг: значение при условии, без него, разница и отношение."""
    comp = flag_comparison(ds, target, **kwargs)
    if comp.empty:
        return comp
    on, off = comp[comp.active].set_index("flag"), comp[~comp.active].set_index("flag")
    out = pd.DataFrame({"label": on["label"], "hours_on": on["hours"], "flights_on": on["flights"],
                        "on": on["value"], "on_lo": on["lo"], "on_hi": on["hi"], "off": off["value"],
                        "diff": on["value"] - off["value"],
                        "ratio": on["value"] / off["value"].replace(0, np.nan)})
    # Интервалы «с условием» и «без него» не пересекаются -> различие устойчиво
    out["separated"] = (on["lo"] > off["hi"]) | (on["hi"] < off["lo"])
    return out.reset_index()


def hourly_profile(ds, columns=("n_flights", "mean_delay_min", "cancel_rate", "delayed15_share"),
                   min_flights=1):
    """Средний суточный профиль (час по МСК): нагрузка и метрики."""
    d = to_msk(ds)
    d = d[d["n_flights"] >= min_flights]
    cols = [c for c in columns if c in d]
    out = d.groupby("hour_msk")[cols].mean()
    return out.reindex(range(24))


def daily_summary(ds):
    """Суточная сводка (дни по МСК): рейсы, задержки (взвешенно), отмены и экстремумы погоды."""
    d = to_msk(ds)
    g = d.groupby("date_msk")
    comp = g["n_completed"].sum()
    wdelay = (d["mean_delay_min"].fillna(0) * d["n_completed"]).groupby(d["date_msk"]).sum()
    out = pd.DataFrame({
        "n_flights": g["n_flights"].sum(), "n_cancelled": g["n_cancelled"].sum(),
        "mean_delay_min": wdelay / comp.replace(0, np.nan),
        "delayed15_share": g["n_delayed15"].sum() / comp.replace(0, np.nan),
        "cancel_rate": g["n_cancelled"].sum() / g["n_flights"].sum().replace(0, np.nan),
        "gust_max": g["wind_gusts_10m"].max(), "vis_min": g["visibility_km"].min(),
        "precip_sum": g["precipitation"].sum(), "temp_mean": g["temperature_2m"].mean(),
    })
    return out[out["n_flights"] > 0]


def date_range_msk(ds):
    d = to_msk(ds)
    return d["date_msk"].min().date(), d["date_msk"].max().date()


def overview_stats(ds):
    """Итоги по аэропорту для карточек обзора."""
    total = int(ds["n_flights"].sum())
    completed = int(ds["n_completed"].sum())
    return {
        "hours": len(ds), "flights": total, "completed": completed,
        "cancelled": int(ds["n_cancelled"].sum()), "diverted": int(ds["n_diverted"].sum()),
        "cancel_rate": ds["n_cancelled"].sum() / total if total else np.nan,
        "delayed15_share": ds["n_delayed15"].sum() / completed if completed else np.nan,
        "mean_delay_min": (float(np.average(ds["mean_delay_min"].dropna(),
                                            weights=ds.loc[ds["mean_delay_min"].notna(), "n_completed"]))
                           if ds["mean_delay_min"].notna().any() and completed else np.nan),
        "hours_with_flights": int((ds["n_flights"] > 0).sum()),
        "hours_with_weather": int(ds[["temperature_2m", "wind_gusts_10m"]].notna().all(axis=1).sum()),
        "metar_imputed": int(ds["metar_imputed"].sum()) if "metar_imputed" in ds else 0,
    }
