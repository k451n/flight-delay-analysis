# --- FDA_results_plots.py ---
"""
Графики для отчёта по результатам FDA_main_analyzer.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # без окна: работает в Docker/CI
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import FDA_results as R

KINDS = ["static", "rolling", "summary", "binned", "flags", "lag", "profile", "daily"]
DEFAULT_ROLLING_FEATURES = ["wind_gusts_10m", "visibility_km", "precipitation"]
BINNED_FEATURES = ["wind_gusts_10m", "visibility_km", "precipitation", "temperature_2m"]
LAG_FEATURES = ["wind_gusts_10m", "visibility_km"]
DPI_DEFAULT = 200

matplotlib.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
})
CAVEAT = "Корреляция не доказывает причинность. Часы - автокоррелированные наблюдения."


# ==============================================================================
# Вспомогательное
# ==============================================================================
class Context:
    """Загруженные данные и параметры одного запуска."""

    def __init__(self, results_dir, out_dir, min_flights, dpi, fmt):
        self.dir, self.dpi, self.fmt = Path(results_dir), dpi, fmt
        self.out = Path(out_dir) if out_dir is not None else None
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
        self.figures = []
        self.info = R.load_run_info(self.dir)
        if min_flights is None:
            min_flights = (self.info.get("args") or {}).get("min_flights", R.DEFAULT_MIN_FLIGHTS)
        self.min_flights = int(min_flights)
        self.airports = R.airports_available(self.dir)
        self.hourly = {i: R.load_hourly(self.dir, i) for i in self.airports}
        self.saved = []

    def save(self, fig, name):
        if self.out is None:
            self.figures.append((name, fig))
            return
        path = self.out / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self.saved.append(path)
        print(f"[PLOT] {path}")

    def subtitle(self, target):
        return f"{R.TARGET_LABELS[target]}; часы с ≥ {self.min_flights} рейсами; {CAVEAT}"


def _subplots(nrows=1, ncols=1, figsize=None, squeeze=True, sharex=False, sharey=False):
    """Figure без pyplot: нет глобального списка открытых фигур, подходит и для файла, и для окна Tk."""
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols, squeeze=squeeze, sharex=sharex, sharey=sharey)


def _ap_name(icao):
    return f"{R.airport_name(icao)} ({icao})" if icao in R.AIRPORTS else icao


def _limit(values, floor=0.3):
    """Симметричный предел цветовой шкалы: не меньше floor, округлён вверх до 0.05."""
    v = np.nanmax(np.abs(values)) if np.size(values) and np.isfinite(values).any() else floor
    return max(floor, float(np.ceil(v * 20) / 20))


def _heatmap(ax, data, rows, cols, vmax, stars=None):
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color="#888")
                continue
            mark = "*" if stars is not None and stars[i, j] else ""
            ax.text(j, i, f"{v:+.2f}{mark}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 * vmax else "#222", fontsize=9)
    return im


# ==============================================================================
# 1. Статическая корреляция: сырая и без суточного цикла
# ==============================================================================
def plot_static(ctx, target, method="pearson"):
    cols, data, stars = [], [], []
    for icao in ctx.airports:
        ds = ctx.hourly[icao]
        tab = R.correlation_table(ds, target, ctx.min_flights, method=method).set_index("feature")
        if tab.empty:
            continue
        tab = tab.reindex(R.FEATURES)
        cols += [f"{icao}\nсырая", f"{icao}\nбез сут. цикла"]
        data += [tab["r_raw"].to_numpy(), tab["r_anomaly"].to_numpy()]
        stars += [np.zeros(len(tab), bool), tab["significant"].fillna(False).to_numpy(bool)]
    if not data:
        print(f"[WARNING] static_{target}: нет данных")
        return
    mat = np.column_stack(data)
    st = np.column_stack(stars)
    keep = ~np.isnan(mat).all(axis=1)
    rows = [R.label(f) for f, k in zip(R.FEATURES, keep) if k]
    fig, ax = _subplots(figsize=(1.9 * len(cols) + 3, 0.5 * len(rows) + 2.2))
    im = _heatmap(ax, mat[keep], rows, cols, _limit(mat), st[keep])
    fig.colorbar(im, ax=ax, label="Коэффициент Спирмена" if method == "spearman" else "Коэффициент Пирсона",
                 shrink=0.85)
    ax.set_title(f"Погода и «{R.TARGET_LABELS[target].lower()}»: корреляция за весь период", loc="left")
    fig.text(0.01, -0.09, "* - 95% интервал блочного бутстрепа (по суткам) не включает 0.  "
             + ctx.subtitle(target), fontsize=8, color="#555")
    ctx.save(fig, f"static_{target}" + ("_spearman" if method == "spearman" else ""))


# ==============================================================================
# 2. Скользящая корреляция
# ==============================================================================
def plot_rolling(ctx, target, feature, window):
    fig, ax = _subplots(figsize=(11, 4.2))
    drawn = False
    notes = []
    for icao in ctx.airports:
        df = R.load_rolling(ctx.dir, icao, window, target)
        if df is None or feature not in df:
            continue
        s = df[feature].dropna()
        ax.plot(s.index, s.to_numpy(), color=R.color(icao), lw=1.2, alpha=0.85, label=_ap_name(icao))
        notes.append(f"{icao}: среднее r = {s.mean():+.2f}, |r| ≥ {R.CORR_STRONG:g} в {100 * (s.abs() >= R.CORR_STRONG).mean():.0f}% окон")
        drawn = True
    if not drawn:
        return
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-R.CORR_STRONG, R.CORR_STRONG, color="#999", alpha=0.10, lw=0)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Конец окна (UTC)")
    ax.set_ylabel("r Пирсона в окне")
    ax.set_title(f"{R.label(feature)} и «{R.TARGET_LABELS[target].lower()}»: скользящая корреляция, окно {window} ч", loc="left")
    ax.legend(loc="upper right", ncol=2)
    fig.text(0.01, -0.09, "Окно - подряд идущие активные часы (не календарные сутки). Серая полоса |r| < 0.3.  "
             + "  ".join(notes), fontsize=8, color="#555")
    ctx.save(fig, f"rolling_{target}_{feature}_w{window}")


def plot_rolling_summary(ctx, target):
    panels = []
    for icao in ctx.airports:
        s = R.load_summary(ctx.dir, icao)
        if s is None or s.empty:
            continue
        s = s[s["target"] == target]
        if s.empty:
            continue
        piv = s.pivot(index="feature", columns="window_h", values="mean_r").reindex(
            [f for f in R.FEATURES if f in set(s["feature"])])
        panels.append((icao, piv))
    if not panels:
        return
    vmax = _limit(np.concatenate([p.to_numpy().ravel() for _, p in panels]), 0.2)
    fig, axes = _subplots(1, len(panels), figsize=(5.2 * len(panels) + 1.5, 4.4), squeeze=False)
    for ax, (icao, piv) in zip(axes[0], panels):
        im = _heatmap(ax, piv.to_numpy(float), [R.label(f) for f in piv.index],
                      [f"{w} ч" for w in piv.columns], vmax)
        ax.set_title(_ap_name(icao), loc="left")
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes[0].tolist(), label="Средняя скользящая r", shrink=0.85)
    fig.suptitle(f"«{R.TARGET_LABELS[target]}»: средняя скользящая корреляция по окнам", x=0.01, ha="left", fontsize=12)
    ctx.save(fig, f"rolling_summary_{target}")


# ==============================================================================
# 3. Эффект по интервалам признака
# ==============================================================================
def _fmt_target(ax, target):
    ax.set_ylabel(R.TARGET_LABELS[target])
    if target in R.RATE_TARGETS:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))


def plot_binned(ctx, target, features=BINNED_FEATURES, bins=8):
    feats = [f for f in features if any(f in ctx.hourly[i] for i in ctx.airports)]
    if not feats:
        return
    ncols = 2
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = _subplots(nrows, ncols, figsize=(11, 3.6 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), feats):
        for icao in ctx.airports:
            b = R.binned_effect(ctx.hourly[icao], feature, target, bins, ctx.min_flights)
            if b.empty:
                continue
            c = R.color(icao)
            ax.plot(b["x_mid"], b["value"], "-o", ms=4, color=c, label=_ap_name(icao))
            ax.fill_between(b["x_mid"], b["lo"], b["hi"], color=c, alpha=0.15, lw=0)
        ax.set_xlabel(R.label(feature))
        _fmt_target(ax, target)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    axes[0][0].legend()
    fig.suptitle(f"«{R.TARGET_LABELS[target]}» по интервалам погодного признака (в каждом ~поровну часов)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.03, "Полоса - 95% интервал. " + ctx.subtitle(target), fontsize=8, color="#555")
    fig.tight_layout()
    ctx.save(fig, f"binned_{target}")


# ==============================================================================
# 4. Погодные флаги
# ==============================================================================
def plot_flags(ctx, target):
    summaries = {}
    for icao in ctx.airports:
        s = R.flag_summary(ctx.hourly[icao], target, min_flights=ctx.min_flights)
        if not s.empty:
            summaries[icao] = s.set_index("flag")
    if not summaries:
        return
    flags = [k for k in R.WEATHER_FLAGS if any(summaries[i].loc[k, "hours_on"] >= 5 for i in summaries)]
    if not flags:
        print(f"[WARNING] flags_{target}: ни одно погодное условие не встретилось в 5+ активных часах")
        return
    fig, axes = _subplots(1, len(summaries), figsize=(6.2 * len(summaries), 0.55 * len(flags) + 2.2),
                             squeeze=False, sharey=True)
    y = np.arange(len(flags))
    for ax, (icao, s) in zip(axes[0], summaries.items()):
        s = s.loc[flags]
        on = s["on"].to_numpy(float)
        err = np.vstack([on - s["on_lo"].to_numpy(float), s["on_hi"].to_numpy(float) - on])
        err = np.clip(np.nan_to_num(err), 0, None)
        ax.barh(y + 0.18, s["off"], height=0.34, color="#bbbbbb", label="без условия")
        ax.barh(y - 0.18, on, height=0.34, xerr=err, color=R.color(icao), label="при условии",
                error_kw={"lw": 1, "capsize": 2})
        for yi, (hrs, sep) in enumerate(zip(s["hours_on"], s["separated"])):
            ax.text(ax.get_xlim()[1], yi - 0.18, f"  {int(hrs)} ч{' *' if sep else ''}", va="center", fontsize=8,
                    color="#444", clip_on=False)
        ax.set_yticks(y, [R.WEATHER_FLAGS[k][0] for k in flags])
        ax.invert_yaxis()
        ax.set_title(_ap_name(icao), loc="left")
        ax.set_xlabel(R.TARGET_LABELS[target])
        if target in R.RATE_TARGETS:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(loc="lower right")
    fig.suptitle(f"«{R.TARGET_LABELS[target]}» при неблагоприятной погоде и без неё", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.09, "Справа - число часов с условием; * - интервалы «при условии» и «без» не пересекаются.  "
             + ctx.subtitle(target), fontsize=8, color="#555")
    fig.tight_layout()
    ctx.save(fig, f"flags_{target}")


# ==============================================================================
# 5. Запаздывание
# ==============================================================================
def plot_lag(ctx, target, features=LAG_FEATURES):
    feats = [f for f in features if any(f in ctx.hourly[i] for i in ctx.airports)]
    if not feats:
        return
    fig, axes = _subplots(1, len(feats), figsize=(5.6 * len(feats), 3.8), squeeze=False, sharey=True)
    for ax, feature in zip(axes[0], feats):
        for icao in ctx.airports:
            lag = R.lag_correlation(ctx.hourly[icao], feature, target, min_flights=ctx.min_flights)
            ax.plot(lag["lag_h"], lag["r"], "-o", ms=3, color=R.color(icao), label=_ap_name(icao))
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8, ls="--")
        ax.set_xlabel("Сдвиг k, ч (метрика в час t+k)")
        ax.set_title(R.label(feature), loc="left")
    axes[0][0].set_ylabel("r по аномалиям")
    axes[0][0].legend()
    fig.suptitle(f"Запаздывание отклика: «{R.TARGET_LABELS[target]}»", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    ctx.save(fig, f"lag_{target}")


# ==============================================================================
# 6. Суточный профиль и суточная динамика
# ==============================================================================
def plot_hourly_profile(ctx):
    fig, axes = _subplots(1, 3, figsize=(14, 3.8))
    spec = [("n_flights", "Рейсов в час (среднее)"), ("mean_delay_min", "Средняя задержка, мин"),
            ("cancel_rate", "Доля отменённых")]
    for ax, (col, title) in zip(axes, spec):
        for icao in ctx.airports:
            prof = R.hourly_profile(ctx.hourly[icao], min_flights=1)
            ax.plot(prof.index, prof[col], "-o", ms=3, color=R.color(icao), label=_ap_name(icao))
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Час суток (МСК)")
        ax.set_title(title, loc="left")
        if col == "cancel_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].legend()
    fig.suptitle("Суточный цикл нагрузки и задержек - причина сравнивать погоду с аномалиями, а не с сырыми значениями",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    ctx.save(fig, "hourly_profile")


def plot_daily(ctx):
    fig, axes = _subplots(2, 1, figsize=(12, 6.4), sharex=True)
    for icao in ctx.airports:
        d = R.daily_summary(ctx.hourly[icao])
        c = R.color(icao)
        for ax, col in zip(axes, ("mean_delay_min", "cancel_rate")):
            ax.plot(d.index, d[col], color=c, lw=0.8, alpha=0.35)
            ax.plot(d.index, d[col].rolling(7, center=True, min_periods=4).mean(), color=c, lw=2,
                    label=_ap_name(icao))
        worst = d["mean_delay_min"].idxmax() if d["mean_delay_min"].notna().any() else None
        if worst is not None:
            axes[0].annotate(f"{worst:%d.%m}: {d.loc[worst, 'mean_delay_min']:.0f} мин", (worst, d.loc[worst, "mean_delay_min"]),
                             textcoords="offset points", xytext=(4, 6), fontsize=8, color=c)
    axes[0].margins(y=0.15)
    axes[0].set_ylabel("Средняя задержка, мин")
    axes[1].set_ylabel("Доля отменённых")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[1].set_xlabel("Дата (МСК)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Суточная динамика: тонкая линия - день, толстая - скользящее среднее за 7 дней", loc="left")
    fig.tight_layout()
    ctx.save(fig, "daily_timeseries")


# ==============================================================================
# Для окна приложения
# ==============================================================================
VIEWS = {
    "static": "Корреляции с погодой: сырые и без суточного цикла",
    "rolling": "Скользящая корреляция во времени",
    "summary": "Сводка скользящих корреляций: признак × окно",
    "binned": "Метрика по интервалам погодных признаков",
    "flags": "Неблагоприятная погода и без неё",
    "lag": "Запаздывание отклика на погоду",
    "profile": "Суточный профиль нагрузки и задержек",
    "daily": "Динамика по дням",
}
VIEWS_NEED_TARGET = {"static", "rolling", "summary", "binned", "flags", "lag"}


def make_figure(ctx, view, target=None, feature=None, window=None, method="pearson"):
    """Строит одну фигуру выбранного вида и возвращает её (или None, если данных нет). Фигура не закрыта."""
    ctx.figures.clear()
    target = target or R.TARGETS[0]
    if view == "static":
        plot_static(ctx, target, method)
    elif view == "rolling":
        windows = R.rolling_windows_available(ctx.dir)
        plot_rolling(ctx, target, feature or DEFAULT_ROLLING_FEATURES[0], window or (windows[-1] if windows else 24))
    elif view == "summary":
        plot_rolling_summary(ctx, target)
    elif view == "binned":
        plot_binned(ctx, target)
    elif view == "flags":
        plot_flags(ctx, target)
    elif view == "lag":
        plot_lag(ctx, target)
    elif view == "profile":
        plot_hourly_profile(ctx)
    elif view == "daily":
        plot_daily(ctx)
    else:
        raise ValueError(f"Неизвестный вид графика: {view}")
    return ctx.figures[-1][1] if ctx.figures else None


# ==============================================================================
# Точка входа
# ==============================================================================
def csv_list(text, cast=str):
    return [cast(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Графики по результатам main.py")
    p.add_argument("--results-dir", default=None, help="папка results/ (по умолчанию: рядом со скриптом)")
    p.add_argument("--output-dir", default=None, help="куда сохранять (по умолчанию: plots/ рядом с results/)")
    p.add_argument("--targets", default=",".join(R.TARGETS))
    p.add_argument("--features", default=",".join(DEFAULT_ROLLING_FEATURES),
                   help="признаки для графиков скользящей корреляции")
    p.add_argument("--windows", default=None, help="окна (часы) для скользящих графиков; по умолчанию - все из results/")
    p.add_argument("--only", default=",".join(KINDS), help=f"какие графики строить: {','.join(KINDS)}")
    p.add_argument("--min-flights", type=int, default=None, help="по умолчанию - как в run_info.json")
    p.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = p.parse_args(argv)
    args.targets, args.features, args.only = csv_list(args.targets), csv_list(args.features), csv_list(args.only)
    for name, valid, given in (("--targets", R.TARGETS, args.targets), ("--features", R.FEATURES, args.features),
                               ("--only", KINDS, args.only)):
        bad = [g for g in given if g not in valid]
        if bad:
            p.error(f"{name}: неизвестные значения {bad}; допустимо: {', '.join(valid)}")
    args.windows = csv_list(args.windows, int) if args.windows else None
    return args


def main(argv=None):
    args = parse_args(argv)
    results_dir = R.find_results_dir(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir.parent / "plots"
    print(f"=== ГРАФИКИ FDA: {results_dir} -> {out_dir} ===")
    if not R.airports_available(results_dir):
        print(f"[ERROR] В {results_dir} нет hourly_dataset_*.csv. Сначала запустите main.py.")
        return 1
    ctx = Context(results_dir, out_dir, args.min_flights, args.dpi, args.format)
    windows = args.windows or R.rolling_windows_available(results_dir)
    only = set(args.only)
    print(f"[INFO] Аэропорты: {', '.join(ctx.airports)}; часы с ≥ {ctx.min_flights} рейсами; окна: {windows or 'нет'}")

    for target in args.targets:
        if "static" in only:
            plot_static(ctx, target)
            plot_static(ctx, target, method="spearman")
        if "rolling" in only:
            for feature in args.features:
                for window in windows:
                    plot_rolling(ctx, target, feature, window)
        if "summary" in only:
            plot_rolling_summary(ctx, target)
        if "binned" in only:
            plot_binned(ctx, target)
        if "flags" in only:
            plot_flags(ctx, target)
        if "lag" in only:
            plot_lag(ctx, target)
    if "profile" in only:
        plot_hourly_profile(ctx)
    if "daily" in only:
        plot_daily(ctx)

    if not ctx.saved:
        print("[ERROR] Ни одного графика не построено - проверьте содержимое results/")
        return 1
    print(f"\n[OK] Сохранено графиков: {len(ctx.saved)} в {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
