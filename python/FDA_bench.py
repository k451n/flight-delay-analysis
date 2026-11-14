# --- FDA_bench.py ---
"""
Сравнение эффективности расчётов: C++ библиотека (ctypes, OpenMP/SIMD) и Python.

Сравниваются три реализации одной и той же задачи (формулы и пороги одинаковы, результаты сверяются):
  cpp     - libaviation_analysis (aviation_analysis.dll / .so) через ctypes - то, что использует main.py;
  numpy   - Python-бэкенд main.py (NumPy, окна блоками);
  python  - чистый Python (циклы, без NumPy) - нижняя точка отсчёта; только для небольших размеров.

Две операции:
  rolling_correlation   скользящая корреляция Пирсона, n точек, окно w  (основная вычислительная нагрузка);
  preprocess_visibility перевод видимости миль -> км и заполнение пропусков.

Что измеряется для каждой точки (операция, n, окно, реализация, потоки):
  time_min_ms / time_median_ms - полное время вызова (с упаковкой данных и ctypes), лучшее и медианное из repeats;
  kernel_min_ms                - чистое время вычисления (для C++ - только вызов библиотеки);
  throughput_gelem_s           - для скользящей корреляции: (n-w+1)*w элементов в секунду, млрд;
  max_abs_diff                 - максимальное расхождение результата с NumPy (контроль точности).
Масштабирование по числу потоков OpenMP: C++ запускается в отдельных процессах с OMP_NUM_THREADS = 1, 2, 4, ...
(число потоков OpenMP задаётся до загрузки библиотеки, в одном процессе его не поменять).

CLI:  python FDA_web_app.py bench [--sizes 4344,20000,100000] [--windows 24,168] [--threads 1,2,4] [--repeats 3]
"""
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import FDA_main_analyzer as M

IMPLS = ("cpp", "numpy", "python")
IMPL_LABELS = {"cpp": "C++ (ctypes)", "numpy": "Python + NumPy", "python": "Чистый Python"}
IMPL_COLORS = {"cpp": "#d9622b", "numpy": "#2a6fb0", "python": "#7a7a7a"}
OPS = ("rolling_correlation", "preprocess_visibility")
OP_LABELS = {"rolling_correlation": "Скользящая корреляция", "preprocess_visibility": "Очистка видимости"}
PRESETS = {
    "quick": {"sizes": [1000, 4344, 20000], "windows": [24, 168], "repeats": 3, "label": "Быстрый (секунды)"},
    "standard": {"sizes": [4344, 20000, 100000, 500000], "windows": [24, 168], "repeats": 3, "label": "Стандартный (~1 мин)"},
    "heavy": {"sizes": [4344, 100000, 500000, 2000000], "windows": [24, 168, 720], "repeats": 5,
              "label": "Тяжёлый (несколько минут)"},
}
RECORD_FIELDS = ["op", "impl", "n", "window", "threads", "repeats", "time_min_ms", "time_median_ms",
                 "kernel_min_ms", "throughput_gelem_s", "max_abs_diff", "note"]
ATOL = 1e-6          # допуск сверки: C++ собран с -ffast-math, порядок суммирования может отличаться
FILL_MISSING_SHARE = 0.15


@dataclass
class BenchConfig:
    sizes: list = field(default_factory=lambda: list(PRESETS["standard"]["sizes"]))
    windows: list = field(default_factory=lambda: list(PRESETS["standard"]["windows"]))
    repeats: int = 3
    ops: list = field(default_factory=lambda: list(OPS))
    impls: list = field(default_factory=lambda: list(IMPLS))
    threads: list = field(default_factory=list)       # для масштабирования C++; пусто = не измерять
    pure_python_limit: int = 5_000_000                # макс. (n - w + 1) * w для чистого Python
    max_case_seconds: float = 20.0                    # если один прогон дольше - повторы сокращаются до 1
    lib: str = None
    seed: int = 12345

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(text):
        return BenchConfig(**json.loads(text))


# ==============================================================================
# Чистый Python (эталон «без NumPy»)
# ==============================================================================
def pure_rolling_correlation(x, y, window):
    """Те же формулы и порог, что в C++: сумма по окну, затем коэффициент Пирсона (списки Python)."""
    n = len(x)
    m = n - window + 1
    if m <= 0:
        return []
    out = [0.0] * m
    sqrt = math.sqrt
    for s in range(m):
        sx = sy = sxx = syy = sxy = 0.0
        for i in range(s, s + window):
            a = x[i]
            b = y[i]
            sx += a
            sy += b
            sxx += a * a
            syy += b * b
            sxy += a * b
        num = window * sxy - sx * sy
        den = (window * sxx - sx * sx) * (window * syy - sy * sy)
        out[s] = 0.0 if den <= 1e-9 else num / sqrt(den)
    return out


# ==============================================================================
# Данные и бэкенды
# ==============================================================================
def make_rolling_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(size=n)
    return x, y


def make_visibility_data(n, seed):
    rng = np.random.default_rng(seed + 1)
    v = rng.choice([0.4, 1.2, 3.1, 5.0, 6.21], size=n)
    v[rng.random(n) < FILL_MISSING_SHARE] = np.nan
    v[0] = np.nan
    return v


def load_cpp(lib=None):
    """(HpcBackend или None, причина недоступности)."""
    try:
        return M.HpcBackend(lib), None
    except M.LibraryError as e:
        return None, str(e)


def system_info(cpp=None, cpp_reason=None):
    info = {"timestamp": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", ""),
            "cpp_library": str(cpp.path) if cpp is not None else None, "cpp_unavailable": cpp_reason}
    if cpp is not None:
        try:
            info["cpp_library_mtime"] = datetime.fromtimestamp(Path(cpp.path).stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            pass
    return info


def default_threads():
    """Сколько потоков получит C++ без ограничений: OMP_NUM_THREADS или число логических ядер."""
    env = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    return int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)


# ==============================================================================
# Измерение
# ==============================================================================
def _measure(call, backend, repeats, max_seconds, warmup=True):
    """
    call() -> результат. Возвращает (результат, [время вызова, с], [чистое время вычисления, с]).
    Для медленных случаев повторы сокращаются: если первый прогон дольше max_seconds - он единственный.
    """
    times, kernels, result = [], [], None
    if warmup:
        t = time.perf_counter()
        result = call()
        first = time.perf_counter() - t
        if first > max_seconds:
            return result, [first], [first]
    for _ in range(max(1, repeats)):
        before = backend.seconds if backend is not None else 0.0
        t = time.perf_counter()
        result = call()
        wall = time.perf_counter() - t
        times.append(wall)
        kernels.append((backend.seconds - before) if backend is not None else wall)
        if wall > max_seconds:
            break
    return result, times, kernels


def _record(op, impl, n, window, threads, times, kernels, diff, note=""):
    best = min(times)
    rec = {"op": op, "impl": impl, "n": int(n), "window": int(window) if window else 0, "threads": int(threads),
           "repeats": len(times), "time_min_ms": best * 1000, "time_median_ms": statistics.median(times) * 1000,
           "kernel_min_ms": min(kernels) * 1000, "throughput_gelem_s": None, "max_abs_diff": diff, "note": note}
    if op == "rolling_correlation" and window and best > 0:
        rec["throughput_gelem_s"] = (n - window + 1) * window / best / 1e9
    return rec


def run_cases(cfg, cpp, progress=None, impls=None):
    """Основной прогон в текущем процессе. Возвращает список записей (см. RECORD_FIELDS)."""
    impls = list(impls or cfg.impls)
    numpy_backend = M.PythonBackend()
    cpp_threads = default_threads()
    records = []
    jobs = []
    for op in cfg.ops:
        for n in cfg.sizes:
            if op == "rolling_correlation":
                jobs += [(op, n, w) for w in cfg.windows if n >= w + 2]
            else:
                jobs.append((op, n, 0))
    for k, (op, n, w) in enumerate(jobs):
        if progress:
            progress(f"{OP_LABELS[op]}: n = {n}" + (f", окно {w}" if w else ""), k / max(1, len(jobs)))
        if op == "rolling_correlation":
            x, y = make_rolling_data(n, cfg.seed)
            calls = {"cpp": (lambda: cpp.rolling_correlation(x, y, w)) if cpp else None,
                     "numpy": lambda: numpy_backend.rolling_correlation(x, y, w)}
            xl, yl = x.tolist(), y.tolist()
            calls["python"] = lambda: np.asarray(pure_rolling_correlation(xl, yl, w))
            python_ok = (n - w + 1) * w <= cfg.pure_python_limit
        else:
            v = make_visibility_data(n, cfg.seed)
            calls = {"cpp": (lambda: cpp.preprocess_visibility(v)) if cpp else None,
                     "numpy": lambda: numpy_backend.preprocess_visibility(v)}
            python_ok = False   # PythonBackend уже написан на циклах Python; отдельной «чистой» версии нет
        backends = {"cpp": cpp, "numpy": numpy_backend, "python": None}
        reference = calls["numpy"]()
        for impl in impls:
            call = calls.get(impl)
            if call is None:
                continue
            if impl == "python" and not python_ok:
                continue
            res, times, kernels = _measure(call, backends[impl], cfg.repeats, cfg.max_case_seconds,
                                           warmup=impl != "python")
            diff = float(np.max(np.abs(np.asarray(res) - reference))) if len(reference) else 0.0
            records.append(_record(op, impl, n, w, cpp_threads if impl == "cpp" else 1, times, kernels, diff))
    return records


def worker_command(cfg):
    """Команда запуска подпроцесса-воркера (для собранного exe и для запуска из исходников)."""
    if getattr(sys, "frozen", False):
        # воркеру нужны потоки stdout: берём консольный exe рядом с оконным (если он есть)
        cli = Path(sys.executable).with_name("fda-cli" + Path(sys.executable).suffix)
        return [str(cli if cli.is_file() else sys.executable), "worker", "--config", cfg.to_json()]
    return [sys.executable, str(Path(__file__).with_name("FDA_web_app.py")), "worker", "--config", cfg.to_json()]


WORKER_TAG = "FDA_BENCH_JSON:"


def worker_main(config_json):
    """Тело воркера: считает только C++ (потоки задаёт OMP_NUM_THREADS окружения), печатает JSON."""
    cfg = BenchConfig.from_json(config_json)
    cpp, reason = load_cpp(cfg.lib)
    if cpp is None:
        print(WORKER_TAG + json.dumps({"error": reason}))
        return 1
    records = run_cases(cfg, cpp, impls=["cpp"])
    print(WORKER_TAG + json.dumps({"records": records, "threads": default_threads()}))
    return 0


def run_thread_scaling(cfg, progress=None):
    """C++ при разном числе потоков OpenMP: отдельный процесс на каждое значение. Возвращает записи."""
    if not cfg.threads:
        return []
    sub = BenchConfig(sizes=[max(cfg.sizes)], windows=[max(cfg.windows)], repeats=cfg.repeats,
                      ops=["rolling_correlation"], impls=["cpp"], threads=[], lib=cfg.lib, seed=cfg.seed,
                      max_case_seconds=cfg.max_case_seconds)
    records = []
    for i, t in enumerate(cfg.threads):
        if progress:
            progress(f"C++: {t} поток(ов) OpenMP", i / len(cfg.threads))
        env = dict(os.environ, OMP_NUM_THREADS=str(t))
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW: без мигающей консоли у оконного exe
        try:
            proc = subprocess.run(worker_command(sub), env=env, capture_output=True, text=True, timeout=1800,
                                  encoding="utf-8", errors="replace", **kwargs)
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"Не удалось запустить подпроцесс для {t} потоков: {e}") from e
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(WORKER_TAG):
                payload = json.loads(line[len(WORKER_TAG):])
        if payload is None or "error" in payload:
            reason = payload.get("error") if payload else (proc.stderr.strip()[-400:] or "нет ответа")
            raise RuntimeError(f"Подпроцесс ({t} потоков) завершился с ошибкой: {reason}")
        for r in payload["records"]:
            r["threads"] = t
            r["note"] = "scaling"
            records.append(r)
    return records


def run_benchmark(cfg, progress=None):
    """Полный бенчмарк: (records, meta). records - список словарей; meta - сведения о системе и конфигурации."""
    cpp, reason = load_cpp(cfg.lib)
    meta = system_info(cpp, reason)
    meta["config"] = asdict(cfg)
    if cpp is None and "cpp" in cfg.impls and progress:
        progress("C++ библиотека не найдена - измеряются только Python-реализации", 0.0)
    records = run_cases(cfg, cpp, progress)
    if cpp is not None and cfg.threads:
        records += run_thread_scaling(cfg, progress)
    if progress:
        progress("Готово", 1.0)
    return records, meta


# ==============================================================================
# Сохранение, загрузка, сводка
# ==============================================================================
def save_results(records, meta, folder, stem=None):
    import pandas as pd
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = stem or f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    df = pd.DataFrame(records, columns=RECORD_FIELDS)
    df.to_csv(folder / f"{stem}.csv", index=False, float_format="%.6g")
    (folder / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder / f"{stem}.csv"


def load_results(csv_path):
    """(DataFrame, meta) из сохранённого бенчмарка; meta - из соседнего *_meta.json, если есть."""
    import pandas as pd
    p = Path(csv_path)
    df = pd.read_csv(p)
    meta_path = p.with_name(p.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return df, meta


def summarize(df):
    """
    Сводная таблица на пару (операция, n, окно): время каждой реализации и ускорения
    (cpp_vs_numpy, cpp_vs_python, numpy_vs_python) по лучшему времени. Только основные записи (без scaling).
    """
    import pandas as pd
    d = df[df["note"].fillna("") != "scaling"]
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index=["op", "n", "window"], columns="impl", values="time_min_ms", aggfunc="min")
    for impl in IMPLS:
        if impl not in piv:
            piv[impl] = np.nan
    out = piv[list(IMPLS)].rename(columns={i: f"{i}_ms" for i in IMPLS}).reset_index()
    out["cpp_vs_numpy"] = out["numpy_ms"] / out["cpp_ms"]
    out["cpp_vs_python"] = out["python_ms"] / out["cpp_ms"]
    out["numpy_vs_python"] = out["python_ms"] / out["numpy_ms"]
    diff = d.groupby(["op", "n", "window"])["max_abs_diff"].max().reset_index()
    return out.merge(diff, on=["op", "n", "window"], how="left")


def scaling_table(df):
    """Масштабирование C++ по потокам: время, ускорение к 1 потоку и эффективность."""
    d = df[(df["note"].fillna("") == "scaling") & (df["impl"] == "cpp")].sort_values("threads")
    if d.empty:
        return d
    d = d[["threads", "n", "window", "time_min_ms", "kernel_min_ms"]].copy()
    base = d.loc[d["threads"].idxmin(), "time_min_ms"]
    d["speedup"] = base / d["time_min_ms"]
    d["efficiency"] = d["speedup"] / (d["threads"] / d["threads"].min())
    return d


def text_report(df, meta=None):
    """Текстовая сводка для консоли и отчёта."""
    lines = []
    if meta:
        lines.append(f"Система: {meta.get('platform', '?')}, ядер {meta.get('cpu_count', '?')}, "
                     f"Python {meta.get('python', '?')}, NumPy {meta.get('numpy', '?')}")
        lines.append(f"C++ библиотека: {meta.get('cpp_library') or 'не найдена (' + str(meta.get('cpp_unavailable')) + ')'}")
    s = summarize(df)
    for op in OPS:
        part = s[s["op"] == op] if not s.empty else s
        if part.empty:
            continue
        lines.append("")
        lines.append(f"{OP_LABELS[op]}: время лучшего прогона, мс")
        lines.append(f"{'n':>9} {'окно':>5} {'C++':>10} {'NumPy':>10} {'Python':>12} {'C++/NumPy':>10} {'C++/Python':>11} {'расхожд.':>9}")
        for _, r in part.sort_values(["n", "window"]).iterrows():
            f = lambda v, d=2: "-" if v is None or np.isnan(v) else f"{v:.{d}f}"
            lines.append(f"{int(r['n']):>9} {int(r['window']) if r['window'] else '-':>5} {f(r['cpp_ms']):>10} "
                         f"{f(r['numpy_ms']):>10} {f(r['python_ms']):>12} "
                         f"{('x' + f(r['cpp_vs_numpy'], 1)) if not np.isnan(r['cpp_vs_numpy']) else '-':>10} "
                         f"{('x' + f(r['cpp_vs_python'], 1)) if not np.isnan(r['cpp_vs_python']) else '-':>11} "
                         f"{r['max_abs_diff']:>9.1e}")
    sc = scaling_table(df)
    if not sc.empty:
        lines.append("")
        lines.append(f"Масштабирование C++ по потокам OpenMP (n = {int(sc['n'].iloc[0])}, окно {int(sc['window'].iloc[0])}):")
        for _, r in sc.iterrows():
            lines.append(f"  {int(r['threads']):>3} поток(ов): {r['time_min_ms']:.2f} мс, ускорение x{r['speedup']:.2f}, "
                         f"эффективность {100 * r['efficiency']:.0f}%")
    bad = df[(df["max_abs_diff"].fillna(0) > ATOL)]
    lines.append("")
    lines.append("Сверка точности: " + ("все результаты совпадают с NumPy в пределах %.0e" % ATOL if bad.empty
                                        else f"ВНИМАНИЕ: {len(bad)} записей расходятся больше {ATOL:.0e}"))
    return "\n".join(lines)


# ==============================================================================
# Графики (matplotlib.figure.Figure без pyplot: годятся и для окна Tk, и для сохранения)
# ==============================================================================
def _figure(w=8.5, h=4.6):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=100)
    return fig


def _style(ax):
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_time_vs_n(df, op="rolling_correlation", window=None):
    """Время от размера задачи (логарифмические оси), по реализациям."""
    d = df[(df["op"] == op) & (df["note"].fillna("") != "scaling")]
    if op == "rolling_correlation":
        windows = sorted(d["window"].unique())
        if not windows:
            return None
        window = window if window in windows else windows[-1]
        d = d[d["window"] == window]
    fig = _figure()
    ax = fig.add_subplot(111)
    for impl in IMPLS:
        s = d[d["impl"] == impl].sort_values("n")
        if s.empty:
            continue
        ax.plot(s["n"], s["time_min_ms"], "-o", ms=4, color=IMPL_COLORS[impl], label=IMPL_LABELS[impl])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число точек n (часов)")
    ax.set_ylabel("Время лучшего прогона, мс")
    ax.set_title(f"{OP_LABELS[op]}" + (f", окно {int(window)} ч" if op == "rolling_correlation" else ""), loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_speedup(df, op="rolling_correlation"):
    """Ускорение C++ относительно NumPy и чистого Python."""
    s = summarize(df)
    s = s[s["op"] == op] if not s.empty else s
    if s.empty:
        return None
    s = s.sort_values(["window", "n"])
    labels = [f"{int(r.n):,}".replace(",", " ") + (f"\nw={int(r.window)}" if r.window else "") for r in s.itertuples()]
    x = np.arange(len(s))
    fig = _figure(max(8.5, 0.8 * len(s) + 3), 4.6)
    ax = fig.add_subplot(111)
    width = 0.38
    ax.bar(x - width / 2, s["cpp_vs_numpy"], width, color=IMPL_COLORS["numpy"], label="к Python + NumPy")
    ax.bar(x + width / 2, s["cpp_vs_python"], width, color=IMPL_COLORS["python"], label="к чистому Python")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Ускорение C++, раз")
    ax.set_title(f"{OP_LABELS[op]}: во сколько раз C++ быстрее", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_scaling(df):
    """Ускорение C++ при росте числа потоков OpenMP (с идеальной линией)."""
    sc = scaling_table(df)
    if sc.empty or len(sc) < 2:
        return None
    fig = _figure(8.5, 4.4)
    ax = fig.add_subplot(111)
    ax.plot(sc["threads"], sc["speedup"], "-o", color=IMPL_COLORS["cpp"], label="C++ / OpenMP")
    t0 = sc["threads"].min()
    ax.plot(sc["threads"], sc["threads"] / t0, "--", color="#999", label="идеальное")
    ax.set_xlabel("Потоков OpenMP")
    ax.set_ylabel(f"Ускорение относительно {int(t0)} потока")
    ax.set_title(f"Масштабирование по потокам (n = {int(sc['n'].iloc[0])}, окно {int(sc['window'].iloc[0])})", loc="left")
    ax.set_xticks(list(sc["threads"]))
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig
