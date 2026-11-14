# --- FDA_pipeline.py ---
"""
Запуск анализа на данных пользователя и служебные пути приложения.

run_analysis():  DataSpec (CSV) -> почасовой датасет -> main.analyze_airport (C++ или Python)
                 -> папка результатов (те же файлы, что создаёт FDA_main_analyzer.py), которую умеют читать
                 fda_results / plot_results / окно приложения.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import FDA_main_analyzer as M
import FDA_import as I

HOURLY_COLUMNS = (M.OPENMETEO_FEATURES + ["vsby_mi", "visibility_km", "ceiling_ft", "n_metar", "metar_imputed"]
                  + M.FLIGHT_COUNT_COLUMNS + ["mean_delay_min", "median_delay_min", "delayed15_share", "cancel_rate"])


# ==============================================================================
# Пути
# ==============================================================================
def app_dir():
    """Папка приложения: рядом с exe (собранное) или с исходниками."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_home():
    """Рабочая папка пользователя (запуски, бенчмарки). Можно переопределить переменной FDA_HOME."""
    home = Path(os.environ.get("FDA_HOME") or (Path.home() / "FDA_Analyzer"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def runs_root():
    p = app_home() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _data_roots():
    roots = [app_dir() / "data"]
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS) / "data")
    return roots


def author_results_dir():
    """Встроенные результаты автора: data/author_results (или FDA_AUTHOR_RESULTS)."""
    env = os.environ.get("FDA_AUTHOR_RESULTS")
    candidates = ([Path(env)] if env else []) + [r / "author_results" for r in _data_roots()]
    for c in candidates:
        if c.is_dir() and any(c.glob("hourly_dataset_*.csv")):
            return c
    return None


def author_benchmark_file():
    for r in _data_roots():
        p = r / "author_results" / "benchmark.csv"
        if p.is_file():
            return p
    return None


def examples_dir():
    for r in _data_roots():
        if (r / "examples").is_dir():
            return r / "examples"
    return app_home() / "examples"


def list_runs():
    """Папки запусков пользователя (новые первыми), в которых есть результаты."""
    root = runs_root()
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("hourly_dataset_*.csv"))]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


# ==============================================================================
# Анализ
# ==============================================================================
def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip()).strip("_")
    return name or "MYDATA"


def prepare_backend(preference="auto", lib=None, verify=False):
    """(основной бэкенд, эталон или None). preference: auto | hpc | python."""
    args = M.parse_args(["--backend", preference] + (["--lib", lib] if lib else []) + (["--verify"] if verify else []))
    try:
        return M.select_backends(args)
    except SystemExit as e:
        raise RuntimeError("C++ библиотека недоступна (см. сообщение выше). Выберите «Авто» или «Python».") from e


def run_analysis(spec, name, outdir=None, windows=(24, 168), min_flights=5, backend_pref="auto", lib=None,
                 verify=False, log=print):
    """
    Полный анализ своих данных. Возвращает словарь: outdir, name, backend, seconds, calls, used_hours, warnings.
    Ошибки в данных -> fda_import.ImportProblem (текст для пользователя).
    """
    name = safe_name(name)
    if outdir is None:
        outdir = runs_root() / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args = M.parse_args(["--windows", ",".join(str(int(w)) for w in windows), "--min-flights", str(int(min_flights)),
                         "--backend", backend_pref, "--output-dir", str(outdir)]
                        + (["--lib", lib] if lib else []) + (["--verify"] if verify else [])
                        + ["--delay-field", spec.delay_field, "--direction", spec.direction])

    log(f"[INFO] Набор данных: {name}; результаты: {outdir}")
    backend, reference = prepare_backend(backend_pref, lib, verify)
    t_start = time.perf_counter()
    res = I.build_dataset(spec, backend)
    for line in I.describe(res):
        log(("[CHECK] " if line.startswith("⚠") else "[DATA] ") + line)
    ds = res.ds
    active = int((ds["n_flights"] >= args.min_flights).sum())
    if active < M.MIN_SAMPLE_HOURS:
        raise I.ImportProblem(f"Слишком мало часов с ≥ {args.min_flights} рейсами: {active} "
                              f"(нужно не менее {M.MIN_SAMPLE_HOURS}).")
    ds[HOURLY_COLUMNS].to_csv(outdir / f"hourly_dataset_{name}.csv", float_format="%.6g")

    static_df, summary_df, verify_res = M.analyze_airport(name, ds, backend, reference, args, outdir)
    M.print_results(name, static_df, summary_df, args.windows)
    if static_df.empty:
        raise I.ImportProblem("Анализ не дал результатов: после отбора часов осталось слишком мало данных.")

    info = {"started_utc": datetime.now(timezone.utc).isoformat(), "backend": backend.name,
            "library": str(backend.path) if backend.path else None,
            "args": {k: (str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v)
                     for k, v in vars(args).items()},
            "period_utc": [str(ds.index.min()), str(ds.index.max() + (ds.index[1] - ds.index[0]))],
            "source": {"kind": spec.kind, "files": [p for p in (spec.hourly, spec.weather, spec.flights, spec.metar) if p]},
            "airports": {name: {"hours": len(ds), "flights": int(ds["n_flights"].sum()),
                                "cancelled": int(ds["n_cancelled"].sum()),
                                "used_hours": {t: len(M.select_rows(ds, t, args.min_flights, M.available_features(ds)))
                                               for t in M.TARGETS}}},
            "calls": backend.calls, "seconds": round(backend.seconds, 6),
            "verify": verify_res if reference is not None else None,
            "wall_seconds": round(time.perf_counter() - t_start, 3)}
    (outdir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[HPC] Бэкенд {backend.name}: {backend.calls} вызовов, {backend.seconds * 1000:.1f} мс чистого счёта")
    log(f"[OK] Готово: {outdir}")
    return {"outdir": outdir, "name": name, "backend": backend.name, "seconds": backend.seconds,
            "calls": backend.calls, "used_hours": info["airports"][name]["used_hours"], "warnings": res.warnings,
            "verify": info["verify"]}
