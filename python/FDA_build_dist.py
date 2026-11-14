# --- FDA_build_dist.py ---
"""
Сборка дистрибутива FDA Analyzer (запускать на той ОС, для которой нужен дистрибутив: Windows -> .exe).

    pip install -r requirements-build.txt
    python FDA_build_dist.py --lib cpp\\build\\Release\\aviation_analysis.dll ^
                             --author-results results --author-benchmark benchmarks\\benchmark_XXXX.csv

Результат: dist/FDA-Analyzer/ (папка) и dist/FDA-Analyzer.zip. Внутри:
    FDA-Analyzer.exe, fda-cli.exe, _internal/ (Python, NumPy, pandas, matplotlib)
    aviation_analysis.dll (+ DLL, лежащие рядом с ней) - C++ библиотека
    data/author_results/  - ваши результаты (вкладка «Результаты») и benchmark.csv (вкладка «Производительность»)
    data/examples/        - учебные CSV (синтетические)
    README.txt
Ключи: --dry-run (собрать только содержимое без PyInstaller - проверка путей), --extra-dll PATH (повторяемый).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "FDA-Analyzer"
RESULT_PATTERNS = ["hourly_dataset_*.csv", "correlations_static_*.csv", "rolling_summary_*.csv", "rolling_*_w*_*.csv",
                   "run_info.json"]
LIB_PATTERNS = ["*.dll", "*.so", "*.so.*", "*.dylib"]


def step(msg):
    print(f"\n=== {msg}")


def copy_results(src, dst):
    src, dst = Path(src), Path(dst)
    if not any(src.glob("hourly_dataset_*.csv")):
        raise SystemExit(f"[ERROR] В {src} нет hourly_dataset_*.csv - это не папка результатов main.py")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in RESULT_PATTERNS:
        for f in src.glob(pat):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                n += 1
    print(f"  результаты автора: {n} файлов -> {dst}")


def copy_benchmark(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "benchmark.csv")
    meta = src.with_name(src.stem + "_meta.json")
    if meta.is_file():
        shutil.copy2(meta, dst / "benchmark_meta.json")
    print(f"  бенчмарк автора: {src.name} -> {dst / 'benchmark.csv'}")


def copy_library(lib, extras, dst):
    """Библиотека и DLL из её папки (libgomp, libstdc++, msmpi ... если лежат рядом) -> рядом с exe."""
    lib = Path(lib)
    if not lib.is_file():
        raise SystemExit(f"[ERROR] Библиотека не найдена: {lib}")
    copied = [lib]
    for pat in LIB_PATTERNS:
        copied += [f for f in lib.parent.glob(pat) if f.is_file() and f != lib]
    copied += [Path(e) for e in extras]
    for f in copied:
        shutil.copy2(f, Path(dst) / f.name)
    print("  C++: " + ", ".join(f.name for f in copied))
    if sys.platform == "win32" and not any(f.name.lower().startswith("msmpi") for f in copied):
        print("  [ВНИМАНИЕ] Если библиотека собрана с MPI, ей нужен msmpi.dll (MS-MPI Redistributable): "
              "добавьте --extra-dll C:\\Windows\\System32\\msmpi.dll или установите MS-MPI на целевом компьютере.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", help="путь к aviation_analysis.dll / libaviation_analysis.so")
    p.add_argument("--extra-dll", action="append", default=[], help="дополнительные DLL (например, msmpi.dll)")
    p.add_argument("--author-results", default=str(ROOT / "results"), help="папка results/ с результатами main.py")
    p.add_argument("--author-benchmark", help="benchmark_*.csv, полученный командой fda-cli bench на вашем компьютере")
    p.add_argument("--out", default=str(ROOT / "dist"))
    p.add_argument("--dry-run", action="store_true", help="не запускать PyInstaller: собрать только содержимое папки")
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    app_dir = out / APP_NAME
    if not args.dry_run:
        step("PyInstaller")
        try:
            import PyInstaller.__main__ as pyi
        except ImportError:
            raise SystemExit("[ERROR] Не установлен PyInstaller: pip install -r requirements-build.txt")
        pyi.run(["--noconfirm", "--clean", "--distpath", str(out), "--workpath", str(ROOT / "build"),
                 str(ROOT / "FDA_analyzer.spec")])
    app_dir.mkdir(parents=True, exist_ok=True)

    step("Данные")
    data = app_dir / "data"
    results = Path(args.author_results)
    if results.is_dir() and any(results.glob("hourly_dataset_*.csv")):
        copy_results(results, data / "author_results")
    else:
        print(f"  [ВНИМАНИЕ] Результаты автора не найдены в {results}: вкладка «Результаты» будет пустой "
              f"(укажите --author-results).")
    if args.author_benchmark:
        copy_benchmark(args.author_benchmark, data / "author_results")
    else:
        print("  [ВНИМАНИЕ] benchmark автора не указан (--author-benchmark): на вкладке 3 не будет кнопки с его данными.")
    sys.path.insert(0, str(ROOT))
    import FDA_import as I
    I.write_examples(data / "examples")
    print(f"  учебные примеры -> {data / 'examples'}")

    step("C++ библиотека")
    if args.lib:
        copy_library(args.lib, args.extra_dll, app_dir)
    else:
        print("  [ВНИМАНИЕ] --lib не указан: приложение будет работать на Python-реализации (медленнее), "
              "сравнение C++ и Python будет недоступно.")

    shutil.copy2(ROOT / "README_DIST.md", app_dir / "README.txt")

    if not args.dry_run:
        step("Проверка собранного приложения")
        cli = app_dir / ("fda-cli.exe" if sys.platform == "win32" else "fda-cli")
        r = subprocess.run([str(cli), "info"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            raise SystemExit("[ERROR] fda-cli info завершился с ошибкой")
    if not args.no_zip and not args.dry_run:
        step("Архив")
        print("  " + shutil.make_archive(str(out / APP_NAME), "zip", out, APP_NAME))
    print(f"\n[OK] Дистрибутив: {app_dir}")


if __name__ == "__main__":
    main()
