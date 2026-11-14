# UML-диаграммы: работа дистрибутива FDA-Analyzer.exe

Диаграммы описывают архитектуру решения и работу собранного приложения `FDA-Analyzer` (issue #9).
Исходники в формате Mermaid лежат в [`uml/*.mmd`](uml), готовые изображения в [`uml/*.png`](uml).
GitHub рисует блоки Mermaid прямо на этой странице.

## 1. Диаграмма развёртывания: состав дистрибутива

Что входит в `dist/FDA-Analyzer/` (его собирает `FDA_build_dist.py`) и как части взаимодействуют на компьютере пользователя.

```mermaid
flowchart TB
    user(["Пользователь<br/>(Windows, без установленного Python)"])

    subgraph dist["«artifact» dist/FDA-Analyzer/ (zip-архив FDA-Analyzer.zip)"]
        direction TB
        gui["FDA-Analyzer.exe<br/>оконное приложение (Tkinter)"]
        cli["fda-cli.exe<br/>консоль: info, analyze, plots, bench, worker"]
        internal["_internal/<br/>Python, NumPy, pandas, matplotlib,<br/>модули FDA_*.py (PyInstaller)"]
        dll["aviation_analysis.dll<br/>C++ ядро (OpenMP, SIMD)"]
        msmpi["msmpi.dll<br/>MS-MPI (зависимость библиотеки)"]
        readme["README.txt"]
        subgraph data["data/"]
            author["author_results/<br/>результаты автора + benchmark.csv"]
            examples["examples/<br/>учебные CSV (синтетика)"]
        end
    end

    subgraph home["Рабочая папка пользователя ~/FDA_Analyzer (FDA_HOME)"]
        runs["runs/<br/>результаты запусков"]
        bench["benchmarks/<br/>benchmark_*.csv"]
    end

    csv[("CSV пользователя<br/>почасовой или weather + flights + metar")]

    user -->|запускает| gui
    user -.->|консоль| cli
    gui -. "использует" .-> internal
    cli -. "использует" .-> internal
    gui -->|"ctypes: загрузка"| dll
    cli -->|"ctypes: загрузка"| dll
    dll --> msmpi
    gui -->|читает| author
    gui -->|читает| examples
    csv -->|импорт| gui
    gui -->|пишет| runs
    gui -->|пишет| bench
    gui -->|"subprocess: worker<br/>OMP_NUM_THREADS=1,2,4…"| cli
```

![Диаграмма развёртывания](uml/01_deployment.png)

## 2. Диаграмма компонентов: модули и зависимости

```mermaid
flowchart LR
    subgraph ui["Слой интерфейса"]
        app["FDA_desktop_app<br/>App, AnalyzeTab,<br/>ResultsTab, BenchTab, Task"]
    end
    subgraph logic["Слой логики"]
        imp["FDA_import<br/>DataSpec, ImportResult,<br/>build_dataset"]
        pipe["FDA_pipeline<br/>run_analysis, пути,<br/>list_runs"]
        res["FDA_results<br/>чтение и пересчёт<br/>результатов"]
        plots["FDA_results_plots<br/>Context, графики"]
        bench["FDA_bench<br/>BenchConfig,<br/>run_benchmark"]
        main["FDA_main_analyzer<br/>assemble_dataset,<br/>analyze_airport"]
    end
    subgraph backends["Вычислительные бэкенды"]
        hpc["HpcBackend<br/>ctypes"]
        py["PythonBackend<br/>NumPy"]
    end
    lib[["aviation_analysis.dll<br/>compute_rolling_correlation<br/>preprocess_weather_vectors"]]
    fs[("Файлы: CSV, run_info.json,<br/>PNG, benchmark.csv")]
    build["FDA_build_dist<br/>сборка дистрибутива"]

    app --> imp
    app --> pipe
    app --> res
    app --> plots
    app --> bench
    pipe --> imp
    pipe --> main
    imp --> main
    bench --> main
    plots --> res
    main --> hpc
    main --> py
    hpc --> lib
    pipe --> fs
    res --> fs
    bench --> fs
    build -. "упаковывает всё<br/>в FDA-Analyzer.exe" .-> app
```

![Диаграмма компонентов](uml/02_components.png)

## 3. Диаграмма классов

```mermaid
classDiagram
    direction LR
    class App {
        +analyze_tab
        +results_tab
        +bench_tab
        +set_status(text)
        +select_tab(index)
    }
    class AnalyzeTab {
        +check()
        +run()
        -_spec() DataSpec
    }
    class ResultsTab {
        +open_run(path)
        +load(path)
        +redraw()
        +save_all()
    }
    class BenchTab {
        +run()
        +open_saved()
        +open_author()
        +redraw()
    }
    class Task {
        +start()
        -_run()
        -_poll()
    }
    class DataSpec {
        +kind
        +hourly
        +weather
        +flights
        +metar
        +naive_tz
        +delay_field
        +direction
    }
    class ImportResult {
        +ds
        +notes
        +warnings
    }
    class ImportProblem
    class BenchConfig {
        +sizes
        +windows
        +repeats
        +threads
        +impls
    }
    class HpcBackend {
        +name = hpc
        +preprocess_visibility()
        +rolling_correlation()
    }
    class PythonBackend {
        +name = python
        +preprocess_visibility()
        +rolling_correlation()
    }
    class WeatherRecord {
        «ctypes.Structure»
    }
    class AviationAnalysisDLL {
        «C++ библиотека»
        compute_rolling_correlation()
        preprocess_weather_vectors()
    }
    class Context {
        +results_dir
        +airports
        +save(fig, name)
    }

    App *-- AnalyzeTab
    App *-- ResultsTab
    App *-- BenchTab
    AnalyzeTab ..> Task : фоновый поток
    BenchTab ..> Task
    ResultsTab ..> Task
    AnalyzeTab ..> DataSpec : создаёт
    AnalyzeTab --> ResultsTab : открывает результат
    DataSpec ..> ImportResult : build_dataset()
    ImportProblem --|> ValueError
    HpcBackend ..> WeatherRecord
    HpcBackend ..> AviationAnalysisDLL : ctypes
    BenchTab ..> BenchConfig
    ResultsTab --> Context
    AnalyzeTab ..> HpcBackend : backend auto/hpc
    AnalyzeTab ..> PythonBackend : backend python
```

![Диаграмма классов](uml/03_classes.png)

## 4. Диаграмма последовательности: анализ своих данных

Вкладка «Анализ своих данных»: от нажатия «Запустить» до отображения результатов.

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant A as AnalyzeTab
    participant T as Task (поток)
    participant P as FDA_pipeline
    participant I as FDA_import
    participant M as FDA_main_analyzer
    participant B as Backend (C++ / Python)
    participant R as ResultsTab
    participant FS as Файлы runs/

    U->>A: выбирает CSV, параметры, «Запустить»
    A->>A: _spec() собирает DataSpec
    A->>T: Task(job).start()
    activate T
    T->>P: run_analysis(spec, name, windows, backend…)
    P->>M: select_backends() (поиск aviation_analysis.dll)
    M-->>P: HpcBackend или PythonBackend
    P->>I: build_dataset(spec, backend)
    I->>I: чтение CSV, часовой пояс → UTC
    I->>M: assemble_dataset(погода, METAR, рейсы)
    M->>B: preprocess_visibility (мили → км)
    I-->>P: ImportResult (ds, warnings)
    P->>FS: hourly_dataset_*.csv
    P->>M: analyze_airport(ds, backend)
    loop цель × окно (24, 168 ч) × признак
        M->>B: rolling_correlation(x, y, window)
        B-->>M: массив r
    end
    M->>FS: correlations_static, rolling_*, rolling_summary
    P->>FS: run_info.json
    P-->>T: результат (outdir, backend, seconds)
    deactivate T
    T-->>A: on_done (в потоке интерфейса)
    A->>R: open_run(outdir)
    R->>FS: загрузка результатов
    R-->>U: графики и таблицы на вкладке «Результаты»
```

![Последовательность: анализ](uml/04_seq_analysis.png)

## 5. Диаграмма последовательности: бенчмарк C++ и Python

Вкладка «Производительность»: для масштабирования по потокам OpenMP каждое значение измеряется в отдельном процессе `fda-cli.exe worker`.

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant BT as BenchTab
    participant T as Task (поток)
    participant B as FDA_bench
    participant L as aviation_analysis.dll
    participant W as fda-cli.exe worker
    participant FS as benchmarks/

    U->>BT: настраивает размеры, окна, потоки, «Измерить»
    BT->>T: Task(job).start()
    activate T
    T->>B: run_benchmark(cfg, progress)
    B->>B: load_cpp() (ищет DLL рядом с exe)
    B->>L: run_cases: C++ (ctypes)
    B->>B: run_cases: NumPy и чистый Python
    Note over B: сверка результатов с NumPy (max_abs_diff)
    loop для каждого числа потоков t
        B->>W: subprocess, OMP_NUM_THREADS=t
        W->>L: run_cases (только C++)
        W-->>B: JSON в stdout (FDA_BENCH_JSON)
    end
    B-->>T: records, meta
    T->>B: save_results()
    B->>FS: benchmark_*.csv
    T-->>BT: on_done (progress, результат)
    BT-->>U: графики времени, ускорения, масштабирования
```

![Последовательность: бенчмарк](uml/05_seq_bench.png)
