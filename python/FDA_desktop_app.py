# --- FDA_desktop_app.py ---
"""
Окно приложения FDA Analyzer (Tkinter + matplotlib). Три режима - три вкладки:

  1. «Анализ своих данных»   - выбрать CSV, проверить, запустить расчёт (C++ или Python);
  2. «Результаты»            - графики и таблицы: результаты автора (встроенные) и ваши запуски;
  3. «Производительность»    - сравнение C++ и Python: время, ускорение, точность, масштабирование по потокам.

Окно - тонкий слой над модулями без графики (fda_import, fda_pipeline, fda_bench, plot_results),
каждый из которых можно запускать и из командной строки (fda_app.py analyze / plots / bench).
Долгие расчёты выполняются в фоновом потоке, окно не зависает.
"""
import contextlib
import io
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import FDA_bench as B
import FDA_main_analyzer as M
import FDA_import as I
import FDA_pipeline as P
import FDA_results as R
import FDA_results_plots as PR

APP_TITLE = "FDA Analyzer: погода и задержки рейсов"
BACKEND_CHOICES = {"Авто (C++, если найдена)": "auto", "Только C++ (hpc)": "hpc", "Только Python": "python"}
TZ_CHOICES = {"UTC": "UTC", "Москва (UTC+3)": "MSK"}
DIRECTION_CHOICES = {"Все рейсы": "both", "Вылеты": "departure", "Прилёты": "arrival"}
DELAY_CHOICES = {"У гейта (revised − scheduled)": "delay_min", "По полосе (runway − scheduled)": "runway_delay_min"}


# ==============================================================================
# Фоновые задачи
# ==============================================================================
class QueueWriter(io.TextIOBase):
    """Складывает print() фонового потока в очередь, чтобы показать в окне."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(("log", text))
        return len(text)

    def flush(self):
        pass


class Task:
    """
    Запускает fn(emit) в потоке. emit("log", текст) / emit("progress", (текст, доля 0..1)).
    Результат и ошибки возвращаются в потоке интерфейса через on_done(result) / on_error(exc, traceback_text).
    """

    def __init__(self, root, fn, on_log=None, on_progress=None, on_done=None, on_error=None, capture_stdout=True):
        self.root, self.fn, self.capture = root, fn, capture_stdout
        self.on_log, self.on_progress, self.on_done, self.on_error = on_log, on_progress, on_done, on_error
        self.q = queue.Queue()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        self.root.after(100, self._poll)

    def _run(self):
        emit = lambda kind, payload: self.q.put((kind, payload))
        try:
            if self.capture:
                writer = QueueWriter(self.q)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = self.fn(emit)
            else:
                result = self.fn(emit)
            self.q.put(("done", result))
        except Exception as e:  # noqa: BLE001 - любая ошибка должна дойти до окна, а не оборвать поток молча
            self.q.put(("error", (e, traceback.format_exc())))

    def _poll(self):
        finished = False
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log" and self.on_log:
                    self.on_log(payload)
                elif kind == "progress" and self.on_progress:
                    self.on_progress(*payload)
                elif kind == "done":
                    finished = True
                    if self.on_done:
                        self.on_done(payload)
                elif kind == "error":
                    finished = True
                    if self.on_error:
                        self.on_error(*payload)
        except queue.Empty:
            pass
        if not finished:
            self.root.after(100, self._poll)


def make_log(parent, height=12):
    """Текстовое поле только для чтения с прокруткой. Возвращает (frame, text)."""
    frame = ttk.Frame(parent)
    text = tk.Text(frame, height=height, wrap="word", state="disabled", font=("Consolas", 9)
                   if sys.platform == "win32" else ("Courier", 9))
    sb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=sb.set)
    text.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    return frame, text


def append_text(widget, s, clear=False):
    widget.configure(state="normal")
    if clear:
        widget.delete("1.0", "end")
    widget.insert("end", s)
    widget.see("end")
    widget.configure(state="disabled")


def browse_file(var, title="Выберите CSV-файл", types=(("CSV", "*.csv *.txt"), ("Все файлы", "*.*"))):
    path = filedialog.askopenfilename(title=title, filetypes=types)
    if path:
        var.set(path)


def show_error(title, exc):
    messagebox.showerror(title, str(exc))


# ==============================================================================
# Вкладка 1: анализ своих данных
# ==============================================================================
class AnalyzeTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.vars = {k: tk.StringVar() for k in ("hourly", "weather", "flights", "metar", "airport", "lib")}
        self.kind = tk.StringVar(value="hourly")
        self.name = tk.StringVar(value="MYDATA")
        self.tz = tk.StringVar(value=list(TZ_CHOICES)[0])
        self.delay = tk.StringVar(value=list(DELAY_CHOICES)[0])
        self.direction = tk.StringVar(value=list(DIRECTION_CHOICES)[0])
        self.backend = tk.StringVar(value=list(BACKEND_CHOICES)[0])
        self.windows = tk.StringVar(value="24,168")
        self.min_flights = tk.IntVar(value=5)
        self.codeshare = tk.BooleanVar(value=False)
        self.cargo = tk.BooleanVar(value=False)
        self.verify = tk.BooleanVar(value=False)
        self.busy = False
        self._build()

    # --- интерфейс
    def _build(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        src = ttk.LabelFrame(left, text=" Источник данных ", padding=8)
        src.pack(fill="x")
        ttk.Radiobutton(src, text="Готовый почасовой CSV (одна таблица)", variable=self.kind, value="hourly",
                        command=self._sync).grid(row=0, column=0, columnspan=3, sticky="w")
        self.rows = {}
        self._file_row(src, 1, "hourly", "Файл:")
        ttk.Radiobutton(src, text="Исходные таблицы: погода + рейсы (+ METAR)", variable=self.kind, value="raw",
                        command=self._sync).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._file_row(src, 3, "weather", "Погода:")
        self._file_row(src, 4, "flights", "Рейсы:")
        self._file_row(src, 5, "metar", "METAR:")
        ttk.Button(src, text="Форматы файлов…", command=self._show_formats).grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Button(src, text="Создать учебные примеры…", command=self._make_examples).grid(
            row=6, column=1, columnspan=2, sticky="e", pady=(8, 0))

        opt = ttk.LabelFrame(left, text=" Параметры ", padding=8)
        opt.pack(fill="x", pady=8)
        self._row(opt, 0, "Название набора:", ttk.Entry(opt, textvariable=self.name, width=24))
        self._row(opt, 1, "Время без пояса:", self._combo(opt, self.tz, TZ_CHOICES))
        self._row(opt, 2, "Аэропорт (ICAO):", ttk.Entry(opt, textvariable=self.vars["airport"], width=10))
        self._row(opt, 3, "Задержка:", self._combo(opt, self.delay, DELAY_CHOICES, 30))
        self._row(opt, 4, "Направление:", self._combo(opt, self.direction, DIRECTION_CHOICES))
        ttk.Checkbutton(opt, text="Включать codeshare-дубли", variable=self.codeshare).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opt, text="Исключить грузовые", variable=self.cargo).grid(row=6, column=0, columnspan=2, sticky="w")
        self._row(opt, 7, "Окна, часов:", ttk.Entry(opt, textvariable=self.windows, width=14))
        self._row(opt, 8, "Мин. рейсов в часе:", ttk.Spinbox(opt, from_=1, to=100, textvariable=self.min_flights, width=6))

        eng = ttk.LabelFrame(left, text=" Вычислитель ", padding=8)
        eng.pack(fill="x")
        self._row(eng, 0, "Реализация:", self._combo(eng, self.backend, BACKEND_CHOICES, 24))
        ttk.Checkbutton(eng, text="Сверить C++ с Python (медленнее)", variable=self.verify).grid(row=1, column=0, columnspan=2, sticky="w")
        row = ttk.Frame(eng)
        row.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Label(row, text="Библиотека:").pack(side="left")
        ttk.Entry(row, textvariable=self.vars["lib"], width=22).pack(side="left", padx=4)
        ttk.Button(row, text="…", width=3, command=lambda: browse_file(
            self.vars["lib"], "Библиотека C++", (("DLL / SO", "*.dll *.so *.dylib"), ("Все файлы", "*.*")))).pack(side="left")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=10)
        self.btn_check = ttk.Button(btns, text="Проверить данные", command=self.check)
        self.btn_check.pack(side="left")
        self.btn_run = ttk.Button(btns, text="Запустить анализ", command=self.run)
        self.btn_run.pack(side="right")

        ttk.Label(right, text="Ход работы").pack(anchor="w")
        frame, self.log = make_log(right, height=30)
        frame.pack(fill="both", expand=True)
        append_text(self.log, "Выберите файлы слева и нажмите «Проверить данные», затем «Запустить анализ».\n"
                              "Нет своих файлов? Нажмите «Создать учебные примеры…» - появятся синтетические CSV.\n")
        self._sync()

    def _combo(self, parent, var, choices, width=18):
        cb = ttk.Combobox(parent, textvariable=var, values=list(choices), state="readonly", width=width)
        return cb

    def _row(self, parent, r, text, widget):
        ttk.Label(parent, text=text).grid(row=r, column=0, sticky="w", pady=1)
        widget.grid(row=r, column=1, sticky="w", padx=4, pady=1)

    def _file_row(self, parent, r, key, text):
        ttk.Label(parent, text=text, width=8).grid(row=r, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=self.vars[key], width=34)
        entry.grid(row=r, column=1, padx=4, pady=1)
        btn = ttk.Button(parent, text="…", width=3, command=lambda: browse_file(self.vars[key]))
        btn.grid(row=r, column=2)
        self.rows[key] = (entry, btn)

    def _sync(self):
        for key, (entry, btn) in self.rows.items():
            enabled = (key == "hourly") == (self.kind.get() == "hourly")
            state = "normal" if enabled else "disabled"
            entry.configure(state=state)
            btn.configure(state=state)

    # --- действия
    def _spec(self):
        v = {k: x.get().strip() for k, x in self.vars.items()}
        return I.DataSpec(kind=self.kind.get(), hourly=v["hourly"], weather=v["weather"], flights=v["flights"],
                          metar=v["metar"], naive_tz=TZ_CHOICES[self.tz.get()], airport=v["airport"],
                          delay_field=DELAY_CHOICES[self.delay.get()], direction=DIRECTION_CHOICES[self.direction.get()],
                          include_codeshare=self.codeshare.get(), exclude_cargo=self.cargo.get())

    def _windows(self):
        try:
            ws = sorted({int(x) for x in self.windows.get().replace(";", ",").split(",") if x.strip()})
        except ValueError:
            raise I.ImportProblem("Окна: ожидается список целых чисел через запятую, например 24,168")
        if not ws or ws[0] < 2:
            raise I.ImportProblem("Окна: размер должен быть не меньше 2 часов")
        return ws

    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.btn_run.configure(state=state)
        self.btn_check.configure(state=state)
        self.app.set_status("Идёт расчёт…" if busy else "Готово")

    def _show_formats(self):
        win = tk.Toplevel(self)
        win.title("Форматы файлов")
        win.geometry("760x560")
        frame, text = make_log(win, height=30)
        frame.pack(fill="both", expand=True)
        append_text(text, I.__doc__ or "")

    def _make_examples(self):
        folder = filedialog.askdirectory(title="Куда сохранить учебные примеры")
        if not folder:
            return
        try:
            names = I.write_examples(folder)
        except Exception as e:  # noqa: BLE001
            show_error("Не удалось создать примеры", e)
            return
        f = Path(folder)
        self.kind.set("hourly")
        self.vars["hourly"].set(str(f / "example_hourly.csv"))
        self.vars["weather"].set(str(f / "example_weather.csv"))
        self.vars["flights"].set(str(f / "example_flights.csv"))
        self.vars["metar"].set(str(f / "example_metar.csv"))
        self.name.set("EXAMPLE")
        self._sync()
        append_text(self.log, f"Создано: {', '.join(names)}\nПути подставлены в форму; это СИНТЕТИЧЕСКИЕ данные.\n")

    def check(self):
        if self.busy:
            return
        try:
            spec = self._spec()
        except Exception as e:  # noqa: BLE001
            show_error("Ошибка", e)
            return
        append_text(self.log, "\n--- Проверка данных ---\n")
        self._set_busy(True)

        def job(emit):
            return I.build_dataset(spec, M.PythonBackend())

        def done(res):
            append_text(self.log, "\n".join(I.describe(res)) + "\n")
            self._set_busy(False)

        def failed(exc, tb):
            append_text(self.log, f"[ERROR] {exc}\n" + ("" if isinstance(exc, I.ImportProblem) else tb))
            self._set_busy(False)

        Task(self, job, self._log, None, done, failed).start()

    def _log(self, text):
        append_text(self.log, text)

    def run(self):
        if self.busy:
            return
        try:
            spec, windows = self._spec(), self._windows()
            name = P.safe_name(self.name.get())
            min_flights = int(self.min_flights.get())
        except Exception as e:  # noqa: BLE001
            show_error("Ошибка в параметрах", e)
            return
        backend, lib, verify = BACKEND_CHOICES[self.backend.get()], self.vars["lib"].get().strip() or None, self.verify.get()
        append_text(self.log, "\n--- Анализ ---\n")
        self._set_busy(True)

        def job(emit):
            return P.run_analysis(spec, name, None, windows, min_flights, backend, lib, verify, log=print)

        def done(result):
            self._set_busy(False)
            self.app.results_tab.open_run(result["outdir"])
            self.app.select_tab(1)
            messagebox.showinfo("Готово", f"Анализ завершён ({result['backend']}). Результаты открыты на вкладке «Результаты».")

        def failed(exc, tb):
            append_text(self.log, f"[ERROR] {exc}\n" + ("" if isinstance(exc, (I.ImportProblem, RuntimeError)) else tb))
            self._set_busy(False)
            show_error("Анализ не выполнен", exc)

        Task(self, job, self._log, None, done, failed).start()


# ==============================================================================
# Вкладка 2: результаты
# ==============================================================================
class ResultsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.sources = {}
        self.extra = {}
        self._fig = None
        self.ctx = None
        self.canvas = None
        self.src = tk.StringVar()
        self.view = tk.StringVar(value=PR.VIEWS["static"])
        self.target = tk.StringVar(value=R.TARGET_LABELS[R.TARGETS[0]])
        self.feature = tk.StringVar(value=R.FEATURE_LABELS["wind_gusts_10m"])
        self.window = tk.StringVar(value="168")
        self.method = tk.StringVar(value="Пирсон")
        self._build()
        self.refresh_sources(select_first=True)

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Результаты:").pack(side="left")
        self.src_combo = ttk.Combobox(top, textvariable=self.src, state="readonly", width=46)
        self.src_combo.pack(side="left", padx=4)
        self.src_combo.bind("<<ComboboxSelected>>", lambda e: self.load(self.sources.get(self.src.get())))
        ttk.Button(top, text="Обновить", command=self.refresh_sources).pack(side="left")
        ttk.Button(top, text="Открыть папку…", command=self.open_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Сохранить все графики…", command=self.save_all).pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, pady=(8, 0))
        charts = ttk.Frame(self.nb)
        tables = ttk.Frame(self.nb)
        about = ttk.Frame(self.nb)
        self.nb.add(charts, text="Графики")
        self.nb.add(tables, text="Таблицы")
        self.nb.add(about, text="О запуске")

        # --- графики
        ctl = ttk.Frame(charts, padding=(0, 6))
        ctl.pack(fill="x")
        self._combo(ctl, "Вид:", self.view, list(PR.VIEWS.values()), 44)
        self._combo(ctl, "Метрика:", self.target, list(R.TARGET_LABELS.values()), 26)
        self.cb_feature = self._combo(ctl, "Признак:", self.feature, list(R.FEATURE_LABELS.values()), 26)
        self.cb_window = self._combo(ctl, "Окно:", self.window, ["24", "168"], 6)
        self.cb_method = self._combo(ctl, "Коэффициент:", self.method, ["Пирсон", "Спирмен"], 9)
        self.plot_frame = ttk.Frame(charts)
        self.plot_frame.pack(fill="both", expand=True)
        self.plot_msg = ttk.Label(self.plot_frame, text="Выберите результаты.", anchor="center")
        self.plot_msg.pack(fill="both", expand=True)

        # --- таблицы
        bar = ttk.Frame(tables, padding=(0, 6))
        bar.pack(fill="x")
        ttk.Button(bar, text="Таблица корреляций (с интервалами)", command=self.fill_correlations).pack(side="left")
        ttk.Button(bar, text="Погодные условия", command=self.fill_flags).pack(side="left", padx=6)
        self.table = ttk.Treeview(tables, show="headings", height=20)
        sb = ttk.Scrollbar(tables, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- о запуске
        frame, self.about = make_log(about, height=25)
        frame.pack(fill="both", expand=True)

    def _combo(self, parent, text, var, values, width):
        ttk.Label(parent, text=text).pack(side="left", padx=(8, 2))
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        return cb

    # --- источники
    def refresh_sources(self, select_first=False):
        self.sources = {}
        author = P.author_results_dir()
        if author:
            self.sources["★ Результаты автора"] = author
        for run in P.list_runs():
            self.sources[run.name] = run
        for label, path in self.extra.items():
            self.sources[label] = path
        self.src_combo.configure(values=list(self.sources))
        if select_first and self.sources:
            first = list(self.sources)[0]
            self.src.set(first)
            self.load(self.sources[first])
        elif not self.sources:
            append_text(self.about, "Результаты автора не найдены (папка data/author_results пуста).\n"
                                    "Запустите анализ на вкладке 1 или откройте папку с результатами main.py.\n", clear=True)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Папка с результатами (hourly_dataset_*.csv)")
        if folder:
            self.open_run(Path(folder))

    def open_run(self, path):
        path = Path(path)
        if not R.airports_available(path):
            messagebox.showwarning("Нет результатов", f"В папке нет файлов hourly_dataset_*.csv:\n{path}")
            return
        label = next((k for k, v in self.sources.items() if Path(v) == path), None)
        if label is None:
            label = path.name or str(path)
            self.extra[label] = path
            self.refresh_sources()
        self.src.set(label)
        self.load(path)

    def save_all(self):
        if self.ctx is None:
            return
        folder = filedialog.askdirectory(title="Куда сохранить графики (PNG)")
        if not folder:
            return
        results = str(self.ctx.dir)
        self.app.set_status("Сохраняю графики…")

        def job(emit):
            return PR.main(["--results-dir", results, "--output-dir", folder, "--dpi", "200"])

        def done(code):
            self.app.set_status("Готово")
            messagebox.showinfo("Графики", f"Сохранено в {folder}" if code == 0 else "Графики построить не удалось.")

        Task(self, job, None, None, done, lambda e, tb: show_error("Ошибка", e)).start()

    # --- загрузка
    def load(self, path):
        if path is None:
            return
        path = Path(path)
        try:
            self.ctx = PR.Context(path, None, None, 100, "png")
        except Exception as e:  # noqa: BLE001
            self.ctx = None
            show_error("Не удалось открыть результаты", e)
            return
        if not self.ctx.airports:
            self.ctx = None
            self._message("В папке нет hourly_dataset_*.csv.")
            return
        windows = R.rolling_windows_available(path) or [24, 168]
        self.cb_window.configure(values=[str(w) for w in windows])
        if self.window.get() not in [str(w) for w in windows]:
            self.window.set(str(windows[-1]))
        self._fill_about()
        for row in self.table.get_children():
            self.table.delete(row)
        self.redraw()
        self.app.set_status(f"Открыто: {path}")

    def _fill_about(self):
        info, ctx = self.ctx.info, self.ctx
        lines = [f"Папка: {ctx.dir}", f"Наборы данных: {', '.join(ctx.airports)}",
                 f"Минимум рейсов в часе: {ctx.min_flights}"]
        if info:
            a = info.get("args") or {}
            lines += [f"Вычислитель: {info.get('backend', '?')}" + (f" ({Path(info['library']).name})" if info.get("library") else ""),
                      f"Вызовов: {info.get('calls', '?')}, чистое время счёта: {1000 * float(info.get('seconds', 0)):.1f} мс",
                      f"Период (UTC): {' .. '.join(info.get('period_utc', []))}",
                      f"Задержка: {a.get('delay_field', 'delay_min')}; направление: {a.get('direction', 'both')}; "
                      f"codeshare {'включён' if a.get('include_codeshare') else 'исключён'}",
                      f"Окна, часов: {a.get('windows')}",
                      f"Запуск: {str(info.get('started_utc', ''))[:19].replace('T', ' ')} UTC"]
            ver = info.get("verify")
            if ver:
                lines.append(f"Сверка C++ и Python: {'OK' if ver.get('mismatch', 1) == 0 else 'РАСХОЖДЕНИЕ'}, "
                             f"макс. разница {ver.get('max_abs_diff', 0):.1e}")
            src = (info.get("source") or {}).get("files")
            if src:
                lines.append("Исходные файлы: " + "; ".join(src))
            for icao, s in (info.get("airports") or {}).items():
                lines.append(f"{icao}: часов {s.get('hours')}, рейсов {s.get('flights')}, отменено {s.get('cancelled')}")
        lines += ["", "Как читать: красный - положительная связь погодного признака с метрикой, синий - отрицательная.",
                  "«Без суточного цикла» - корреляция по отклонениям от среднего значения для того же часа суток;",
                  "это корреляции, а не доказательство причинности."]
        append_text(self.about, "\n".join(lines) + "\n", clear=True)

    # --- графики
    def _keys(self):
        view = {v: k for k, v in PR.VIEWS.items()}[self.view.get()]
        target = {v: k for k, v in R.TARGET_LABELS.items()}[self.target.get()]
        feature = {v: k for k, v in R.FEATURE_LABELS.items()}[self.feature.get()]
        return view, target, feature, int(self.window.get() or 0) or None

    def _message(self, text):
        self._clear_plot()
        self.plot_msg = ttk.Label(self.plot_frame, text=text, anchor="center")
        self.plot_msg.pack(fill="both", expand=True)

    def _clear_plot(self):
        for child in self.plot_frame.winfo_children():
            child.destroy()
        self.canvas = None
        self._fig = None

    def redraw(self):
        if self.ctx is None:
            return
        view, target, feature, window = self._keys()
        self.cb_feature.configure(state="readonly" if view == "rolling" else "disabled")
        self.cb_window.configure(state="readonly" if view == "rolling" else "disabled")
        self.cb_method.configure(state="readonly" if view == "static" else "disabled")
        method = "spearman" if self.method.get() == "Спирмен" else "pearson"
        self.app.set_status("Строю график…")
        self.update_idletasks()
        try:
            fig = PR.make_figure(self.ctx, view, target, feature, window, method)
        except Exception as e:  # noqa: BLE001
            self._message(f"Не удалось построить график: {e}")
            self.app.set_status("Готово")
            return
        if fig is None:
            self._message("Для этого вида и метрики нет данных (например, не было нужных окон или условий).")
        else:
            self._show_figure(fig)
        self.app.set_status("Готово")

    def _show_figure(self, fig):
        self._clear_plot()
        self._fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    # --- таблицы
    def _fill_table(self, columns, rows, widths=None):
        self.table.configure(columns=[c for c, _ in columns])
        for c, title in columns:
            self.table.heading(c, text=title)
            self.table.column(c, width=(widths or {}).get(c, 110), anchor="w")
        for row in self.table.get_children():
            self.table.delete(row)
        for r in rows:
            self.table.insert("", "end", values=r)

    def fill_correlations(self):
        if self.ctx is None:
            return
        _, target, _, _ = self._keys()
        method = "spearman" if self.method.get() == "Спирмен" else "pearson"
        self.app.set_status("Считаю корреляции и интервалы…")
        self.update_idletasks()
        rows = []
        for icao in self.ctx.airports:
            t = R.correlation_table(self.ctx.hourly[icao], target, self.ctx.min_flights, method, ci=True, n_boot=200)
            for _, r in t.iterrows():
                f = lambda v: "–" if v != v else f"{v:+.3f}"
                rows.append((icao, R.label(r["feature"]), int(r["n_hours"]), f(r["r_raw"]), f(r["r_anomaly"]),
                             f(r["ci_lo"]), f(r["ci_hi"]), "да" if r.get("significant") else "нет"))
        self._fill_table([("ds", "Набор"), ("feature", "Признак"), ("n", "Часов"), ("raw", "r сырая"),
                          ("anom", "r без сут. цикла"), ("lo", "95% от"), ("hi", "95% до"), ("sig", "0 вне интервала")],
                         rows, {"feature": 220})
        self.app.set_status("Готово")

    def fill_flags(self):
        if self.ctx is None:
            return
        _, target, _, _ = self._keys()
        rows = []
        for icao in self.ctx.airports:
            s = R.flag_summary(self.ctx.hourly[icao], target, min_flights=self.ctx.min_flights)
            for _, r in s.iterrows():
                fmt = lambda v: "–" if v != v else f"{v:.3f}"
                rows.append((icao, r["label"], int(r["hours_on"]), fmt(r["on"]), fmt(r["off"]), fmt(r["diff"]),
                             fmt(r["ratio"]), "да" if r["separated"] else "нет"))
        self._fill_table([("ds", "Набор"), ("flag", "Условие"), ("h", "Часов"), ("on", "При условии"), ("off", "Без"),
                          ("diff", "Разница"), ("ratio", "Отношение"), ("sep", "Интервалы не пересекаются")],
                         rows, {"flag": 200, "sep": 180})


# ==============================================================================
# Вкладка 3: производительность
# ==============================================================================
CHART_CHOICES = ["Время от размера задачи", "Ускорение C++", "Масштабирование по потокам", "Текстовый отчёт"]


class BenchTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.df = None
        self.meta = {}
        self.canvas = None
        self.busy = False
        preset = B.PRESETS["standard"]
        self.preset = tk.StringVar(value=preset["label"])
        self.sizes = tk.StringVar(value=",".join(map(str, preset["sizes"])))
        self.windows = tk.StringVar(value=",".join(map(str, preset["windows"])))
        self.repeats = tk.IntVar(value=preset["repeats"])
        self.scaling = tk.BooleanVar(value=True)
        self.threads = tk.StringVar(value=self._default_threads_text())
        self.pure = tk.BooleanVar(value=True)
        self.lib = tk.StringVar()
        self.chart = tk.StringVar(value=CHART_CHOICES[0])
        self.op = tk.StringVar(value=B.OP_LABELS["rolling_correlation"])
        self.win_sel = tk.StringVar()
        self.progress = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Нажмите «Запустить» или откройте сохранённые результаты.")
        self._build()

    @staticmethod
    def _default_threads_text():
        import os
        cores = os.cpu_count() or 1
        ts, t = [], 1
        while t < cores:
            ts.append(t)
            t *= 2
        ts.append(cores)
        return ",".join(str(x) for x in sorted(set(ts)))

    def _build(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text=" Параметры измерения ", padding=8)
        box.pack(fill="x")
        ttk.Label(box, text="Набор:").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(box, textvariable=self.preset, values=[p["label"] for p in B.PRESETS.values()],
                          state="readonly", width=26)
        cb.grid(row=0, column=1, sticky="w", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        for r, (text, var, w) in enumerate([("Размеры n:", self.sizes, 28), ("Окна, ч:", self.windows, 14)], start=1):
            ttk.Label(box, text=text).grid(row=r, column=0, sticky="w")
            ttk.Entry(box, textvariable=var, width=w).grid(row=r, column=1, sticky="w", padx=4)
        ttk.Label(box, text="Повторов:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(box, from_=1, to=20, textvariable=self.repeats, width=5).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Checkbutton(box, text="Чистый Python (медленно, только малые n)", variable=self.pure).grid(
            row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(box, text="Масштабирование C++ по потокам:", variable=self.scaling).grid(
            row=5, column=0, columnspan=2, sticky="w")
        ttk.Entry(box, textvariable=self.threads, width=16).grid(row=6, column=1, sticky="w", padx=4)
        ttk.Label(box, text="Библиотека:").grid(row=7, column=0, sticky="w")
        row = ttk.Frame(box)
        row.grid(row=7, column=1, sticky="w")
        ttk.Entry(row, textvariable=self.lib, width=20).pack(side="left", padx=4)
        ttk.Button(row, text="…", width=3, command=lambda: browse_file(
            self.lib, "Библиотека C++", (("DLL / SO", "*.dll *.so *.dylib"), ("Все файлы", "*.*")))).pack(side="left")

        self.btn_run = ttk.Button(left, text="Запустить измерение", command=self.run)
        self.btn_run.pack(fill="x", pady=(10, 4))
        ttk.Button(left, text="Открыть сохранённые результаты…", command=self.open_saved).pack(fill="x", pady=2)
        self.btn_author = ttk.Button(left, text="Результаты автора (его компьютер)", command=self.open_author)
        self.btn_author.pack(fill="x", pady=2)
        ttk.Button(left, text="Сохранить график…", command=self.save_chart).pack(fill="x", pady=2)
        ttk.Progressbar(left, variable=self.progress, maximum=1.0).pack(fill="x", pady=(10, 2))
        ttk.Label(left, textvariable=self.status, wraplength=320, justify="left").pack(fill="x")

        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        for text, var, values, width, key in [("График:", self.chart, CHART_CHOICES, 30, "chart"),
                                              ("Операция:", self.op, list(B.OP_LABELS.values()), 22, "op"),
                                              ("Окно:", self.win_sel, [], 6, "win")]:
            ttk.Label(ctl, text=text).pack(side="left", padx=(8, 2))
            c = ttk.Combobox(ctl, textvariable=var, values=values, state="readonly", width=width)
            c.pack(side="left")
            c.bind("<<ComboboxSelected>>", lambda e: self.redraw())
            setattr(self, f"cb_{key}", c)
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.report_frame, self.report = make_log(right, height=20)
        self._message("Результатов пока нет.")

    def _apply_preset(self):
        key = {p["label"]: k for k, p in B.PRESETS.items()}[self.preset.get()]
        p = B.PRESETS[key]
        self.sizes.set(",".join(map(str, p["sizes"])))
        self.windows.set(",".join(map(str, p["windows"])))
        self.repeats.set(p["repeats"])

    def _config(self):
        try:
            sizes = [int(x) for x in self.sizes.get().replace(";", ",").split(",") if x.strip()]
            windows = [int(x) for x in self.windows.get().replace(";", ",").split(",") if x.strip()]
            threads = [int(x) for x in self.threads.get().replace(";", ",").split(",") if x.strip()] if self.scaling.get() else []
        except ValueError:
            raise ValueError("Размеры, окна и потоки - списки целых чисел через запятую.")
        if not sizes or not windows or min(sizes) < 10 or min(windows) < 2 or any(t < 1 for t in threads):
            raise ValueError("Проверьте значения: n ≥ 10, окно ≥ 2, потоки ≥ 1.")
        return B.BenchConfig(sizes=sorted(set(sizes)), windows=sorted(set(windows)), repeats=int(self.repeats.get()),
                             threads=sorted(set(threads)), impls=[i for i in B.IMPLS if i != "python" or self.pure.get()],
                             lib=self.lib.get().strip() or None)

    # --- запуск
    def run(self):
        if self.busy:
            return
        try:
            cfg = self._config()
        except ValueError as e:
            show_error("Параметры измерения", e)
            return
        self.busy = True
        self.btn_run.configure(state="disabled")
        self.progress.set(0.0)
        self.app.set_status("Идёт измерение… Не запускайте тяжёлые программы: это исказит результаты.")

        def job(emit):
            records, meta = B.run_benchmark(cfg, lambda msg, frac: emit("progress", (msg, frac)))
            path = B.save_results(records, meta, P.app_home() / "benchmarks")
            return records, meta, path

        def progress(msg, frac):
            self.status.set(msg)
            self.progress.set(frac)

        def done(result):
            import pandas as pd
            records, meta, path = result
            self.busy = False
            self.btn_run.configure(state="normal")
            self.set_results(pd.DataFrame(records, columns=B.RECORD_FIELDS), meta)
            self.status.set(f"Готово. Сохранено: {path}")
            self.app.set_status("Готово")

        def failed(exc, tb):
            self.busy = False
            self.btn_run.configure(state="normal")
            self.status.set(f"Ошибка: {exc}")
            show_error("Измерение не выполнено", exc)

        Task(self, job, None, progress, done, failed).start()

    def open_saved(self):
        path = filedialog.askopenfilename(title="Результаты измерения", initialdir=str(P.app_home() / "benchmarks"),
                                          filetypes=(("CSV", "*.csv"), ("Все файлы", "*.*")))
        if path:
            self._open(Path(path))

    def open_author(self):
        path = P.author_benchmark_file()
        if path is None:
            messagebox.showinfo("Нет данных", "Результаты автора не найдены (data/author_results/benchmark.csv).")
            return
        self._open(path)

    def _open(self, path):
        try:
            df, meta = B.load_results(path)
        except Exception as e:  # noqa: BLE001
            show_error("Не удалось открыть файл", e)
            return
        self.set_results(df, meta)
        self.status.set(f"Открыто: {path}")

    def set_results(self, df, meta):
        self.df, self.meta = df, meta
        main_rows = df[df["note"].fillna("") != "scaling"]
        windows = sorted({int(w) for w in main_rows.loc[main_rows["op"] == "rolling_correlation", "window"].unique()})
        self.cb_win.configure(values=[str(w) for w in windows])
        if windows and self.win_sel.get() not in [str(w) for w in windows]:
            self.win_sel.set(str(windows[-1]))
        self.redraw()

    # --- отображение
    def _clear(self):
        for child in self.plot_frame.winfo_children():
            child.destroy()
        self.report_frame.pack_forget()
        self.plot_frame.pack(fill="both", expand=True, pady=(8, 0))

    def _message(self, text):
        self._clear()
        ttk.Label(self.plot_frame, text=text, anchor="center", wraplength=500).pack(fill="both", expand=True)

    def redraw(self):
        if self.df is None or self.df.empty:
            return
        chart = self.chart.get()
        op = {v: k for k, v in B.OP_LABELS.items()}[self.op.get()]
        window = int(self.win_sel.get()) if self.win_sel.get().isdigit() else None
        if chart == CHART_CHOICES[3]:
            self.plot_frame.pack_forget()
            self.report_frame.pack(fill="both", expand=True, pady=(8, 0))
            append_text(self.report, B.text_report(self.df, self.meta) + "\n", clear=True)
            return
        if chart == CHART_CHOICES[0]:
            fig = B.fig_time_vs_n(self.df, op, window)
        elif chart == CHART_CHOICES[1]:
            fig = B.fig_speedup(self.df, op)
        else:
            fig = B.fig_scaling(self.df)
        if fig is None:
            self._message("Для этого графика нет данных (например, не измерялось масштабирование по потокам "
                          "или не найдена C++ библиотека).")
            return
        self._clear()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()
        self._last_figure = fig

    def save_chart(self):
        fig = getattr(self, "_last_figure", None)
        if fig is None or self.chart.get() == CHART_CHOICES[3]:
            messagebox.showinfo("Сохранение", "Сначала выберите график (не текстовый отчёт).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG", "*.png"), ("PDF", "*.pdf")))
        if path:
            fig.savefig(path, dpi=200, bbox_inches="tight")


# ==============================================================================
# Главное окно
# ==============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800")
        self.minsize(1000, 640)
        self.status = tk.StringVar(value="Готово")  # до вкладок: они пишут в строку состояния при создании
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        self.analyze_tab = AnalyzeTab(self.nb, self)
        self.results_tab = ResultsTab(self.nb, self)
        self.bench_tab = BenchTab(self.nb, self)
        self.nb.add(self.analyze_tab, text="  1. Анализ своих данных  ")
        self.nb.add(self.results_tab, text="  2. Результаты  ")
        self.nb.add(self.bench_tab, text="  3. Производительность C++ и Python  ")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", side="bottom")

    def set_status(self, text):
        self.status.set(text)

    def select_tab(self, index):
        self.nb.select(index)


def main():
    if sys.platform == "win32":
        try:  # чёткий текст на экранах с масштабированием
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001 - не критично
            pass
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
