# ml_gui/main_window.py
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime


class MainWindowML:
    def __init__(self, root):
        self.root = root
        root.title("APNEA AI - Машинное обучение (Раздел 2.6)")
        root.geometry("1400x900")

        # Переменные состояния
        self.data_loaded = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.patient_ids_train = None
        self.patient_ids_test = None
        self.feature_names = None
        self.task_type = "classification"  # or "regression"
        self.trained_models = {}  # name -> model
        self.metrics = {}  # name -> dict of metrics

        # Статус и прогресс
        self.status_var = tk.StringVar(value="Готов")
        self.progress_var = tk.IntVar(value=0)

        # Создаём вкладки
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Импортируем классы вкладок
        from ml_gui.tab_load import LoadDataTab
        from ml_gui.tab_train import TrainTab
        from ml_gui.tab_interpret import InterpretTab

        self.tab_load = LoadDataTab(self.notebook, self)
        self.tab_train = TrainTab(self.notebook, self)
        self.tab_interpret = InterpretTab(self.notebook, self)

        self.notebook.add(self.tab_load, text="1. Загрузка и настройки")
        self.notebook.add(self.tab_train, text="2. Обучение и оценка")
        self.notebook.add(self.tab_interpret, text="3. Интерпретация и отчёт")

        # Нижняя панель статуса
        bottom_frame = ttk.Frame(root)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_label = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Лог (общий)
        log_frame = ttk.LabelFrame(root, text="Лог", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def set_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()