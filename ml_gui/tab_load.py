# ml_gui/tab_load.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from core.api_client import get_epochs, get_studies
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN

class LoadDataTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.stop_flag = False

        # Переменные API
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.token = tk.StringVar(value=DEFAULT_TOKEN)

        # Тип эпох
        self.data_type = tk.IntVar(value=1)      # 1-тонические, 2-все, 3-положение
        self.target_task = tk.StringVar(value="classification")   # classification / regression
        self.normalization = tk.StringVar(value="patient")        # patient / global
        self.test_size = tk.DoubleVar(value=0.2)

        # Фильтры исследований
        self.age_min = tk.IntVar(value=18)
        self.age_max = tk.IntVar(value=70)
        self.include_ahi = tk.BooleanVar(value=True)
        self.exclude_central = tk.BooleanVar(value=True)

        self._create_widgets()

    def _create_widgets(self):
        # Фрейм API
        frame_api = ttk.LabelFrame(self, text="Подключение к серверу", padding=5)
        frame_api.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_api, text="Адрес API:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame_api, textvariable=self.api_url, width=80).grid(row=0, column=1, padx=5)
        ttk.Label(frame_api, text="Токен:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame_api, textvariable=self.token, width=80, show="*").grid(row=1, column=1, padx=5)

        # Фрейм параметров выборки
        frame_data = ttk.LabelFrame(self, text="Параметры выборки", padding=5)
        frame_data.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_data, text="Тип эпох:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(frame_data, textvariable=self.data_type, values=[1,2,3], state='readonly', width=5).grid(row=0, column=1, padx=5)
        ttk.Label(frame_data, text="1-тонические, 2-все, 3-положение").grid(row=0, column=2, sticky=tk.W)

        ttk.Label(frame_data, text="Задача:").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(frame_data, text="Классификация эпох (has_apnea)", variable=self.target_task, value="classification").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(frame_data, text="Регрессия AHI (агрегированная по пациенту)", variable=self.target_task, value="regression").grid(row=1, column=2, sticky=tk.W)

        ttk.Label(frame_data, text="Нормализация:").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(frame_data, text="Пациент-специфическая (z-score)", variable=self.normalization, value="patient").grid(row=2, column=1, sticky=tk.W)
        ttk.Radiobutton(frame_data, text="Глобальная", variable=self.normalization, value="global").grid(row=2, column=2, sticky=tk.W)

        ttk.Label(frame_data, text="Размер тестовой выборки:").grid(row=3, column=0, sticky=tk.W)
        ttk.Scale(frame_data, from_=0.1, to=0.3, variable=self.test_size, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(frame_data, textvariable=self.test_size).grid(row=3, column=2)

        # Фильтры
        frame_filters = ttk.LabelFrame(self, text="Критерии включения исследований", padding=5)
        frame_filters.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(frame_filters, text="Исключить центральное/смешанное апноэ", variable=self.exclude_central).pack(anchor=tk.W)
        ttk.Checkbutton(frame_filters, text="Только исследования с AHI", variable=self.include_ahi).pack(anchor=tk.W)
        ttk.Label(frame_filters, text="Возраст от").pack(anchor=tk.W)
        ttk.Spinbox(frame_filters, from_=18, to=100, textvariable=self.age_min, width=5).pack(anchor=tk.W)
        ttk.Label(frame_filters, text="до").pack(anchor=tk.W)
        ttk.Spinbox(frame_filters, from_=18, to=100, textvariable=self.age_max, width=5).pack(anchor=tk.W)

        # Кнопка загрузки
        self.load_btn = ttk.Button(self, text="Загрузить данные и подготовить выборку", command=self.load_data)
        self.load_btn.pack(pady=10)

    def load_data(self):
        self.stop_flag = False
        self.load_btn.config(state=tk.DISABLED)
        self.main_app.set_status("Загрузка данных...")
        threading.Thread(target=self._load_thread, daemon=True).start()

    def _load_thread(self):
        try:
            api_url = self.api_url.get().rstrip('/')
            token = self.token.get().strip()
            if not api_url or not token:
                messagebox.showerror("Ошибка", "Укажите API URL и токен")
                return

            # 1. Загружаем исследования
            studies = get_studies(api_url, token)
            if not studies:
                self.main_app.log("Нет исследований")
                return
            df_studies = pd.DataFrame(studies)

            # Применяем фильтры
            if self.exclude_central.get():
                allowed = ['no_impairment', 'mild', 'moderate', 'severe']
                df_studies = df_studies[df_studies['breathing_impairment_severity'].isin(allowed)]
            if self.include_ahi.get():
                df_studies = df_studies[df_studies['ahi'].notna()]
            if 'age_at_study' in df_studies.columns:
                df_studies = df_studies[(df_studies['age_at_study'] >= self.age_min.get()) &
                                         (df_studies['age_at_study'] <= self.age_max.get())]

            study_ids = df_studies['study_id'].unique().tolist()
            self.main_app.log(f"Исследований после фильтрации: {len(study_ids)}")
            if not study_ids:
                self.main_app.log("Нет исследований после применения фильтров")
                return

            # 2. Загружаем эпохи
            def progress_cb(page, total, _):
                self.main_app.set_progress(int(page/total*100))
            epochs = get_epochs(api_url, token, study_ids=study_ids, data_type=self.data_type.get(),
                                stop_check=lambda: self.stop_flag, progress_callback=progress_cb)
            if not epochs or self.stop_flag:
                self.main_app.log("Нет эпох или загрузка прервана")
                return
            df_epochs = pd.DataFrame(epochs)
            self.main_app.log(f"Загружено {len(df_epochs)} эпох")

            # Определяем целевые признаки (исключаем метаданные)
            meta_cols = ['id', 'patient_id', 'study_id', 'epoch_onset', 'epoch_stage', 'has_apnea', 'data_type', 'created_at']
            feature_cols = [c for c in df_epochs.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df_epochs[c])]
            self.main_app.feature_names = feature_cols
            self.main_app.log(f"Количество признаков: {len(feature_cols)}")

            # Целевая переменная
            if self.target_task.get() == 'classification':
                y = df_epochs['has_apnea'].values
                self.main_app.task_type = 'classification'
                self.main_app.log("Задача: классификация эпох (has_apnea)")
            else:
                # Регрессия AHI – агрегация по пациентам
                self.main_app.log("Регрессия AHI: агрегация признаков по пациентам")
                df_epochs['patient_id'] = df_epochs['patient_id'].astype(int)
                patient_means = df_epochs.groupby('patient_id')[feature_cols].mean().reset_index()
                # Добавляем AHI из таблицы исследований
                ahi_map = df_studies.set_index('study_id')['ahi'].to_dict()
                study_patient = df_epochs[['patient_id', 'study_id']].drop_duplicates()
                study_patient['ahi'] = study_patient['study_id'].map(ahi_map)
                patient_ahi = study_patient.groupby('patient_id')['ahi'].first().reset_index()
                patient_means = patient_means.merge(patient_ahi, on='patient_id', how='inner')
                X = patient_means[feature_cols].values
                y_raw = patient_means['ahi'].values
                # Преобразуем в float, заменяя некорректные значения на NaN
                y = pd.to_numeric(y_raw, errors='coerce')
                # Удаляем строки с NaN в y
                valid = ~np.isnan(y)
                X = X[valid]
                y = y[valid]
                patient_ids_for_split = patient_means['patient_id'].values[valid]
                self.main_app.task_type = 'regression'
                self.normalization.set('global')
                self.main_app.log(f"Агрегировано {len(X)} пациентов после удаления пропусков AHI")
                # Пропускаем нормализацию и разделение, обработаем ниже
                self._finalize_data(X, y, patient_ids_for_split)
                return

            # 3. Нормализация (только для классификации эпох)
            X = df_epochs[feature_cols].values
            patient_ids = df_epochs['patient_id'].values

            if self.normalization.get() == 'patient':
                X_norm = np.zeros_like(X)
                for pid in np.unique(patient_ids):
                    mask = patient_ids == pid
                    X_pid = X[mask]
                    mean = np.nanmean(X_pid, axis=0)
                    std = np.nanstd(X_pid, axis=0)
                    std[std == 0] = 1
                    X_norm[mask] = (X_pid - mean) / std
                X = X_norm
            else:
                mean = np.nanmean(X, axis=0)
                std = np.nanstd(X, axis=0)
                std[std == 0] = 1
                X = (X - mean) / std

            # Удаление строк с NaN
            nan_rows = np.isnan(X).any(axis=1)
            X = X[~nan_rows]
            y = y[~nan_rows]
            patient_ids = patient_ids[~nan_rows]
            self.main_app.log(f"После удаления пропусков: {X.shape[0]} эпох")

            self._finalize_data(X, y, patient_ids)

        except Exception as e:
            self.main_app.log(f"Ошибка: {e}")
            import traceback
            self.main_app.log(traceback.format_exc())
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.load_btn.config(state=tk.NORMAL)
            self.main_app.set_progress(0)

    def _finalize_data(self, X, y, groups):
        """Разделение train/test по группам (пациентам)"""
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size.get(), random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        self.main_app.X_train = X_train
        self.main_app.X_test = X_test
        self.main_app.y_train = y_train
        self.main_app.y_test = y_test
        self.main_app.patient_ids_train = groups_train
        self.main_app.patient_ids_test = groups_test
        self.main_app.data_loaded = True

        self.main_app.log(f"Размер обучающей выборки: {X_train.shape}")
        self.main_app.log(f"Размер тестовой выборки: {X_test.shape}")
        if self.main_app.task_type == 'classification':
            unique, counts = np.unique(y_train, return_counts=True)
            self.main_app.log(f"Распределение классов в train: {dict(zip(unique, counts))}")
        else:
            if np.issubdtype(y_train.dtype, np.number):
                self.main_app.log(f"Диапазон AHI в train: {np.min(y_train):.2f} - {np.max(y_train):.2f}")
            else:
                self.main_app.log("AHI train: нечисловые данные, проверьте входные данные")
        self.main_app.set_status("Данные готовы. Перейдите на вкладку 'Обучение'.")