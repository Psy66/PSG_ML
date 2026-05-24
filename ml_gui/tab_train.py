# ml_gui/tab_train.py (полная обновлённая версия)
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, accuracy_score, f1_score,
                             balanced_accuracy_score, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import os
import torch

class TrainTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.stop_flag = False
        self.trained_models = {}
        self.metrics = {}
        self.results_df = None
        self.current_figure = None

        self.models_selection = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP']
        self.model_vars = {name: tk.BooleanVar(value=True) for name in self.models_selection}
        self.use_grid_search = tk.BooleanVar(value=False)
        self.n_cv_folds = tk.IntVar(value=5)

        self._create_widgets()

    def _create_widgets(self):
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        models_frame = ttk.LabelFrame(left_frame, text="Модели для обучения", padding=5)
        models_frame.pack(fill=tk.X, pady=5)
        for name in self.models_selection:
            ttk.Checkbutton(models_frame, text=name, variable=self.model_vars[name]).pack(anchor=tk.W)

        cv_frame = ttk.LabelFrame(left_frame, text="Кросс-валидация", padding=5)
        cv_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cv_frame, text="Число фолдов (GroupKFold):").pack(anchor=tk.W)
        ttk.Spinbox(cv_frame, from_=2, to=10, textvariable=self.n_cv_folds, width=5).pack(anchor=tk.W)
        ttk.Checkbutton(cv_frame, text="Подбор гиперпараметров (GridSearchCV)", variable=self.use_grid_search).pack(anchor=tk.W)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.train_btn = ttk.Button(btn_frame, text="Запустить обучение", command=self.run_training)
        self.train_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.save_models_btn = ttk.Button(btn_frame, text="Сохранить модели", command=self.save_models_to_disk, state=tk.DISABLED)
        self.save_models_btn.pack(side=tk.LEFT, padx=2)

        self.right_notebook = ttk.Notebook(self)
        self.right_notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_metrics = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_metrics, text="Метрики")
        self.metrics_tree = ttk.Treeview(self.tab_metrics, show='headings')
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)

        self.tab_plots = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_plots, text="Графики")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_log = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_log, text="Лог обучения")
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    def stop_training(self):
        self.stop_flag = True
        self.log("Остановка обучения...")

    def run_training(self):
        if not self.main_app.data_loaded:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные на вкладке 'Загрузка'.")
            return

        self.stop_flag = False
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_models_btn.config(state=tk.DISABLED)
        self.log("Начало обучения...")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            X_train = self.main_app.X_train
            X_test = self.main_app.X_test
            y_train = self.main_app.y_train
            y_test = self.main_app.y_test
            patient_train = self.main_app.patient_ids_train

            self.is_regression = (self.main_app.task_type == 'regression')
            self.log(f"Задача: {'регрессия' if self.is_regression else 'классификация'}")
            self.log(f"Размеры: train {X_train.shape}, test {X_test.shape}")

            groups = patient_train
            cv = GroupKFold(n_splits=self.n_cv_folds.get())

            models_to_train = [name for name in self.models_selection if self.model_vars[name].get()]
            if not models_to_train:
                self.log("Не выбрано ни одной модели.")
                return

            self.trained_models = {}
            self.metrics = {}
            all_metrics = []

            for model_name in models_to_train:
                if self.stop_flag:
                    break
                self.log(f"\n=== Обучение {model_name} ===")
                model = self._get_model(model_name)
                if model is None:
                    self.log(f"  Модель {model_name} не поддерживается для текущей задачи")
                    continue

                # --- Кросс-валидация / подбор гиперпараметров ---
                if self.use_grid_search.get():
                    param_grid = self._get_param_grid(model_name)
                    if param_grid:
                        self.log("  Подбор гиперпараметров (GridSearchCV)...")
                        scoring = 'roc_auc' if not self.is_regression else 'r2'
                        gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
                        gs.fit(X_train, y_train, groups=groups)
                        model = gs.best_estimator_
                        self.log(f"  Лучшие параметры: {gs.best_params_}")
                    else:
                        self.log("  Кросс-валидация...")
                        scores = []
                        for train_idx, val_idx in cv.split(X_train, y_train, groups):
                            X_tr, X_val = X_train[train_idx], X_train[val_idx]
                            y_tr, y_val = y_train[train_idx], y_train[val_idx]
                            model_clone = self._get_model(model_name)
                            model_clone.fit(X_tr, y_tr)
                            if self.is_regression:
                                score = model_clone.score(X_val, y_val)
                            else:
                                score = roc_auc_score(y_val, model_clone.predict_proba(X_val)[:,1])
                            scores.append(score)
                        self.log(f"  CV {self.n_cv_folds.get()}-fold: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                self.trained_models[model_name] = model

                # --- Предсказания ---
                if self.is_regression:
                    y_pred = model.predict(X_test)
                    # Инициализация словаря для модели, если его нет
                    if model_name not in self.metrics:
                        self.metrics[model_name] = {}
                    self.metrics[model_name]['y_pred'] = y_pred
                else:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    self.metrics[model_name] = {'y_prob': y_prob, 'y_pred': y_pred}

                # --- Расчёт метрик ---
                metrics_dict = {}
                if self.is_regression:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics_dict = {'MSE': mse, 'R2': r2}
                else:
                    acc = accuracy_score(y_test, y_pred)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    sens = tp / (tp + fn) if (tp+fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn+fp) > 0 else 0
                    metrics_dict = {'Accuracy': acc, 'Balanced Acc': bal_acc, 'F1': f1,
                                    'AUC': auc, 'Sensitivity': sens, 'Specificity': spec}

                # Обновляем метрики модели (ключ уже существует или создаём)
                if model_name not in self.metrics:
                    self.metrics[model_name] = {}
                self.metrics[model_name].update(metrics_dict)
                all_metrics.append({'Model': model_name, **metrics_dict})
                self.log(f"  Тестовые метрики: {metrics_dict}")

            if self.stop_flag:
                self.log("Обучение прервано.")
                return

            self.results_df = pd.DataFrame(all_metrics)
            self._display_metrics_table()

            if not self.is_regression:
                self._plot_roc_pr_curves()
                self._plot_confusion_matrices()
                self._plot_calibration_curve()

            self.save_models_btn.config(state=tk.NORMAL)
            self.main_app.tab_interpret.update_model_list()
            self.log("Обучение завершено.")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _get_model(self, name):
        if self.is_regression:
            if name == 'Linear Regression':
                return LinearRegression()
            elif name == 'Random Forest':
                return RandomForestRegressor(n_estimators=100, n_jobs=-1)
            elif name == 'XGBoost':
                # Отключаем GPU, чтобы избежать предупреждений о несоответствии устройств
                params = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu'}
                return xgb.XGBRegressor(**params)
            elif name == 'MLP':
                return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True,
                                    validation_fraction=0.1, random_state=42)
        else:
            if name == 'Linear Regression':
                return LogisticRegression(max_iter=1000, class_weight='balanced')
            elif name == 'Random Forest':
                return RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
            elif name == 'XGBoost':
                params = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                          'tree_method': 'hist', 'device': 'cpu'}
                return xgb.XGBClassifier(**params)
            elif name == 'MLP':
                return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True,
                                     validation_fraction=0.1, random_state=42)
        return None

    def _get_param_grid(self, name):
        if self.is_regression:
            if name == 'Random Forest':
                return {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
            elif name == 'XGBoost':
                return {'learning_rate': [0.01, 0.1], 'max_depth': [3, 6], 'subsample': [0.8, 1.0]}
            elif name == 'MLP':
                return {'hidden_layer_sizes': [(64,), (128,64)], 'alpha': [0.0001, 0.001]}
        else:
            if name == 'Linear Regression':
                return {'C': [0.1, 1.0, 10.0]}
            elif name == 'Random Forest':
                return {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
            elif name == 'XGBoost':
                return {'learning_rate': [0.01, 0.1], 'max_depth': [3, 6], 'subsample': [0.8, 1.0]}
            elif name == 'MLP':
                return {'hidden_layer_sizes': [(64,), (128,64)], 'alpha': [0.0001, 0.001]}
        return None

    def _display_metrics_table(self):
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        if self.results_df is None or self.results_df.empty:
            return
        cols = list(self.results_df.columns)
        self.metrics_tree['columns'] = cols
        self.metrics_tree['show'] = 'headings'
        for col in cols:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=100, anchor='center')
        for _, row in self.results_df.iterrows():
            self.metrics_tree.insert('', 'end', values=list(row))

    def _plot_roc_pr_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, m in self.metrics.items():
            if 'y_prob' in m:
                fpr, tpr, _ = roc_curve(self.main_app.y_test, m['y_prob'])
                axes[0].plot(fpr, tpr, label=f"{name} (AUC={m['AUC']:.3f})")
        axes[0].plot([0,1], [0,1], 'k--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves')
        axes[0].legend()
        for name, m in self.metrics.items():
            if 'y_prob' in m:
                prec, rec, _ = precision_recall_curve(self.main_app.y_test, m['y_prob'])
                axes[1].plot(rec, prec, label=name)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend()
        fig.tight_layout()
        self._show_plot(fig)

    def _plot_confusion_matrices(self):
        models = [name for name, m in self.metrics.items() if 'y_pred' in m and not self.is_regression]
        if not models:
            return
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
        if len(models) == 1:
            axes = [axes]
        for ax, name in zip(axes, models):
            y_pred = self.metrics[name]['y_pred']
            cm = confusion_matrix(self.main_app.y_test, y_pred)
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_xticklabels(['No Apnea','Apnea'])
            ax.set_yticklabels(['No Apnea','Apnea'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(name)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i,j], ha='center', va='center', color='red')
        fig.tight_layout()
        self._show_plot(fig)

    def _plot_calibration_curve(self):
        from sklearn.calibration import calibration_curve
        fig, ax = plt.subplots(figsize=(6,5))
        for name, m in self.metrics.items():
            if 'y_prob' in m:
                prob_true, prob_pred = calibration_curve(self.main_app.y_test, m['y_prob'], n_bins=10)
                ax.plot(prob_pred, prob_true, marker='o', label=name)
        ax.plot([0,1], [0,1], 'k--', label='Perfect')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration curves')
        ax.legend()
        self._show_plot(fig)

    def _show_plot(self, fig):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    def save_models_to_disk(self):
        if not self.trained_models:
            messagebox.showwarning("Нет моделей", "Сначала обучите модели.")
            return
        from tkinter import filedialog
        dir_path = filedialog.askdirectory(title="Выберите папку для сохранения моделей")
        if not dir_path:
            return
        for name, model in self.trained_models.items():
            joblib.dump(model, os.path.join(dir_path, f"{name}.joblib"))
        self.log(f"Модели сохранены в {dir_path}")