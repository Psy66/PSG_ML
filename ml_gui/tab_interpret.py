# ml_gui/tab_interpret.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import shap
import joblib
import os
import tempfile
import base64
import io
import webbrowser
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Увеличиваем лимит открытых фигур
plt.rcParams['figure.max_open_warning'] = 50


class InterpretTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_model = None
        self.current_model_name = None
        self.shap_values = None
        self.shap_explainer = None
        self.X_sample = None
        self.current_figure = None
        self._figures = []  # для отслеживания открытых фигур

        self._create_widgets()

    def _create_widgets(self):
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        model_frame = ttk.LabelFrame(left_frame, text="Выбор модели", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state='readonly')
        self.model_combo.pack(fill=tk.X, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        ttk.Button(model_frame, text="Загрузить модель (из файла)", command=self.load_model_from_file).pack(fill=tk.X,
                                                                                                            pady=2)

        viz_frame = ttk.LabelFrame(left_frame, text="Настройки визуализации", padding=5)
        viz_frame.pack(fill=tk.X, pady=5)
        ttk.Label(viz_frame, text="SHAP subsample size:").pack(anchor=tk.W)
        self.shap_sample_size = tk.IntVar(value=1000)
        ttk.Spinbox(viz_frame, from_=100, to=5000, textvariable=self.shap_sample_size, width=8).pack(anchor=tk.W)

        ttk.Label(viz_frame, text="t-SNE perplexity:").pack(anchor=tk.W)
        self.tsne_perp = tk.IntVar(value=30)
        ttk.Spinbox(viz_frame, from_=5, to=100, textvariable=self.tsne_perp, width=8).pack(anchor=tk.W)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        self.shap_btn = ttk.Button(btn_frame, text="Рассчитать SHAP", command=self.compute_shap, state=tk.DISABLED)
        self.shap_btn.pack(fill=tk.X, pady=2)
        self.tsne_btn = ttk.Button(btn_frame, text="Построить t-SNE", command=self.plot_tsne, state=tk.DISABLED)
        self.tsne_btn.pack(fill=tk.X, pady=2)
        self.save_plots_btn = ttk.Button(btn_frame, text="Сохранить графики (PNG)", command=self.save_plots,
                                         state=tk.DISABLED)
        self.save_plots_btn.pack(fill=tk.X, pady=2)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать HTML-отчёт", command=self.generate_report,
                                     state=tk.DISABLED)
        self.report_btn.pack(fill=tk.X, pady=2)

        self.right_notebook = ttk.Notebook(self)
        self.right_notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_feature_importance = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_feature_importance, text="Важность признаков")
        self.fi_frame = ttk.Frame(self.tab_feature_importance)
        self.fi_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_shap_summary = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_shap_summary, text="SHAP summary")
        self.shap_summary_frame = ttk.Frame(self.tab_shap_summary)
        self.shap_summary_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_shap_dependence = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_shap_dependence, text="SHAP dependence")
        self.shap_dep_frame = ttk.Frame(self.tab_shap_dependence)
        self.shap_dep_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_tsne = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_tsne, text="t-SNE / PCA")
        self.tsne_frame = ttk.Frame(self.tab_tsne)
        self.tsne_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_log = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_log, text="Лог")
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    def update_model_list(self):
        models = list(self.main_app.trained_models.keys())
        self.model_combo['values'] = models
        if models:
            self.model_combo.set(models[0])
            self.on_model_selected()
        else:
            self.model_var.set('')
            self.shap_btn.config(state=tk.DISABLED)
            self.tsne_btn.config(state=tk.DISABLED)
            self.save_plots_btn.config(state=tk.DISABLED)
            self.report_btn.config(state=tk.DISABLED)

    def on_model_selected(self, event=None):
        name = self.model_var.get()
        if name and name in self.main_app.trained_models:
            self.current_model_name = name
            self.current_model = self.main_app.trained_models[name]
            self.shap_btn.config(state=tk.NORMAL)
            self.tsne_btn.config(state=tk.NORMAL)
            self.save_plots_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.log(f"Выбрана модель: {name}")
            self.plot_feature_importance()
        else:
            self.current_model = None
            self.shap_btn.config(state=tk.DISABLED)
            self.tsne_btn.config(state=tk.DISABLED)
            self.save_plots_btn.config(state=tk.DISABLED)
            self.report_btn.config(state=tk.DISABLED)

    def load_model_from_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib"), ("Pickle files", "*.pkl")])
        if path:
            try:
                model = joblib.load(path)
                name = os.path.basename(path).replace('.joblib', '').replace('.pkl', '')
                self.main_app.trained_models[name] = model
                self.update_model_list()
                self.log(f"Модель загружена: {name}")
            except Exception as e:
                self.log(f"Ошибка загрузки: {e}")
                messagebox.showerror("Ошибка", str(e))

    def plot_feature_importance(self):
        if self.current_model is None:
            return
        if hasattr(self.current_model, 'feature_importances_'):
            importances = self.current_model.feature_importances_
        elif hasattr(self.current_model, 'coef_'):
            coef = self.current_model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef[0])
            else:
                importances = np.abs(coef)
        else:
            self.log("Невозможно извлечь важность признаков для этой модели")
            return

        feature_names = self.main_app.feature_names
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        indices = np.argsort(importances)[-20:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {self.current_model_name}')
        fig.tight_layout()
        self._show_plot(fig, self.fi_frame)
        self.feature_importance_fig = fig
        self._figures.append(fig)

    def compute_shap(self):
        if self.current_model is None:
            messagebox.showwarning("Нет модели", "Сначала выберите модель.")
            return
        threading.Thread(target=self._shap_thread, daemon=True).start()

    def _shap_thread(self):
        try:
            self.log("Расчёт SHAP значений... (может занять время)")
            X = self.main_app.X_train
            if X is None or len(X) == 0:
                self.log("Нет обучающих данных для SHAP")
                return
            sample_size = min(self.shap_sample_size.get(), X.shape[0])
            np.random.seed(42)
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            self.X_sample = X[idx]

            model_type = str(type(self.current_model)).lower()
            if 'xgboost' in model_type or 'randomforest' in model_type:
                explainer = shap.TreeExplainer(self.current_model)
                shap_values = explainer.shap_values(self.X_sample)
            elif 'logistic' in model_type or 'linear' in model_type:
                explainer = shap.LinearExplainer(self.current_model, self.X_sample)
                shap_values = explainer.shap_values(self.X_sample)
            else:
                background = shap.sample(self.X_sample, 100)
                if hasattr(self.current_model, 'predict_proba'):
                    explainer = shap.KernelExplainer(self.current_model.predict_proba, background)
                    shap_values = explainer.shap_values(self.X_sample)
                else:
                    explainer = shap.KernelExplainer(self.current_model.predict, background)
                    shap_values = explainer.shap_values(self.X_sample)

            self.shap_values = shap_values
            self.shap_explainer = explainer

            # Summary plot
            fig, ax = plt.subplots(figsize=(12, 8))
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap.summary_plot(shap_values[1], self.X_sample, feature_names=self.main_app.feature_names,
                                  show=False, plot_type='dot', max_display=20)
            else:
                shap.summary_plot(shap_values, self.X_sample, feature_names=self.main_app.feature_names,
                                  show=False, plot_type='dot', max_display=20)
            plt.tight_layout()
            self._show_plot(fig, self.shap_summary_frame)
            self.shap_summary_fig = fig
            self._figures.append(fig)

            # Dependence plots для топ-2 признаков
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean(np.abs(shap_values[1]), axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-2:][::-1]
            top_features = [self.main_app.feature_names[i] for i in top_indices]

            fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
            for i, idx_feat in enumerate(top_indices):
                ax = axes[i]
                if isinstance(shap_values, list):
                    shap.dependence_plot(idx_feat, shap_values[1], self.X_sample,
                                         feature_names=self.main_app.feature_names,
                                         ax=ax, show=False)
                else:
                    shap.dependence_plot(idx_feat, shap_values, self.X_sample,
                                         feature_names=self.main_app.feature_names,
                                         ax=ax, show=False)
                ax.set_title(f"Dependence: {top_features[i]}")
            fig2.tight_layout()
            self._show_plot(fig2, self.shap_dep_frame)
            self.shap_dep_fig = fig2
            self._figures.append(fig2)

            self.log("SHAP анализ завершён.")
        except Exception as e:
            self.log(f"Ошибка SHAP: {e}")
            import traceback
            self.log(traceback.format_exc())

    def plot_tsne(self):
        if self.main_app.X_train is None:
            messagebox.showwarning("Нет данных", "Нет обучающей выборки.")
            return
        threading.Thread(target=self._tsne_thread, daemon=True).start()

    def _tsne_thread(self):
        try:
            self.log("t-SNE визуализация...")
            X = self.main_app.X_train
            y = self.main_app.y_train
            sample_size = min(2000, X.shape[0])
            np.random.seed(42)
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]

            pca = PCA(n_components=min(50, X_sample.shape[1]))
            X_pca = pca.fit_transform(X_sample)

            tsne = TSNE(n_components=2, perplexity=self.tsne_perp.get(), random_state=42, n_jobs=-1)
            X_tsne = tsne.fit_transform(X_pca)

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='coolwarm', alpha=0.6, s=10)
            ax.set_title(f't-SNE projection (perplexity={self.tsne_perp.get()})')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax, label='has_apnea' if self.main_app.task_type == 'classification' else 'AHI')
            fig.tight_layout()
            self._show_plot(fig, self.tsne_frame)
            self.tsne_fig = fig
            self._figures.append(fig)
            self.log("t-SNE завершён.")
        except Exception as e:
            self.log(f"Ошибка t-SNE: {e}")

    def _show_plot(self, fig, target_frame):
        # Закрываем старые фигуры для освобождения памяти
        for widget in target_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                try:
                    plt.close(widget.figure)
                except:
                    pass
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=target_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    def save_plots(self):
        from tkinter import filedialog
        dir_path = filedialog.askdirectory(title="Выберите папку для сохранения графиков")
        if not dir_path:
            return
        saved = []
        for name, fig in [('feature_importance', getattr(self, 'feature_importance_fig', None)),
                          ('shap_summary', getattr(self, 'shap_summary_fig', None)),
                          ('shap_dependence', getattr(self, 'shap_dep_fig', None)),
                          ('tsne', getattr(self, 'tsne_fig', None))]:
            if fig is not None:
                path = os.path.join(dir_path, f"{name}.png")
                fig.savefig(path, dpi=150, bbox_inches='tight')
                saved.append(path)
        self.log(f"Сохранено графиков: {len(saved)} в {dir_path}")

    def generate_report(self):
        if self.current_model is None:
            messagebox.showwarning("Нет модели", "Выберите модель.")
            return
        if self.shap_values is None:
            messagebox.showwarning("SHAP не рассчитан", "Сначала нажмите 'Рассчитать SHAP'.")
            return

        metrics = self.main_app.metrics.get(self.current_model_name, {})

        def fig_to_base64(fig):
            if fig is None:
                return ""
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        fi_img = fig_to_base64(getattr(self, 'feature_importance_fig', None))
        shap_img = fig_to_base64(getattr(self, 'shap_summary_fig', None))
        dep_img = fig_to_base64(getattr(self, 'shap_dep_fig', None))
        tsne_img = fig_to_base64(getattr(self, 'tsne_fig', None))

        # Сводная таблица всех моделей для сравнения
        comparison_df = pd.DataFrame(self.main_app.metrics).T.reset_index()
        comparison_df.rename(columns={'index': 'Model'}, inplace=True)
        # Удаляем служебные колонки
        cols_to_keep = [c for c in comparison_df.columns if not c.startswith('y_')]
        comparison_df = comparison_df[cols_to_keep]
        comparison_html = comparison_df.to_html(index=False, float_format="%.4f")

        # Таблица метрик текущей модели
        metrics_html = "<table border='1'>匙<th>Метрика</th><th>Значение</th></tr>"
        for k, v in metrics.items():
            if not k.startswith('y_'):
                if isinstance(v, float):
                    metrics_html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
                else:
                    metrics_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        metrics_html += "</table>"

        # SHAP топ-5 признаков
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            mean_shap = np.mean(np.abs(shap_vals[1]), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
        top5_idx = np.argsort(mean_shap)[-5:][::-1]
        top5_features = [self.main_app.feature_names[i] for i in top5_idx]
        top5_imp = mean_shap[top5_idx]
        shap_table = "<table border='1'><tr><th>Признак</th><th>Средний |SHAP|</th></tr>"
        for f, imp in zip(top5_features, top5_imp):
            shap_table += f"<tr><td>{f}</td><td>{imp:.4f}</td></tr>"
        shap_table += "</table>"

        # Интерпретация в зависимости от задачи
        if self.main_app.task_type == 'regression':
            interpretation = "Положительный вклад SHAP означает, что признак увеличивает предсказанное значение AHI. Отрицательный – уменьшает."
        else:
            interpretation = "Положительный вклад SHAP означает, что признак увеличивает вероятность класса 'апноэ'. Отрицательный – уменьшает."

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ML отчёт - {self.current_model_name}</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .plot {{ margin: 20px 0; text-align: center; }}
</style>
</head>
<body>
<h1>Отчёт о машинном обучении (раздел 2.6)</h1>
<p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>Модель:</strong> {self.current_model_name}</p>
<p><strong>Задача:</strong> {'Классификация эпох (апноэ vs без)' if self.main_app.task_type == 'classification' else 'Регрессия AHI'}</p>

<h2>Сравнение всех обученных моделей</h2>
{comparison_html}

<h2>Метрики выбранной модели</h2>
{metrics_html}

<h2>Важность признаков (топ-20)</h2>
<div class="plot"><img src="data:image/png;base64,{fi_img}" style="max-width:100%;"/></div>

<h2>SHAP Summary plot</h2>
<div class="plot"><img src="data:image/png;base64,{shap_img}" style="max-width:100%;"/></div>

<h2>SHAP Dependence plots (топ-2 признака)</h2>
<div class="plot"><img src="data:image/png;base64,{dep_img}" style="max-width:100%;"/></div>

<h2>t-SNE проекция обучающей выборки</h2>
<div class="plot"><img src="data:image/png;base64,{tsne_img}" style="max-width:100%;"/></div>

<h2>Топ-5 признаков по SHAP</h2>
{shap_table}

<p><em>Интерпретация:</em> {interpretation}</p>
</body>
</html>"""
        fd, path = tempfile.mkstemp(suffix='.html', prefix='ml_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт: {path}")