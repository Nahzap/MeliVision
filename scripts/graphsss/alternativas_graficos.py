import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
import json
try:
    import seaborn as sns
except Exception:
    sns = None

# Global configuration
plt.rcParams['font.family'] = 'sans-serif'

METRICS_CANONICAL_ORDER = [
    'train_loss', 'val_loss',
    'recall_at_1', 'recall_at_5', 'recall_at_10',
    'mAP@R', 'NMI', 'F1_macro',
    'accuracy', 'balanced_accuracy', 'precision_macro', 'precision_weighted', 'F1_weighted',
    'learning_rate', 'epoch_time_sec', 'best_recall_at_1'
]

OFFICIAL_3D_REFERENCES = [
    "Matplotlib Axes3D API: https://matplotlib.org/stable/api/toolkits/mplot3d/axes3d.html",
    "Matplotlib mplot3d gallery: https://matplotlib.org/stable/gallery/mplot3d/index.html",
    "Matplotlib fill_between3d example: https://matplotlib.org/stable/gallery/mplot3d/fillbetween3d.html",
    "Matplotlib fillunder3d example: https://matplotlib.org/stable/gallery/mplot3d/fillunder3d.html",
    "Matplotlib bars in planes: https://matplotlib.org/stable/gallery/mplot3d/bars3d.html",
    "Matplotlib custom hillshading: https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html",
    "Plotly 3D charts overview: https://plotly.com/python/3d-charts/",
    "Academic reference (multivariate trajectories): http://arxiv.org/pdf/2008.01645v2",
    "Academic reference (trajectory uncertainty tubes): https://arxiv.org/html/2508.13505v1",
]

CONFIG_JSON_PATH = Path(__file__).resolve().parent / "alternativas_graficos_config.json"
MODEL_ORDER_OPTIONS = [
    'As selected',
    'Alphabetical',
    'Reverse alphabetical',
    'ConvNeXt > DINOv2 > DeiT > ResNet',
    'ResNet > DeiT > DINOv2 > ConvNeXt'
]


def _build_default_config(modelos, metricas, default_metrics):
    return {
        "selected_models": list(modelos),
        "selected_metrics": [m for m in metricas if m in default_metrics],
        "metric_priorities": {m: i + 1 for i, m in enumerate(metricas)},
        "normalize": False,
        "order_models": "ConvNeXt > DINOv2 > DeiT > ResNet",
    }


def load_visual_config(modelos, metricas, default_metrics):
    cfg = _build_default_config(modelos, metricas, default_metrics)
    if not CONFIG_JSON_PATH.exists():
        return cfg

    try:
        raw = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read config JSON ({CONFIG_JSON_PATH}): {exc}")
        return cfg

    selected_models = [m for m in raw.get("selected_models", []) if m in modelos]
    if selected_models:
        cfg["selected_models"] = selected_models

    selected_metrics = [m for m in raw.get("selected_metrics", []) if m in metricas]
    if selected_metrics:
        cfg["selected_metrics"] = selected_metrics

    priorities = raw.get("metric_priorities", {})
    if isinstance(priorities, dict):
        for m in metricas:
            val = priorities.get(m)
            try:
                if val is not None:
                    v = int(val)
                    if v > 0:
                        cfg["metric_priorities"][m] = v
            except Exception:
                pass

    cfg["normalize"] = bool(raw.get("normalize", cfg["normalize"]))
    legacy_order_map = {
        "Como seleccionado": "As selected",
        "Alfabético": "Alphabetical",
        "Alfabético inverso": "Reverse alphabetical",
    }
    order_models = raw.get("order_models", cfg["order_models"])
    order_models = legacy_order_map.get(order_models, order_models)
    if order_models in MODEL_ORDER_OPTIONS:
        cfg["order_models"] = order_models

    return cfg


def save_visual_config(selected_models, selected_metrics, metric_priorities, normalize, order_models):
    payload = {
        "selected_models": list(selected_models),
        "selected_metrics": list(selected_metrics),
        "metric_priorities": {k: int(v) for k, v in metric_priorities.items()},
        "normalize": bool(normalize),
        "order_models": str(order_models),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    CONFIG_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Configuration saved to: {CONFIG_JSON_PATH}")


def aplicar_orden(items, modo, kind='metric'):
    items = list(items)
    if modo == 'Alphabetical':
        return sorted(items)
    if modo == 'Reverse alphabetical':
        return sorted(items, reverse=True)
    if modo == 'Canonical (recommended)' and kind == 'metric':
        rank = {m: i for i, m in enumerate(METRICS_CANONICAL_ORDER)}
        return sorted(items, key=lambda x: rank.get(x, 10_000))
    if modo == 'Losses first' and kind == 'metric':
        losses = {'train_loss', 'val_loss'}
        return sorted(items, key=lambda x: (0 if x in losses else 1, x))
    if modo == 'ConvNeXt > DINOv2 > DeiT > ResNet' and kind == 'model':
        pref = ['convnext_v2_tiny', 'dinov2_vits14', 'deit_small_patch16_224', 'resnet50']
        rank = {m: i for i, m in enumerate(pref)}
        return sorted(items, key=lambda x: rank.get(x, 10_000))
    if modo == 'ResNet > DeiT > DINOv2 > ConvNeXt' and kind == 'model':
        pref = ['resnet50', 'deit_small_patch16_224', 'dinov2_vits14', 'convnext_v2_tiny']
        rank = {m: i for i, m in enumerate(pref)}
        return sorted(items, key=lambda x: rank.get(x, 10_000))
    return items


def format_metric_label(metric_name):
    key = str(metric_name).strip().lower().replace(" ", "_")
    table = {
        "train_loss": "Train Loss",
        "val_loss": "Val Loss",
        "recall_at_1": "Recall@1",
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "map@r": "mAP@R",
        "nmi": "NMI",
        "f1_macro": "F1 Macro",
        "f1_weighted": "F1 Weighted",
        "precision_macro": "Precision Macro",
        "precision_weighted": "Precision Weighted",
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Accuracy",
        "learning_rate": "Learning Rate",
        "epoch_time_sec": "Epoch Time (s)",
        "best_recall_at_1": "Best Recall@1",
    }
    return table.get(key, str(metric_name).replace("_", " ").title())


def format_model_label(model_name):
    key = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")
    if "convnext" in key:
        return "ConvNeXt"
    if "dinov2" in key:
        return "DINOv2"
    if "deit" in key:
        return "DeiT"
    if "resnet" in key:
        return "ResNet-50"
    return str(model_name)


def preparar_matrices_3d(
    df,
    modelos_seleccionados,
    metricas_seleccionadas,
    normalizar=False,
    orden_modelos='As selected',
    orden_metricas='As selected'
):
    """
    Build 3D grids with:
      - X: epochs
      - Y: metrics
      - Z: value per model (one sheet per model)
    """
    epocas = sorted(list(df['epoch'].unique()))

    modelos_seleccionados = aplicar_orden(modelos_seleccionados, orden_modelos, kind='model')
    metricas_seleccionadas = aplicar_orden(metricas_seleccionadas, orden_metricas, kind='metric')

    # Keep only existing columns in the wide CSV
    modelos_validos = []
    for modelo in modelos_seleccionados:
        found = any(f"{m}__{modelo}" in df.columns for m in metricas_seleccionadas)
        if found:
            modelos_validos.append(modelo)

    y_labels_display = [format_metric_label(m) for m in metricas_seleccionadas]
    X, Y = np.meshgrid(epocas, range(len(metricas_seleccionadas)))

    # model -> Z matrix (n_metrics x n_epochs)
    z_por_modelo = {}
    for modelo in modelos_validos:
        Zm = np.zeros((len(metricas_seleccionadas), len(epocas)), dtype=float)
        for i, metrica in enumerate(metricas_seleccionadas):
            col = f"{metrica}__{modelo}"
            serie = []
            for e in epocas:
                if col in df.columns:
                    val = df[df['epoch'] == e][col].values
                    serie.append(val[0] if len(val) > 0 else np.nan)
                else:
                    serie.append(np.nan)
            serie = pd.Series(serie).interpolate(limit_direction='both').fillna(0.0).values
            Zm[i, :] = serie
        z_por_modelo[modelo] = Zm

    if normalizar and z_por_modelo:
        # Normalize per metric globally (across models) to compare shapes.
        for i in range(len(metricas_seleccionadas)):
            stack = np.array([z_por_modelo[m][i, :] for m in z_por_modelo.keys()])
            vmin = float(np.nanmin(stack))
            vmax = float(np.nanmax(stack))
            if vmax > vmin:
                for m in z_por_modelo.keys():
                    z_por_modelo[m][i, :] = (z_por_modelo[m][i, :] - vmin) / (vmax - vmin)
            else:
                for m in z_por_modelo.keys():
                    z_por_modelo[m][i, :] = 0.0

    return X, Y, z_por_modelo, epocas, y_labels_display, modelos_validos

def aplicar_estilo_3d(ax, titulo, y_labels, x_values=None):
    """Apply common labels, view angles, and formatting to each 3D axis."""
    ax.set_title(f"Model Metrics Comparison\n{titulo}", fontweight='bold', pad=15, fontsize=12)
    ax.set_xlabel('Epoch (X)', labelpad=10)
    ax.set_ylabel('Metrics (Y)', labelpad=25)
    ax.set_zlabel('Performance (Z)', labelpad=10)
    
    # Adjust label font size based on number of selections
    font_s = 10 if len(y_labels) <= 4 else (8 if len(y_labels) <= 8 else 6)
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, rotation=-15, ha='left', fontsize=font_s)
    
    ax.view_init(elev=25, azim=-55)

    # Integer X ticks every 4 epochs (no decimal grid labels)
    if x_values is not None and len(x_values) > 0:
        xv = sorted({int(round(v)) for v in x_values})
        x_min, x_max = xv[0], xv[-1]
        first_tick = x_min if x_min % 4 == 0 else x_min + (4 - x_min % 4)
        ticks = list(range(first_tick, x_max + 1, 4))
        if x_min not in ticks:
            ticks = [x_min] + ticks
        if x_max not in ticks:
            ticks.append(x_max)
        ticks = sorted(set(ticks))
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    
    # Clean pane backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

class VisualizadorInteractivo3D:
    def __init__(self, X, Y, z_por_modelo, y_labels, modelos):
        self.X = X
        self.Y = Y
        self.z_por_modelo = z_por_modelo
        self.y_labels = y_labels
        self.modelos = modelos

        all_vals = np.concatenate([z.flatten() for z in z_por_modelo.values()]) if z_por_modelo else np.array([0.0])
        self.z_min = float(np.nanmin(all_vals))
        self.z_max = float(np.nanmax(all_vals))
        span = max(self.z_max - self.z_min, 1e-6)
        self.z_floor = self.z_min - span * 0.05

        # Fixed colors per model
        self.colores_modelo = {
            'convnext_v2_tiny': '#2ca02c',   # verde
            'dinov2_vits14': '#1f77b4',      # azul
            'deit_small_patch16_224': '#f1c40f',  # amarillo
            'resnet50': '#d62728',           # rojo
        }
        if sns is not None:
            sns.set_theme(style="whitegrid", context="notebook")
            palette = sns.color_palette("tab10", n_colors=max(4, len(modelos)))
            self.fallback_colores = [tuple(list(c) + [1.0]) for c in palette]
        else:
            self.fallback_colores = plt.cm.tab10(np.linspace(0, 1, max(4, len(modelos))))
        self.indice_actual = 0
        self.alpha_scale = 1.0
        # 20 fully 3D proposals + 2 heatmap exceptions at the end.
        self.total_estilos = 22
        self.output_dir = Path(__file__).resolve().parent / "exports"
        self.current_projection = '3d'
        self.aux_axes = []
        self.aux_texts = []
        
        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.20, left=0.1, right=0.84)
        
        self.actualizar_grafico()
        
        ax_prev = plt.axes([0.32, 0.06, 0.12, 0.05])
        ax_next = plt.axes([0.46, 0.06, 0.12, 0.05])
        ax_alpha = plt.axes([0.60, 0.074, 0.14, 0.018])
        ax_save = plt.axes([0.76, 0.06, 0.14, 0.05])
        
        self.btn_prev = Button(ax_prev, '⬅ Previous')
        self.btn_next = Button(ax_next, 'Next ➔')
        self.btn_save = Button(ax_save, '💾 Save PNG')
        self.alpha_slider = Slider(ax_alpha, 'Alpha', 0.10, 1.00, valinit=1.00, valstep=0.05)
        
        self.btn_prev.on_clicked(self.anterior)
        self.btn_next.on_clicked(self.siguiente)
        self.btn_save.on_clicked(self.guardar_png)
        self.alpha_slider.on_changed(self.on_alpha_change)

    def _ensure_axis(self, projection='3d'):
        target = '3d' if projection == '3d' else '2d'
        if self.current_projection != target:
            self.fig.delaxes(self.ax)
            if target == '3d':
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
            self.current_projection = target
        self.ax.clear()

    def _set_2d_epoch_ticks(self, x_values):
        if x_values is None or len(x_values) == 0:
            return
        xv = sorted({int(round(v)) for v in x_values})
        x_min, x_max = xv[0], xv[-1]
        first_tick = x_min if x_min % 4 == 0 else x_min + (4 - x_min % 4)
        ticks = list(range(first_tick, x_max + 1, 4))
        if x_min not in ticks:
            ticks = [x_min] + ticks
        if x_max not in ticks:
            ticks.append(x_max)
        ticks = sorted(set(ticks))
        self.ax.set_xticks([t - x_min for t in ticks])
        self.ax.set_xticklabels([str(t) for t in ticks], rotation=0)

    def _draw_model_colorbars(self, rows_per_model, total_rows):
        """Draw thin vertical gradient columns aligned to each model heatmap block."""
        self._clear_aux_axes()

        n_models = max(1, len(self.modelos))
        bbox = self.ax.get_position()  # figure-relative axis box
        gap_x = 0.008
        col_w = 0.016  # thin columns
        x_col = bbox.x1 + gap_x

        # Vertical gradient: top=100% (full color), bottom=0% (white).
        grad = np.linspace(0.0, 1.0, 256, dtype=float).reshape(-1, 1)

        for i, modelo in enumerate(self.modelos):
            start = i * rows_per_model
            end = start + rows_per_model
            top_frac = 1.0 - (start / float(total_rows))
            bot_frac = 1.0 - (end / float(total_rows))
            block_bottom = bbox.y0 + bot_frac * bbox.height
            block_height = max(0.001, (top_frac - bot_frac) * bbox.height)
            pad = block_height * 0.06
            bottom = block_bottom + pad
            height = max(0.001, block_height - 2 * pad)

            lax = self.fig.add_axes([x_col, bottom, col_w, height])
            lax.imshow(
                grad,
                aspect='auto',
                cmap=self._model_cmap(modelo, i),
                interpolation='nearest',
                origin='lower'
            )
            lax.set_xticks([])
            lax.set_yticks([])
            for sp in lax.spines.values():
                sp.set_visible(False)
            lax.text(
                1.95, 0.5,
                format_model_label(modelo),
                transform=lax.transAxes,
                ha='left',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='black'
            )
            lax.text(1.22, 1.00, "100%", transform=lax.transAxes, ha='left', va='top', fontsize=6, color='black')
            lax.text(1.22, 0.00, "0%", transform=lax.transAxes, ha='left', va='bottom', fontsize=6, color='black')
            self.aux_axes.append(lax)

        # Global vertical labels (black, non-overlapping).
        t1 = self.fig.text(
            x_col + col_w + 0.075, bbox.y0 + bbox.height * 0.62,
            "Performance (%)",
            rotation=90, ha='center', va='center', fontsize=9, color='black'
        )
        t2 = self.fig.text(
            x_col + col_w + 0.100, bbox.y0 + bbox.height * 0.62,
            "Model",
            rotation=90, ha='center', va='center', fontsize=9, color='black'
        )
        self.aux_texts.extend([t1, t2])

    def _clear_aux_axes(self):
        for ax in self.aux_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        self.aux_axes = []
        for txt in self.aux_texts:
            try:
                txt.remove()
            except Exception:
                pass
        self.aux_texts = []

    def _build_performance_by_model(self):
        """
        Convert raw metric values into a comparable performance scale [0, 100].
        Loss-like metrics are inverted (lower loss => higher performance).
        """
        perf = {m: np.array(self.z_por_modelo[m], dtype=float).copy() for m in self.modelos}
        if not perf:
            return perf

        n_metrics = perf[self.modelos[0]].shape[0]
        for r in range(n_metrics):
            stacked = np.array([perf[m][r, :] for m in self.modelos], dtype=float)
            vmin = float(np.nanmin(stacked))
            vmax = float(np.nanmax(stacked))
            span = vmax - vmin
            metric_name = str(self.y_labels[r]).lower()
            is_loss = "loss" in metric_name

            if span <= 1e-12:
                for m in self.modelos:
                    perf[m][r, :] = 50.0
                continue

            for m in self.modelos:
                norm = (perf[m][r, :] - vmin) / span
                if is_loss:
                    norm = 1.0 - norm
                perf[m][r, :] = np.clip(norm * 100.0, 0.0, 100.0)

        return perf

    def siguiente(self, event):
        self.indice_actual = (self.indice_actual + 1) % self.total_estilos
        self.actualizar_grafico()

    def anterior(self, event):
        self.indice_actual = (self.indice_actual - 1) % self.total_estilos
        self.actualizar_grafico()

    def color_modelo(self, modelo, idx):
        key = str(modelo).strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "convnext_v2_tiny": "convnext_v2_tiny",
            "convnextv2_tiny": "convnext_v2_tiny",
            "resnet50": "resnet50",
            "resnet_50": "resnet50",
            "deit_small_patch16_224": "deit_small_patch16_224",
            "deit_s": "deit_small_patch16_224",
            "deits": "deit_small_patch16_224",
            "dinov2_vits14": "dinov2_vits14",
            "dinov2_vit_s_14": "dinov2_vits14",
            "dinov2_vit_s14": "dinov2_vits14",
            "dinov2_vit_s": "dinov2_vits14",
        }
        canonical = alias_map.get(key, key)
        if canonical in self.colores_modelo:
            return self.colores_modelo[canonical]
        return self.fallback_colores[idx]

    def _y_shift(self, mi):
        n = max(1, len(self.modelos))
        return (mi - (n - 1) / 2.0) * 0.14

    def _model_cmap(self, modelo, idx):
        base = self.color_modelo(modelo, idx)
        return mcolors.LinearSegmentedColormap.from_list(
            f"{modelo}_mono",
            [(1.0, 1.0, 1.0), mcolors.to_rgb(base)]
        )

    def _a(self, base_alpha):
        return max(0.02, min(1.0, float(base_alpha) * float(self.alpha_scale)))

    def on_alpha_change(self, value):
        self.alpha_scale = float(value)
        self.actualizar_grafico()

    def guardar_png(self, event):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"visual_3d_{self.indice_actual+1:02d}_{ts}.png"
        self.fig.savefig(out, dpi=320, bbox_inches='tight')
        print(f"[OK] Image saved to: {out}")

    def actualizar_grafico(self):
        idx = self.indice_actual
        X, Y = self.X, self.Y
        self._clear_aux_axes()

        if idx >= 20:
            self._ensure_axis(projection='2d')
            epoch_values = [int(round(v)) for v in X[0, :]]
            epoch_sort_idx = np.argsort(epoch_values)
            epoch_sorted = [epoch_values[i] for i in epoch_sort_idx]
            perf_by_model = self._build_performance_by_model()

            if idx == 20:
                # Heatmap 1: model blocks with model-specific palettes
                rows_per_model = len(self.y_labels)
                total_rows = rows_per_model * len(self.modelos)
                n_epochs = len(epoch_sorted)
                row_labels = []
                last_im = None

                for mi, modelo in enumerate(self.modelos):
                    start = mi * rows_per_model
                    end = start + rows_per_model
                    block = perf_by_model[modelo][:, epoch_sort_idx]
                    last_im = self.ax.imshow(
                        block,
                        aspect='auto',
                        cmap=self._model_cmap(modelo, mi),
                        vmin=0.0,
                        vmax=100.0,
                        alpha=self._a(1.0),
                        interpolation='nearest',
                        extent=(-0.5, n_epochs - 0.5, end - 0.5, start - 0.5),
                        origin='upper'
                    )
                    row_labels.extend(list(self.y_labels))

                    # Visual separators and compact model tag.
                    if mi > 0:
                        self.ax.axhline(start - 0.5, color='white', linewidth=1.0, alpha=0.95)
                    y_frac = 1.0 - ((start + rows_per_model / 2.0) / float(total_rows))
                    self.ax.text(
                        -0.075, y_frac,
                        format_model_label(modelo),
                        transform=self.ax.transAxes,
                        ha='right',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color=self.color_modelo(modelo, mi)
                    )

                self.ax.set_xlabel("Epoch")
                self.ax.set_ylabel("Metric")
                self.ax.set_yticks(np.arange(total_rows))
                self.ax.set_yticklabels(row_labels, fontsize=7)
                self._set_2d_epoch_ticks(epoch_sorted)
                self.ax.set_ylim(total_rows - 0.5, -0.5)
                self.ax.grid(False)
                title = f"Model Metrics Comparison\n Performance heatmaps"
                self.ax.set_title(title, fontweight='bold', fontsize=12)
                self._draw_model_colorbars(rows_per_model=rows_per_model, total_rows=total_rows)

            elif idx == 21:
                # Heatmap 2: per-model heat strips (mean performance) with model palettes
                matrix_rows = []
                row_labels = []
                for modelo in self.modelos:
                    Zp = perf_by_model[modelo]
                    matrix_rows.append(np.mean(Zp, axis=0)[epoch_sort_idx])
                    row_labels.append(modelo)
                M = np.array(matrix_rows, dtype=float)
                n_models = len(self.modelos)
                n_epochs = len(epoch_sorted)
                last_im = None
                for mi, modelo in enumerate(self.modelos):
                    row = M[mi:mi + 1, :]
                    last_im = self.ax.imshow(
                        row,
                        aspect='auto',
                        cmap=self._model_cmap(modelo, mi),
                        vmin=0.0,
                        vmax=100.0,
                        alpha=self._a(1.0),
                        interpolation='nearest',
                        extent=(-0.5, n_epochs - 0.5, mi + 0.5, mi - 0.5),
                        origin='upper'
                    )
                    if mi > 0:
                        self.ax.axhline(mi - 0.5, color='white', linewidth=1.0, alpha=0.95)

                self.ax.set_xlabel("Epoch")
                self.ax.set_ylabel("Model")
                self.ax.set_yticks(np.arange(n_models))
                self.ax.set_yticklabels([format_model_label(m) for m in row_labels], fontsize=9)
                self._set_2d_epoch_ticks(epoch_sorted)
                self.ax.set_ylim(n_models - 0.5, -0.5)
                self.ax.grid(False)
                title = f"Model Metrics Comparison\n[{idx+1}/{self.total_estilos}] heatmap (model strips with per-model palettes)"
                self.ax.set_title(title, fontweight='bold', fontsize=12)
                self._draw_model_colorbars(rows_per_model=1, total_rows=n_models)

            plt.draw()
            return

        self._ensure_axis(projection='3d')

        if idx == 0:
            # Official: Axes3D.plot_surface
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.plot_surface(
                    X, Ys, Z,
                    color=self.color_modelo(modelo, mi),
                    alpha=self._a(0.78),
                    linewidth=0.2,
                    edgecolor=self.color_modelo(modelo, mi),
                    antialiased=True
                )
            titulo = f'[{idx+1}/{self.total_estilos}] plot_surface (one sheet per model)'

        elif idx == 1:
            # Official: Axes3D.plot_wireframe
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.plot_wireframe(
                    X, Ys, Z,
                    color=self.color_modelo(modelo, mi),
                    linewidth=1.1,
                    alpha=self._a(0.95)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] plot_wireframe'

        elif idx == 2:
            # Official: Axes3D.plot_trisurf
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                self.ax.plot_trisurf(
                    X.flatten(), Ys.flatten(), Z.flatten(),
                    color=color, alpha=self._a(0.55), linewidth=0.35, edgecolor=color
                )
            titulo = f'[{idx+1}/{self.total_estilos}] plot_trisurf'

        elif idx == 3:
            # Official: Axes3D.contour
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.contour(
                    X, Ys, Z,
                    levels=16,
                    colors=[self.color_modelo(modelo, mi)],
                    linewidths=1.1,
                    alpha=self._a(0.9)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] contour3D'

        elif idx == 4:
            # Official: Axes3D.contourf with floor projection
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                self.ax.contourf(
                    X, Ys, Z,
                    zdir='z',
                    offset=self.z_floor,
                    levels=12,
                    colors=[color],
                    alpha=self._a(0.18)
                )
                self.ax.plot_surface(X, Ys, Z, color=color, alpha=self._a(0.48), linewidth=0.2, edgecolor=color)
            titulo = f'[{idx+1}/{self.total_estilos}] contourf3D projected'

        elif idx == 5:
            # Official: Axes3D.scatter
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                self.ax.scatter(X.flatten(), Ys.flatten(), Z.flatten(), color=color, s=12, alpha=self._a(0.8))
            titulo = f'[{idx+1}/{self.total_estilos}] scatter3D'

        elif idx == 6:
            # Official: Axes3D.bar3d
            n_models = max(1, len(self.modelos))
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                x = X.flatten()
                y = Y.flatten() + (mi - (n_models - 1) / 2.0) * 0.18
                z0 = np.full_like(x, self.z_floor, dtype=float)
                dx = np.full_like(x, 0.32, dtype=float)
                dy = np.full_like(x, 0.14, dtype=float)
                dz = Z.flatten() - self.z_floor
                self.ax.bar3d(
                    x, y, z0, dx, dy, dz,
                    color=color,
                    alpha=self._a(0.35),
                    shade=True,
                    edgecolor=color,
                    linewidth=0.25
                )
            titulo = f'[{idx+1}/{self.total_estilos}] bar3d'

        elif idx == 7:
            # Official: Axes3D.stem
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                markerline, stemlines, baseline = self.ax.stem(
                    X.flatten(),
                    Ys.flatten(),
                    Z.flatten(),
                    linefmt='-',
                    markerfmt='o',
                    basefmt=' '
                )
                markerline.set_color(color)
                markerline.set_markersize(3)
                markerline.set_alpha(self._a(0.55))
                stemlines.set_color(color)
                stemlines.set_alpha(self._a(0.30))
                stemlines.set_linewidth(0.7)
            titulo = f'[{idx+1}/{self.total_estilos}] stem3D'

        elif idx == 8:
            # Official: Axes3D.quiver
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                # Approximate local direction field from grid gradient
                gx, gy = np.gradient(Z)
                U = np.ones_like(Z) * 0.9
                V = np.zeros_like(Z)
                W = gx * 0.45 + gy * 0.25
                self.ax.quiver(
                    X, Ys, Z,
                    U, V, W,
                    length=0.6,
                    normalize=True,
                    color=color,
                    alpha=self._a(0.55)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] quiver3D'

        elif idx == 9:
            # Official: Axes3D.plot (3D line plot)
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                for r in range(Z.shape[0]):
                    self.ax.plot(X[r, :], Ys[r, :], Z[r, :], color=color, lw=1.7, alpha=self._a(0.95))
            titulo = f'[{idx+1}/{self.total_estilos}] plot (3D line)'

        elif idx == 10:
            # Official: Axes3D.contour with extend3d
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.contour(
                    X, Ys, Z,
                    levels=12,
                    colors=[self.color_modelo(modelo, mi)],
                    linewidths=1.0,
                    alpha=self._a(0.85),
                    extend3d=True
                )
            titulo = f'[{idx+1}/{self.total_estilos}] contour3D (extend3d)'

        elif idx == 11:
            # Official: Axes3D.tricontour
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.tricontour(
                    X.flatten(), Ys.flatten(), Z.flatten(),
                    levels=14,
                    colors=[self.color_modelo(modelo, mi)],
                    linewidths=1.1,
                    alpha=self._a(0.85)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] tricontour3D'

        elif idx == 12:
            # Official: Axes3D.tricontourf
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                self.ax.tricontourf(
                    X.flatten(), Ys.flatten(), Z.flatten(),
                    levels=10,
                    colors=[color],
                    alpha=self._a(0.35)
                )
                self.ax.tricontour(
                    X.flatten(), Ys.flatten(), Z.flatten(),
                    levels=10,
                    colors=[color],
                    linewidths=0.8,
                    alpha=self._a(0.65)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] tricontourf3D'

        elif idx == 13:
            # Official: Axes3D.plot_surface with per-model colormap
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                self.ax.plot_surface(
                    X, Ys, Z,
                    cmap=self._model_cmap(modelo, mi),
                    alpha=self._a(0.72),
                    linewidth=0.25,
                    antialiased=True
                )
            titulo = f'[{idx+1}/{self.total_estilos}] plot_surface (colormap)'

        elif idx == 14:
            # Official: Axes3D.errorbar (aggregated per epoch)
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                z_mean = np.mean(Z, axis=0)
                z_std = np.std(Z, axis=0)
                y_center = np.full_like(z_mean, fill_value=(len(self.y_labels)-1)/2, dtype=float) + self._y_shift(mi)
                self.ax.errorbar(
                    X[0, :], y_center, z_mean,
                    zerr=z_std,
                    fmt='-o',
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2,
                    alpha=self._a(0.9)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] errorbar3D (mean±std)'

        elif idx == 15:
            # Official: Axes3D.fill_between + bound lines
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                low = np.min(Z, axis=0)
                high = np.max(Z, axis=0)
                y_band = np.full_like(low, fill_value=(len(self.y_labels)-1)/2, dtype=float) + self._y_shift(mi)
                self.ax.fill_between(
                    X[0, :], y_band, low,
                    X[0, :], y_band, high,
                    color=color,
                    alpha=self._a(0.18)
                )
                self.ax.plot(X[0, :], y_band, high, color=color, lw=1.4, alpha=self._a(0.9))
                self.ax.plot(X[0, :], y_band, low, color=color, lw=1.1, alpha=self._a(0.7), linestyle='--')
            titulo = f'[{idx+1}/{self.total_estilos}] fill_between3D (envelope)'

        elif idx == 16:
            # Official: Axes3D.fill_between (fill under 3D lines)
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                for r in range(Z.shape[0]):
                    self.ax.fill_between(
                        X[r, :], Ys[r, :], Z[r, :],
                        X[r, :], Ys[r, :], np.full_like(Z[r, :], self.z_floor),
                        facecolors=color, alpha=self._a(0.22)
                    )
                    self.ax.plot(X[r, :], Ys[r, :], Z[r, :], color=color, lw=1.1, alpha=self._a(0.9))
            titulo = f'[{idx+1}/{self.total_estilos}] fill_under3D lines'

        elif idx == 17:
            # Official: Axes3D.bar (2D bars in different planes)
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                # Use metric-mean profile across epochs on separate y-planes.
                z_profile = np.mean(Z, axis=0)
                y_plane = float(mi) + 0.5
                self.ax.bar(
                    X[0, :], z_profile,
                    zs=y_plane, zdir='y',
                    color=color, alpha=self._a(0.72), width=0.65,
                    edgecolor=color, linewidth=0.25
                )
            # Align y ticks with model planes for readability in this mode.
            self.ax.set_yticks([float(i) + 0.5 for i in range(len(self.modelos))])
            self.ax.set_yticklabels(self.modelos, fontsize=8)
            titulo = f'[{idx+1}/{self.total_estilos}] bar in y-planes'

        elif idx == 18:
            # Official: custom hillshaded 3D surfaces using LightSource + plot_surface
            ls = LightSource(azdeg=270, altdeg=45)
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                Ys = Y + self._y_shift(mi)
                cmap = self._model_cmap(modelo, mi)
                rgb = ls.shade(Z, cmap=cmap, vert_exag=0.6, blend_mode='soft')
                self.ax.plot_surface(
                    X, Ys, Z,
                    facecolors=rgb,
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                    alpha=self._a(0.96)
                )
            titulo = f'[{idx+1}/{self.total_estilos}] custom hillshaded surface'

        elif idx == 19:
            # Official: contour projections on side planes (x/y walls)
            x_min = float(np.min(X))
            for mi, modelo in enumerate(self.modelos):
                Z = self.z_por_modelo[modelo]
                color = self.color_modelo(modelo, mi)
                Ys = Y + self._y_shift(mi)
                self.ax.plot_surface(X, Ys, Z, color=color, alpha=self._a(0.22), linewidth=0, edgecolor='none')
                self.ax.contour(X, Ys, Z, zdir='z', offset=self.z_floor, levels=10, colors=[color], linewidths=0.9, alpha=self._a(0.9))
                self.ax.contour(X, Ys, Z, zdir='x', offset=x_min, levels=10, colors=[color], linewidths=0.8, alpha=self._a(0.65))
            titulo = f'[{idx+1}/{self.total_estilos}] contour projections on walls'

        self.ax.set_zlim(self.z_floor, self.z_max * 1.05 if self.z_max > 0 else 1.0)
        aplicar_estilo_3d(self.ax, titulo, self.y_labels, x_values=X[0, :])
        if idx == 17:
            # This mode uses model planes on Y instead of metric rows.
            self.ax.set_ylabel('Models (Y)', labelpad=25)
            self.ax.set_yticks([float(i) + 0.5 for i in range(len(self.modelos))])
            self.ax.set_yticklabels(self.modelos, fontsize=8)
        # Fixed legend for models
        handles = []
        for mi, modelo in enumerate(self.modelos):
            handles.append(
                plt.Line2D([0], [0], color=self.color_modelo(modelo, mi), lw=3, label=modelo)
            )
        self.ax.legend(handles=handles, title='Models', loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9)
        plt.draw()

def iniciar_interfaz_configuracion(ruta_archivo):
    """Open a Tkinter window to select filters before building 3D matrices."""
    try:
        df = pd.read_csv(ruta_archivo)
    except Exception as e:
        print(f"Error loading file {ruta_archivo}: {e}")
        return

    # Dynamically extract all metrics and models
    modelos_set = set()
    metricas_set = set()
    for col in df.columns:
        if '__' in col:
            partes = col.split('__')
            if len(partes) == 2:
                metricas_set.add(partes[0])
                modelos_set.add(partes[1])

    modelos = sorted(list(modelos_set))
    metricas = sorted(list(metricas_set))

    # --- Configure Tkinter window ---
    root = tk.Tk()
    root.title("3D Visualization Configuration")
    root.geometry("650x500")
    
    defaults = {
        'train_loss', 'val_loss', 'recall_at_1', 'recall_at_5', 'recall_at_10',
        'mAP@R', 'NMI', 'F1_macro', 'accuracy', 'balanced_accuracy',
        'precision_macro', 'precision_weighted', 'F1_weighted'
    }
    loaded_cfg = load_visual_config(modelos, metricas, defaults)

    # State variables (loaded from JSON memory)
    mod_vars = {m: tk.BooleanVar(value=(m in loaded_cfg["selected_models"])) for m in modelos}
    met_vars = {m: tk.BooleanVar(value=(m in loaded_cfg["selected_metrics"])) for m in metricas}
    var_normalizar = tk.BooleanVar(value=bool(loaded_cfg["normalize"]))

    lbl_info = tk.Label(
        root,
        text="Select data for official 3D proposals (numeric priority per metric):",
        font=('Arial', 12, 'bold')
    )
    lbl_info.pack(pady=10)

    # Main split frame
    frame_principal = tk.Frame(root)
    frame_principal.pack(fill='both', expand=True, padx=20)

    # Left column: models
    frame_izq = tk.LabelFrame(frame_principal, text="1. Select Models", font=('Arial', 10, 'bold'))
    frame_izq.pack(side='left', fill='both', expand=True, padx=5)
    for m in modelos:
        tk.Checkbutton(frame_izq, text=m, variable=mod_vars[m]).pack(anchor='w', padx=5, pady=2)

    # Right column: metrics + numeric priority
    frame_der = tk.LabelFrame(frame_principal, text="2. Select Metrics", font=('Arial', 10, 'bold'))
    frame_der.pack(side='right', fill='both', expand=True, padx=5)
    
    canvas_met = tk.Canvas(frame_der, borderwidth=0, highlightthickness=0)
    scroll_met = tk.Scrollbar(frame_der, orient="vertical", command=canvas_met.yview)
    scroll_frame = tk.Frame(canvas_met)
    
    scroll_frame.bind("<Configure>", lambda e: canvas_met.configure(scrollregion=canvas_met.bbox("all")))
    canvas_met.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas_met.configure(yscrollcommand=scroll_met.set)
    
    canvas_met.pack(side="left", fill="both", expand=True)
    scroll_met.pack(side="right", fill="y")
    
    met_priority_vars = {
        m: tk.IntVar(value=int(loaded_cfg["metric_priorities"].get(m, i + 1)))
        for i, m in enumerate(metricas)
    }

    header = tk.Frame(scroll_frame)
    header.pack(fill='x', padx=5, pady=(2, 4))
    tk.Label(header, text="Metric", font=('Arial', 9, 'bold')).pack(side='left', anchor='w')
    tk.Label(header, text="Priority", font=('Arial', 9, 'bold')).pack(side='right', anchor='e')

    for m in metricas:
        row = tk.Frame(scroll_frame)
        row.pack(fill='x', padx=5, pady=1)
        tk.Checkbutton(row, text=m, variable=met_vars[m]).pack(side='left', anchor='w')
        tk.Spinbox(
            row,
            from_=1,
            to=999,
            width=5,
            textvariable=met_priority_vars[m],
            justify='center'
        ).pack(side='right', anchor='e')

    # Extra options
    frame_opciones = tk.Frame(root)
    frame_opciones.pack(fill='x', padx=20, pady=10)
    tk.Checkbutton(frame_opciones, text="Normalize values to (0 to 1)\n(Recommended when mixing metrics with different scales, e.g., Accuracy vs Time)", 
                   variable=var_normalizar, font=('Arial', 9, 'italic')).pack(anchor='center')

    frame_orden = tk.LabelFrame(root, text="3. Display Order", font=('Arial', 10, 'bold'))
    frame_orden.pack(fill='x', padx=20, pady=6)
    var_orden_modelos = tk.StringVar(value=loaded_cfg["order_models"])
    var_orden_metricas = tk.StringVar(value='Canonical (recommended)')

    tk.Label(frame_orden, text="Model order:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=6, pady=4)
    tk.OptionMenu(frame_orden, var_orden_modelos, *MODEL_ORDER_OPTIONS).grid(row=0, column=1, sticky='w', padx=6, pady=4)

    tk.Label(
        frame_orden,
        text="Metric order is defined by numeric priority inputs",
        font=('Arial', 9, 'italic')
    ).grid(row=1, column=0, columnspan=2, sticky='w', padx=6, pady=4)

    # Generate button action
    def on_generar():
        mods_sel = [m for m, var in mod_vars.items() if var.get()]
        mets_sel_raw = [m for m, var in met_vars.items() if var.get()]

        if not mods_sel:
            messagebox.showwarning("Warning", "You must select at least one model.")
            return
        if not mets_sel_raw:
            messagebox.showwarning("Warning", "You must select at least one metric.")
            return

        # Order by numeric priority (smaller integer = higher priority)
        try:
            mets_sel = sorted(
                mets_sel_raw,
                key=lambda m: (int(met_priority_vars[m].get()), metricas.index(m))
            )
        except Exception:
            messagebox.showerror("Error", "Invalid priorities. Use integer values in priority fields.")
            return

        # Persist selected configuration for future runs
        try:
            save_visual_config(
                selected_models=mods_sel,
                selected_metrics=mets_sel_raw,
                metric_priorities={m: int(met_priority_vars[m].get()) for m in metricas},
                normalize=var_normalizar.get(),
                order_models=var_orden_modelos.get(),
            )
        except Exception as exc:
            print(f"[WARN] Could not save JSON configuration: {exc}")

        # Close window and process
        root.destroy()
        
        print("Processing and filtering data based on current selection...")
        X, Y, z_por_modelo, epocas, y_labels, modelos_validos = preparar_matrices_3d(
            df,
            mods_sel,
            mets_sel,
            normalizar=var_normalizar.get(),
            orden_modelos=var_orden_modelos.get(),
            # Numeric priority defines final metric order.
            orden_metricas='As selected'
        )
        if len(modelos_validos) == 0:
            messagebox.showerror("Error", "No matching columns were found for the current selection.")
            return
        
        print(f"Launching 3D viewer with {len(modelos_validos)} model sheets and {len(y_labels)} metrics.")
        visor = VisualizadorInteractivo3D(X, Y, z_por_modelo, y_labels, modelos_validos)
        plt.show()

    btn_generar = tk.Button(root, text="🚀 Filter and Generate 3D Visualization", 
                            command=on_generar, bg='#2563eb', fg='white', font=('Arial', 12, 'bold'))
    btn_generar.pack(pady=15)

    def on_fuentes():
        messagebox.showinfo("Official sources used", "\n\n".join(OFFICIAL_3D_REFERENCES))

    btn_fuentes = tk.Button(
        root,
        text="📚 View Official Sources",
        command=on_fuentes,
        bg='#111827',
        fg='white',
        font=('Arial', 10, 'bold')
    )
    btn_fuentes.pack(pady=(0, 8))

    # Start UI loop
    root.mainloop()

if __name__ == "__main__":
    # Read your wide-format CSV file
    archivo_csv = r'C:\Users\askna\Documents\GitHub\MetricLearning\scripts\graphsss\epoch_model_metrics_wide_a4.csv'
    iniciar_interfaz_configuracion(archivo_csv)