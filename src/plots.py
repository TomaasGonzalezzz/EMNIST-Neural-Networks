"""Utilidades de plotting para exploracion y evaluacion de models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _como_imagen(imagen: np.ndarray) -> np.ndarray:
    """Devuelve un sample de EMNIST como imagen 28x28."""

    imagen = np.asarray(imagen)
    if imagen.ndim == 1:
        return imagen.reshape(28, 28)
    return imagen


def graficar_samples(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    indices: list[int] | np.ndarray | None = None,
    n: int = 3,
    random_state: int = 7,
    title: str = "Samples del dataset",
    cmap: str = "gray",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica una grilla chica de samples de EMNIST."""

    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=n, replace=False)
    else:
        indices = np.asarray(indices)
        n = len(indices)

    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, indices):
        ax.imshow(_como_imagen(X[int(idx)]), cmap=cmap)
        label_text = f" | class {int(y[int(idx)])}" if y is not None else ""
        ax.set_title(f"idx {int(idx)}{label_text}")
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes


def graficar_distribucion_clases(
    distribucion: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Class distribution",
) -> tuple[plt.Figure, plt.Axes]:
    """Grafica counts por class desde ``data.distribucion_clases``."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.bar(distribucion["clase"], distribucion["cantidad"], color="#496f5d")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(distribucion["clase"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def graficar_history_training(
    historial: dict[str, list[float]],
    *,
    metric: str = "loss",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grafica curvas train/validation guardadas durante training."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    for key, valores in historial.items():
        if key.endswith(metric):
            epochs = historial.get("epoch", range(1, len(valores) + 1))
            ax.plot(
                epochs,
                valores,
                marker="o",
                markersize=2.2,
                linewidth=1.9,
                alpha=0.9,
                label=key,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def graficar_loss_y_accuracy(
    historial: dict[str, list[float]],
    *,
    title: str = "Training evolution",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica cross-entropy y accuracy lado a lado."""

    epochs = historial.get("epoch", range(1, len(historial["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    curve_style = {"marker": "o", "markersize": 2.2, "linewidth": 1.9, "alpha": 0.9}

    axes[0].plot(epochs, historial["train_loss"], label="train", **curve_style)
    axes[0].plot(epochs, historial["validation_loss"], label="validation", **curve_style)
    axes[0].set_title("Cross-entropy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, historial["train_accuracy"], label="train", **curve_style)
    axes[1].plot(epochs, historial["validation_accuracy"], label="validation", **curve_style)
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes


def graficar_confusion_matrix(
    matriz: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Confusion matrix",
    normalizar: bool = False,
    annotate: bool = False,
    annotation_fontsize: int = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """Grafica una confusion matrix como heatmap de Matplotlib."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 16) if annotate else (7, 6))
    else:
        fig = ax.figure

    valores = np.asarray(matriz, dtype=float)
    if normalizar:
        totales_fila = valores.sum(axis=1, keepdims=True)
        valores = np.divide(valores, np.maximum(totales_fila, 1.0))

    imagen = ax.imshow(valores, cmap="Blues", vmin=0)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(valores.shape[1]))
    ax.set_yticks(np.arange(valores.shape[0]))
    ax.set_xticklabels(np.arange(valores.shape[1]))
    ax.set_yticklabels(np.arange(valores.shape[0]))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7 if annotate else 8)
    ax.tick_params(axis="y", labelsize=7 if annotate else 8)

    if annotate:
        umbral = valores.max() / 2 if valores.size else 0
        for fila in range(valores.shape[0]):
            for col in range(valores.shape[1]):
                if normalizar:
                    label = f"{valores[fila, col]:.2f}"
                else:
                    label = f"{int(matriz[fila, col])}"
                color = "white" if valores[fila, col] > umbral else "black"
                ax.text(
                    col,
                    fila,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=annotation_fontsize,
                )

    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def tablas_resumen_confusion(
    matriz: np.ndarray,
    *,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve metrics por class y confusions off-diagonal mas frecuentes."""

    matriz = np.asarray(matriz)
    labels = np.arange(matriz.shape[0])
    soporte = matriz.sum(axis=1)
    predicho = matriz.sum(axis=0)
    verdaderos_positivos = np.diag(matriz)

    por_clase = pd.DataFrame(
        {
            "clase": labels,
            "support": soporte,
            "correctas": verdaderos_positivos,
            "recall": np.divide(
                verdaderos_positivos,
                np.maximum(soporte, 1),
            ),
            "precision": np.divide(
                verdaderos_positivos,
                np.maximum(predicho, 1),
            ),
        }
    )
    por_clase["errores"] = por_clase["support"] - por_clase["correctas"]

    fuera_diagonal = matriz.copy()
    np.fill_diagonal(fuera_diagonal, 0)
    filas, cols = np.nonzero(fuera_diagonal)
    conteos = fuera_diagonal[filas, cols]
    orden = np.argsort(conteos)[::-1][:top_n]
    confusiones = pd.DataFrame(
        {
            "real": filas[orden],
            "predicha": cols[orden],
            "cantidad": conteos[orden],
        }
    )
    confusiones["par"] = (
        confusiones["real"].astype(str) + " -> " + confusiones["predicha"].astype(str)
    )
    return por_clase, confusiones


def graficar_confusion_diagnostics(
    matriz: np.ndarray,
    *,
    top_n: int = 20,
    title: str = "Confusion diagnostics",
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Grafica diagnostics legibles derivados de una confusion matrix."""

    por_clase, confusiones = tablas_resumen_confusion(matriz, top_n=top_n)
    peor_recall = por_clase.sort_values("recall").head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    axes[0].bar(por_clase["clase"], por_clase["recall"], color="#335c67")
    axes[0].set_title("Recall by class", fontsize=14)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].barh(
        peor_recall["clase"].astype(str),
        peor_recall["recall"],
        color="#9a031e",
    )
    axes[1].set_title(f"{top_n} classes with lowest recall", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(confusiones["par"], confusiones["cantidad"], color="#7f4f24")
    axes[2].set_title(f"Top {top_n} confusions", fontsize=14)
    axes[2].set_xlabel("Count")
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes, por_clase, confusiones


def graficar_grilla_metrics(
    resumen: pd.DataFrame,
    *,
    title: str = "Model comparison",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica cross-entropy, accuracy y F1 para todos los experimentos."""

    ordenado = resumen.sort_values("validation_loss").reset_index(drop=True)
    x = np.arange(len(ordenado))
    ancho = 0.38

    fig, axes = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    especificaciones = [
        ("loss", "Cross-entropy", "lower is better"),
        ("accuracy", "Accuracy", "higher is better"),
        ("f1_macro", "F1 macro", "higher is better"),
    ]

    for ax, (metric, label, subtitle) in zip(axes, especificaciones):
        train_col = f"train_{metric}"
        validation_col = f"validation_{metric}"
        ax.bar(
            x - ancho / 2,
            ordenado[train_col],
            ancho,
            label="train",
            color="#7f4f24",
        )
        ax.bar(
            x + ancho / 2,
            ordenado[validation_col],
            ancho,
            label="validation",
            color="#335c67",
        )
        ax.set_title(f"{label} ({subtitle})", fontsize=14)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Model")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(ordenado["model_id"])
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    return fig, axes


def graficar_dashboard_model_selection(
    resumen: pd.DataFrame,
    *,
    title: str = "Model selection dashboard",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica diagnostics de model selection sin etiquetas amontonadas."""

    ordenado = resumen.sort_values("validation_loss").reset_index(drop=True)
    columnas_heatmap = [
        "validation_loss",
        "validation_accuracy",
        "validation_f1_macro",
        "total_seconds",
        "parameter_count",
    ]
    normalizado = ordenado[columnas_heatmap].copy()
    denominador_loss = normalizado["validation_loss"].max() - normalizado["validation_loss"].min()
    if denominador_loss > 0:
        normalizado["validation_loss"] = 1 - (
            (normalizado["validation_loss"] - normalizado["validation_loss"].min())
            / denominador_loss
        )
    for column in [
        "validation_accuracy",
        "validation_f1_macro",
        "total_seconds",
        "parameter_count",
    ]:
        denominador = normalizado[column].max() - normalizado[column].min()
        if denominador > 0:
            normalizado[column] = (normalizado[column] - normalizado[column].min()) / denominador
    normalizado["total_seconds"] = 1 - normalizado["total_seconds"]
    normalizado["parameter_count"] = 1 - normalizado["parameter_count"]
    normalizado.index = ordenado["model_id"]
    normalizado = normalizado.rename(
        columns={
            "validation_loss": "CE score",
            "validation_accuracy": "Accuracy score",
            "validation_f1_macro": "F1 score",
            "total_seconds": "Time score",
            "parameter_count": "Size score",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    imagen = axes[0].imshow(normalizado, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_title("Normalized score by criterion", fontsize=14)
    axes[0].set_xticks(np.arange(normalizado.shape[1]))
    axes[0].set_xticklabels(normalizado.columns, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(normalizado.shape[0]))
    axes[0].set_yticklabels(normalizado.index)
    fig.colorbar(imagen, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].barh(ordenado["model_id"], ordenado["validation_loss"], color="#335c67")
    axes[1].set_title("Ranking by validation cross-entropy", fontsize=14)
    axes[1].set_xlabel("Cross-entropy")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes


def graficar_comparacion_modelos_finales(
    resumen: pd.DataFrame,
    *,
    title: str = "Final model comparison on test",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica final test metrics para M0, M1, M2 y M3."""

    ordenado = resumen.copy()
    x = np.arange(len(ordenado))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    especificaciones = [
        ("test_loss", "Cross-entropy", "lower is better", "#7f4f24"),
        ("test_accuracy", "Accuracy", "higher is better", "#335c67"),
        ("test_f1_macro", "F1 macro", "higher is better", "#496f5d"),
    ]

    for ax, (column, label, subtitle, color) in zip(axes, especificaciones):
        barras = ax.bar(x, ordenado[column], color=color)
        ax.set_title(f"{label} ({subtitle})", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(ordenado["model_id"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        for bar in barras:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes


def graficar_robustness_noise(
    robustness: pd.DataFrame,
    *,
    title: str = "Robustness under Gaussian noise",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica performance del model para distintos niveles de input noise."""

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharex=True)
    especificaciones = [
        ("loss", "Cross-entropy", "lower is better"),
        ("accuracy", "Accuracy", "higher is better"),
        ("f1_macro", "F1 macro", "higher is better"),
    ]

    for ax, (metric, label, subtitle) in zip(axes, especificaciones):
        for model_id, group in robustness.groupby("model_id"):
            ordenado = group.sort_values("noise_std")
            ax.plot(
                ordenado["noise_std"],
                ordenado[metric],
                marker="o",
                linewidth=2,
                label=model_id,
            )
        ax.set_title(f"{label} ({subtitle})", fontsize=14)
        ax.set_xlabel("Noise standard deviation")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[-1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes
