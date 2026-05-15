"""Metricas NumPy para clasificacion multiclass."""

from __future__ import annotations

import numpy as np


def accuracy_desde_confusion_matrix(matriz: np.ndarray) -> float:
    """Calcula accuracy a partir de una confusion matrix."""

    matriz = np.asarray(matriz)
    return float(np.trace(matriz) / np.sum(matriz))


def f1_macro_desde_confusion_matrix(
    matriz: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Calcula F1 macro a partir de una confusion matrix."""

    matriz = np.asarray(matriz)
    verdaderos_positivos = np.diag(matriz)
    precision = verdaderos_positivos / np.maximum(matriz.sum(axis=0), eps)
    recall = verdaderos_positivos / np.maximum(matriz.sum(axis=1), eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    return float(np.mean(f1))
