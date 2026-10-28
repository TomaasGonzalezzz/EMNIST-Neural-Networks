"""Helpers de carga y preprocesamiento para EMNIST ByMerge."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class ParticionesDataset(NamedTuple):
    """Contenedor de indices para train/validation/test."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def raiz_proyecto() -> Path:
    """Devuelve la raiz del proyecto asumiendo que este modulo vive en ``src/``."""

    return Path(__file__).resolve().parents[1]


def cargar_arrays_emnist(
    dir_datos: str | Path = "Datasets",
    *,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray]:
    """Carga los arrays de imagenes y targets de EMNIST.

    ``mmap_mode="r"`` permite leer el array grande sin copiarlo completo a
    memoria. 
    """

    ruta_datos = Path(dir_datos)
    if not ruta_datos.is_absolute():
        ruta_datos = raiz_proyecto() / ruta_datos

    X_imagenes = np.load(ruta_datos / "X_images.npy", mmap_mode=mmap_mode)
    y_imagenes = np.load(ruta_datos / "y_images.npy", mmap_mode=mmap_mode)
    return X_imagenes, y_imagenes


def resumir_dataset(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Devuelve una tabla compacta con informacion general del dataset."""

    resumen = {
        "cantidad_observaciones": int(X.shape[0]),
        "alto": int(X.shape[1]) if X.ndim >= 2 else None,
        "ancho": int(X.shape[2]) if X.ndim >= 3 else None,
        "dtype_X": str(X.dtype),
        "min_pixel": int(np.min(X)),
        "max_pixel": int(np.max(X)),
        "cantidad_clases": int(np.unique(y).size),
        "min_clase": int(np.min(y)),
        "max_clase": int(np.max(y)),
        "dtype_y": str(y.dtype),
    }
    return pd.DataFrame([resumen])


def distribucion_clases(y: np.ndarray) -> pd.DataFrame:
    """Devuelve count y proporcion por class label."""

    labels, conteos = np.unique(y, return_counts=True)
    distribucion = pd.DataFrame(
        {
            "clase": labels.astype(int),
            "cantidad": conteos.astype(int),
            "proporcion": conteos / conteos.sum(),
        }
    )
    return distribucion


def indices_split_estratificado(
    y: np.ndarray,
    *,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> ParticionesDataset:
    """Crea indices estratificados de train/validation/test usando solo NumPy."""

    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + validation_size + test_size debe sumar 1.")

    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    partes_train: list[np.ndarray] = []
    partes_validation: list[np.ndarray] = []
    partes_test: list[np.ndarray] = []

    for label in np.unique(y):
        indices_clase = np.flatnonzero(y == label)
        rng.shuffle(indices_clase)

        n_clase = indices_clase.size
        n_train = int(np.floor(train_size * n_clase))
        n_validation = int(np.floor(validation_size * n_clase))

        partes_train.append(indices_clase[:n_train])
        partes_validation.append(indices_clase[n_train : n_train + n_validation])
        partes_test.append(indices_clase[n_train + n_validation :])

    idx_train = np.concatenate(partes_train)
    idx_validation = np.concatenate(partes_validation)
    idx_test = np.concatenate(partes_test)

    rng.shuffle(idx_train)
    rng.shuffle(idx_validation)
    rng.shuffle(idx_test)
    return ParticionesDataset(train=idx_train, validation=idx_validation, test=idx_test)


def tabla_tamanos_split(particiones: ParticionesDataset) -> pd.DataFrame:
    """Devuelve una tabla simple con tamanos y proporciones de cada split."""

    tamanos = {
        "train": len(particiones.train),
        "validation": len(particiones.validation),
        "test": len(particiones.test),
    }
    total = sum(tamanos.values())
    return pd.DataFrame(
        {
            "split": list(tamanos.keys()),
            "cantidad": list(tamanos.values()),
            "proporcion": [value / total for value in tamanos.values()],
        }
    )
