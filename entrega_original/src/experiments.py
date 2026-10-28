"""Utilidades de grilla para elegir el modelo NumPy avanzado."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .models import ClasificadorMLP
from .training import (
    evaluar_clasificador,
    schedule_decay_exponencial,
    schedule_decay_lineal,
    entrenar_mini_batch,
)


@dataclass(frozen=True)
class ConfigExperimento:
    """Configuracion de un experimento de modelo avanzado."""

    model_id: str
    description: str
    architecture: tuple[int, ...]
    optimizer: str
    batch_size: int
    epochs: int
    learning_rate: float
    schedule: str
    final_lr: float | None = None
    decay_rate: float | None = None
    min_lr: float | None = None
    l2_lambda: float = 0.0
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-3
    random_state: int = 42


def grilla_experimentos_por_defecto() -> list[ConfigExperimento]:
    """Devuelve la grilla de modelos usada en el notebook."""

    input_dim = 28 * 28
    output_dim = 47
    epochs = 60
    return [
        ConfigExperimento(
            "MA",
            "Mini-batch SGD base con learning rate constante.",
            (input_dim, 128, 64, output_dim),
            "sgd",
            4096,
            epochs,
            0.15,
            "constant",
        ),
        ConfigExperimento(
            "MB",
            "Mini-batch SGD base con schedule lineal.",
            (input_dim, 128, 64, output_dim),
            "sgd",
            4096,
            epochs,
            0.18,
            "linear",
            final_lr=0.02,
        ),
        ConfigExperimento(
            "MC",
            "Mini-batch SGD base con schedule exponencial.",
            (input_dim, 128, 64, output_dim),
            "sgd",
            4096,
            epochs,
            0.18,
            "exponential",
            decay_rate=0.94,
            min_lr=0.02,
        ),
        ConfigExperimento(
            "MD",
            "Mini-batch SGD con batches mas chicos.",
            (input_dim, 128, 64, output_dim),
            "sgd",
            2048,
            epochs,
            0.12,
            "linear",
            final_lr=0.015,
        ),
        ConfigExperimento(
            "ME",
            "Adam sobre la arquitectura base.",
            (input_dim, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.001,
            "constant",
        ),
        ConfigExperimento(
            "MF",
            "Adam base con schedule lineal.",
            (input_dim, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.0012,
            "linear",
            final_lr=0.00010,
        ),
        ConfigExperimento(
            "MG",
            "Adam base con schedule exponencial.",
            (input_dim, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.0012,
            "exponential",
            decay_rate=0.94,
            min_lr=0.00010,
        ),
        ConfigExperimento(
            "MH",
            "Adam base con L2 y early stopping.",
            (input_dim, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.001,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MI",
            "Adam + L2 con arquitectura 256-128 y schedule lineal.",
            (input_dim, 256, 128, output_dim),
            "adam",
            4096,
            epochs,
            0.0011,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MJ",
            "Adam + L2 con arquitectura 256-128 y schedule exponencial.",
            (input_dim, 256, 128, output_dim),
            "adam",
            4096,
            epochs,
            0.0011,
            "exponential",
            decay_rate=0.94,
            min_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MK",
            "Adam + L2 con arquitectura 256-128-64.",
            (input_dim, 256, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.0012,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "ML",
            "Adam + L2 con arquitectura 384-192-96.",
            (input_dim, 384, 192, 96, output_dim),
            "adam",
            4096,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MM",
            "Adam + L2 con arquitectura 512-256-128.",
            (input_dim, 512, 256, 128, output_dim),
            "adam",
            4096,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MN",
            "Adam + L2 con arquitectura 512-256-128-64.",
            (input_dim, 512, 256, 128, 64, output_dim),
            "adam",
            4096,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MO",
            "Adam + L2 fuerte con arquitectura 512-256-128.",
            (input_dim, 512, 256, 128, output_dim),
            "adam",
            4096,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            l2_lambda=5e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
        ConfigExperimento(
            "MP",
            "Adam + L2 con arquitectura 768-384-192.",
            (input_dim, 768, 384, 192, output_dim),
            "adam",
            4096,
            epochs,
            0.0008,
            "linear",
            final_lr=0.00008,
            l2_lambda=1e-5,
            early_stopping=True,
            patience=10,
            min_delta=5e-4,
        ),
    ]


def rutas_artifacts_experimento(
    dir_artifacts,
    model_id,
    *,
    nombre_grilla="grid",
):
    """Devuelve paths usados por un experimento."""

    base = Path(dir_artifacts) / nombre_grilla
    return {
        "base": base,
        "history": base / f"{model_id}_history.npz",
        "parameters": base / f"{model_id}_parameters.npz",
        "metrics": base / f"{model_id}_metrics.npz",
        "config": base / f"{model_id}_config.csv",
    }


def armar_learning_rate_schedule(config):
    """Arma el callable de schedule que espera ``entrenar_mini_batch``."""

    if config.schedule == "constant":
        return None
    if config.schedule == "linear":
        final_lr = config.final_lr if config.final_lr is not None else config.learning_rate
        return lambda epoch: schedule_decay_lineal(
            epoch,
            initial_lr=config.learning_rate,
            final_lr=final_lr,
            total_epochs=config.epochs,
        )
    if config.schedule == "exponential":
        decay_rate = config.decay_rate if config.decay_rate is not None else 0.9
        min_lr = config.min_lr if config.min_lr is not None else 0.0
        return lambda epoch: schedule_decay_exponencial(
            epoch,
            initial_lr=config.learning_rate,
            decay_rate=decay_rate,
            min_lr=min_lr,
        )
    raise ValueError(f"Schedule desconocido: {config.schedule}")


def cargar_historial(path):
    """Carga un history npz como diccionario de listas."""

    data = np.load(path)
    return {key: data[key].tolist() for key in data.files}


def guardar_historial(path, historial):
    """Guarda un diccionario de history de training."""

    serializable = {key: np.asarray(value) for key, value in historial.items()}
    np.savez_compressed(path, **serializable)


def guardar_parametros(path, model):
    """Guarda parametros del modelo."""

    np.savez_compressed(path, **model.parametros)


def cargar_modelo_desde_parametros(
    path,
    architecture,
    *,
    dtype=np.float32,
):
    """Crea un modelo y carga parametros guardados."""

    model = ClasificadorMLP(list(architecture), random_state=42, dtype=dtype)
    data = np.load(path)
    model.fijar_parametros({key: data[key] for key in data.files})
    return model


def guardar_metrics(
    path,
    *,
    metrics_train,
    metrics_validation,
):
    """Guarda metrics finales de train/validation."""

    np.savez_compressed(
        path,
        train_loss=np.asarray(metrics_train["loss"]),
        train_accuracy=np.asarray(metrics_train["accuracy"]),
        train_f1_macro=np.asarray(metrics_train["f1_macro"]),
        train_n_samples=np.asarray(metrics_train["n_samples"]),
        train_confusion_matrix=metrics_train["confusion_matrix"],
        validation_loss=np.asarray(metrics_validation["loss"]),
        validation_accuracy=np.asarray(metrics_validation["accuracy"]),
        validation_f1_macro=np.asarray(metrics_validation["f1_macro"]),
        validation_n_samples=np.asarray(metrics_validation["n_samples"]),
        validation_confusion_matrix=metrics_validation["confusion_matrix"],
    )


def cargar_metrics(path):
    """Carga arrays de metrics guardadas."""

    data = np.load(path)
    return {key: data[key] for key in data.files}


def config_a_fila(config):
    """Aplana una config para armar tablas."""

    fila = asdict(config)
    fila["architecture"] = "-".join(str(size) for size in config.architecture)
    return fila


def resumir_experimento(
    config,
    historial,
    metrics,
):
    """Arma una fila de summary a partir de history y metrics."""

    fila = config_a_fila(config)
    losses_validation = np.asarray(historial["validation_loss"], dtype=float)
    idx_mejor_epoch = int(np.argmin(losses_validation))
    fila.update(
        {
            "completed_epochs": len(historial["epoch"]),
            "best_epoch": int(historial["epoch"][idx_mejor_epoch]),
            "best_history_validation_loss": float(losses_validation[idx_mejor_epoch]),
            "train_loss": float(metrics["train_loss"]),
            "train_accuracy": float(metrics["train_accuracy"]),
            "train_f1_macro": float(metrics["train_f1_macro"]),
            "validation_loss": float(metrics["validation_loss"]),
            "validation_accuracy": float(metrics["validation_accuracy"]),
            "validation_f1_macro": float(metrics["validation_f1_macro"]),
            "total_seconds": float(np.sum(historial["elapsed_seconds"])),
            "mean_epoch_seconds": float(np.mean(historial["elapsed_seconds"])),
            "parameter_count": int(
                sum(
                    fan_in * fan_out + fan_out
                    for fan_in, fan_out in zip(
                        config.architecture[:-1],
                        config.architecture[1:],
                    )
                )
            ),
        }
    )
    return fila


def correr_experimento(
    config,
    X_fuente,
    y_fuente,
    indices_train,
    indices_validation,
    *,
    dir_artifacts,
    nombre_grilla="grid",
    force=False,
    indices_curva_train=None,
    indices_curva_validation=None,
    tamano_chunk_eval=32768,
    n_classes=47,
    dtype=np.float32,
    verbose=True,
):
    """Entrena o carga un experimento y devuelve su fila de summary."""

    rutas = rutas_artifacts_experimento(
        dir_artifacts,
        config.model_id,
        nombre_grilla=nombre_grilla,
    )
    rutas["base"].mkdir(parents=True, exist_ok=True)

    existe = (
        rutas["history"].exists()
        and rutas["parameters"].exists()
        and rutas["metrics"].exists()
        and rutas["config"].exists()
    )

    if existe and not force:
        historial = cargar_historial(rutas["history"])
        metrics = cargar_metrics(rutas["metrics"])
        return resumir_experimento(config, historial, metrics)

    if verbose:
        print(f"\n=== {config.model_id}: {config.description} ===")

    model = ClasificadorMLP(
        list(config.architecture),
        random_state=config.random_state,
        dtype=dtype,
    )
    historial = entrenar_mini_batch(
        model,
        X_fuente,
        y_fuente,
        indices_train,
        indices_validation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        lr_schedule=armar_learning_rate_schedule(config),
        l2_lambda=config.l2_lambda,
        early_stopping=config.early_stopping,
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best=True,
        indices_eval_train=indices_curva_train,
        indices_eval_validation=indices_curva_validation,
        tamano_chunk_eval=tamano_chunk_eval,
        n_classes=n_classes,
        dtype=dtype,
        normalizar=True,
        aplanar=True,
        random_state=config.random_state,
        verbose=verbose,
    )

    metrics_train = evaluar_clasificador(
        model,
        X_fuente,
        y_fuente,
        indices_train,
        tamano_chunk=tamano_chunk_eval,
        n_classes=n_classes,
        dtype=dtype,
        normalizar=True,
        aplanar=True,
    )
    metrics_validation = evaluar_clasificador(
        model,
        X_fuente,
        y_fuente,
        indices_validation,
        tamano_chunk=tamano_chunk_eval,
        n_classes=n_classes,
        dtype=dtype,
        normalizar=True,
        aplanar=True,
    )

    guardar_historial(rutas["history"], historial)
    guardar_parametros(rutas["parameters"], model)
    guardar_metrics(
        rutas["metrics"],
        metrics_train=metrics_train,
        metrics_validation=metrics_validation,
    )
    pd.DataFrame([config_a_fila(config)]).to_csv(rutas["config"], index=False)
    metrics = cargar_metrics(rutas["metrics"])
    return resumir_experimento(config, historial, metrics)


def correr_grilla_experimentos(
    configuraciones,
    X_fuente,
    y_fuente,
    indices_train,
    indices_validation,
    *,
    dir_artifacts,
    nombre_grilla="grid",
    force=False,
    tamano_curva_train=None,
    tamano_curva_validation=None,
    tamano_chunk_eval=32768,
    n_classes=47,
    dtype=np.float32,
    verbose=True,
):
    """Entrena/carga todos los experimentos y guarda el summary de la grilla."""

    filas = []
    rng = np.random.default_rng(123)
    indices_curva_train = None
    indices_curva_validation = None
    if tamano_curva_train is not None and tamano_curva_train < len(indices_train):
        indices_curva_train = rng.choice(
            np.asarray(indices_train),
            size=tamano_curva_train,
            replace=False,
        )
    if tamano_curva_validation is not None and tamano_curva_validation < len(indices_validation):
        indices_curva_validation = rng.choice(
            np.asarray(indices_validation),
            size=tamano_curva_validation,
            replace=False,
        )

    for config in configuraciones:
        fila = correr_experimento(
            config,
            X_fuente,
            y_fuente,
            indices_train,
            indices_validation,
            dir_artifacts=dir_artifacts,
            nombre_grilla=nombre_grilla,
            force=force,
            indices_curva_train=indices_curva_train,
            indices_curva_validation=indices_curva_validation,
            tamano_chunk_eval=tamano_chunk_eval,
            n_classes=n_classes,
            dtype=dtype,
            verbose=verbose,
        )
        filas.append(fila)

    resumen = pd.DataFrame(filas)
    ruta_resumen = Path(dir_artifacts) / f"{nombre_grilla}_experiment_summary.csv"
    resumen.to_csv(ruta_resumen, index=False)
    return resumen


def seleccionar_mejor_experimento(resumen):
    """Selecciona el mejor experimento por validation cross-entropy."""

    return resumen.sort_values(
        ["validation_loss", "validation_f1_macro", "validation_accuracy"],
        ascending=[True, False, False],
    ).iloc[0]
