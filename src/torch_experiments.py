"""Utilidades de grilla para models PyTorch del punto 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .torch_models import MLPTorch, contar_parametros
from .torch_training import (
    evaluar_modelo_torch,
    obtener_device_torch,
    entrenar_modelo_torch,
)


@dataclass(frozen=True)
class ConfigExperimentoTorch:
    """Configuracion de un experimento PyTorch."""

    model_id: str
    description: str
    hidden_sizes: tuple[int, ...]
    activation: str
    dropout: float
    batch_norm: bool
    optimizer: str
    batch_size: int
    epochs: int
    learning_rate: float
    schedule: str
    final_lr: float | None = None
    decay_rate: float | None = None
    min_lr: float | None = None
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 5e-4
    patience_suba_validation: int | None = 5
    delta_suba_validation: float = 0.0
    random_state: int = 42


def config_experimento_m2() -> ConfigExperimentoTorch:
    """Devuelve M2: misma arquitectura e hiperparametros principales que M1."""

    return ConfigExperimentoTorch(
        model_id="M2",
        description="PyTorch equivalente a M1: 512-256-128, ReLU, Adam y L2.",
        hidden_sizes=(512, 256, 128),
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        optimizer="adam",
        batch_size=4096,
        epochs=100,
        learning_rate=0.001,
        schedule="linear",
        final_lr=0.0001,
        weight_decay=5e-5,
        early_stopping=True,
        patience=10,
        min_delta=5e-4,
        patience_suba_validation=5,
    )


def grilla_experimentos_torch_por_defecto() -> list[ConfigExperimentoTorch]:
    """Devuelve el pool amplio de PyTorch usado para elegir M3."""

    epochs = 60
    batch_size = 4096
    return [
        ConfigExperimentoTorch(
            "P4A",
            "ReLU sin BatchNorm ni Dropout; control directo de M1 en PyTorch.",
            (512, 256, 128),
            "relu",
            0.0,
            False,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4B",
            "ReLU con BatchNorm para estabilizar capas profundas.",
            (512, 256, 128),
            "relu",
            0.0,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4C",
            "ReLU con Dropout bajo para reducir overfitting.",
            (512, 256, 128),
            "relu",
            0.10,
            False,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4D",
            "ReLU con BatchNorm y Dropout bajo.",
            (512, 256, 128),
            "relu",
            0.10,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4E",
            "ReLU con BatchNorm y Dropout medio.",
            (512, 256, 128),
            "relu",
            0.20,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4F",
            "LeakyReLU con BatchNorm y Dropout bajo.",
            (512, 256, 128),
            "leaky_relu",
            0.10,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4G",
            "SiLU con BatchNorm y Dropout bajo.",
            (512, 256, 128),
            "silu",
            0.10,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4H",
            "GELU con BatchNorm y Dropout bajo.",
            (512, 256, 128),
            "gelu",
            0.10,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4I",
            "SiLU con BatchNorm y Dropout medio.",
            (512, 256, 128),
            "silu",
            0.20,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4J",
            "GELU con BatchNorm y Dropout medio.",
            (512, 256, 128),
            "gelu",
            0.20,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
        ),
        ConfigExperimentoTorch(
            "P4K",
            "GELU con BatchNorm, Dropout bajo y label smoothing.",
            (512, 256, 128),
            "gelu",
            0.10,
            True,
            "adam",
            batch_size,
            epochs,
            0.0010,
            "linear",
            final_lr=0.00010,
            weight_decay=5e-5,
            label_smoothing=0.05,
        ),
        ConfigExperimentoTorch(
            "P4L",
            "SiLU con AdamW, BatchNorm y Dropout medio.",
            (512, 256, 128),
            "silu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0010,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4M",
            "GELU con AdamW, BatchNorm y Dropout medio.",
            (512, 256, 128),
            "gelu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0010,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4N",
            "LeakyReLU mas ancho: 768-384-192 con BatchNorm y Dropout.",
            (768, 384, 192),
            "leaky_relu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0008,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4O",
            "SiLU mas ancho: 768-384-192 con BatchNorm y Dropout.",
            (768, 384, 192),
            "silu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0008,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4P",
            "GELU mas ancho: 768-384-192 con BatchNorm y Dropout.",
            (768, 384, 192),
            "gelu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0008,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4Q",
            "GELU con cuello mas lento: 512-512-256.",
            (512, 512, 256),
            "gelu",
            0.15,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0008,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4R",
            "SiLU mas profundo: 512-256-128-64.",
            (512, 256, 128, 64),
            "silu",
            0.10,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0009,
            "cosine",
            final_lr=0.00005,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4S",
            "GELU ancho: 1024-512-256 con BatchNorm y Dropout.",
            (1024, 512, 256),
            "gelu",
            0.20,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0007,
            "cosine",
            final_lr=0.00004,
            weight_decay=1e-4,
        ),
        ConfigExperimentoTorch(
            "P4T",
            "GELU ancho y profundo: 1024-512-256-128.",
            (1024, 512, 256, 128),
            "gelu",
            0.20,
            True,
            "adamw",
            batch_size,
            epochs,
            0.0007,
            "cosine",
            final_lr=0.00004,
            weight_decay=1e-4,
        ),
    ]


def rutas_artifacts_experimento(
    dir_artifacts: str | Path,
    model_id: str,
    *,
    nombre_grilla: str = "torch_grid",
) -> dict[str, Path]:
    """Devuelve paths de artifacts para un experimento PyTorch."""

    base = Path(dir_artifacts) / nombre_grilla
    return {
        "base": base,
        "history": base / f"{model_id}_history.npz",
        "model": base / f"{model_id}_model.pt",
        "metrics": base / f"{model_id}_metrics.npz",
        "config": base / f"{model_id}_config.csv",
    }


def config_a_fila(config: ConfigExperimentoTorch) -> dict[str, object]:
    """Aplana una config PyTorch para armar tablas."""

    fila = asdict(config)
    fila["hidden_sizes"] = "-".join(str(size) for size in config.hidden_sizes)
    return fila


def guardar_historial(path: str | Path, historial: dict[str, list[float]]) -> None:
    """Guarda un diccionario de history de training."""

    serializable = {key: np.asarray(value) for key, value in historial.items()}
    np.savez_compressed(path, **serializable)


def cargar_historial(path: str | Path) -> dict[str, list[float]]:
    """Carga un history PyTorch guardado."""

    data = np.load(path, allow_pickle=True)
    return {key: data[key].tolist() for key in data.files}


def guardar_metrics(
    path: str | Path,
    *,
    metrics_train: dict[str, float | int | np.ndarray],
    metrics_validation: dict[str, float | int | np.ndarray],
) -> None:
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


def cargar_metrics(path: str | Path) -> dict[str, np.ndarray]:
    """Carga arrays de metrics guardadas."""

    data = np.load(path)
    return {key: data[key] for key in data.files}


def armar_modelo_desde_config(config: ConfigExperimentoTorch) -> MLPTorch:
    """Instancia un MLP PyTorch desde una config de experimento."""

    return MLPTorch(
        input_dim=28 * 28,
        hidden_sizes=list(config.hidden_sizes),
        output_dim=47,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    )


def guardar_modelo(path: str | Path, model: torch.nn.Module, config: ConfigExperimentoTorch) -> None:
    """Guarda el state de un model PyTorch junto con su config."""

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config_a_fila(config),
        },
        path,
    )


def cargar_modelo(
    path: str | Path,
    config: ConfigExperimentoTorch,
    *,
    device: torch.device | None = None,
) -> MLPTorch:
    """Carga un model PyTorch desde un state_dict guardado."""

    if device is None:
        device = obtener_device_torch()
    model = armar_modelo_desde_config(config)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def resumir_experimento(
    config: ConfigExperimentoTorch,
    historial: dict[str, list[float]],
    metrics: dict[str, np.ndarray],
) -> dict[str, object]:
    """Arma una fila de summary desde history y metrics PyTorch."""

    fila = config_a_fila(config)
    losses_validation = np.asarray(historial["validation_loss"], dtype=float)
    idx_mejor_epoch = int(np.argmin(losses_validation))
    cantidad_parametros = contar_parametros(armar_modelo_desde_config(config))
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
            "parameter_count": int(cantidad_parametros),
        }
    )
    return fila


def correr_experimento_torch(
    config: ConfigExperimentoTorch,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices_train: np.ndarray,
    indices_validation: np.ndarray,
    *,
    dir_artifacts: str | Path,
    nombre_grilla: str = "torch_grid",
    force: bool = False,
    indices_curva_train: np.ndarray | None = None,
    indices_curva_validation: np.ndarray | None = None,
    batch_size_eval: int = 32768,
    n_classes: int = 47,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Entrena o carga un experimento PyTorch y devuelve su fila de summary."""

    rutas = rutas_artifacts_experimento(
        dir_artifacts,
        config.model_id,
        nombre_grilla=nombre_grilla,
    )
    rutas["base"].mkdir(parents=True, exist_ok=True)

    existe = (
        rutas["history"].exists()
        and rutas["model"].exists()
        and rutas["metrics"].exists()
        and rutas["config"].exists()
    )
    if existe and not force:
        historial = cargar_historial(rutas["history"])
        metrics = cargar_metrics(rutas["metrics"])
        return resumir_experimento(config, historial, metrics)

    if device is None:
        device = obtener_device_torch()
    if verbose:
        print(f"\n=== {config.model_id}: {config.description} ===")

    model = armar_modelo_desde_config(config)
    historial = entrenar_modelo_torch(
        model,
        X_fuente,
        y_fuente,
        indices_train,
        indices_validation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        nombre_optimizer=config.optimizer,
        schedule=config.schedule,
        final_lr=config.final_lr,
        decay_rate=config.decay_rate,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        early_stopping=config.early_stopping,
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best=True,
        patience_suba_validation=config.patience_suba_validation,
        delta_suba_validation=config.delta_suba_validation,
        indices_eval_train=indices_curva_train,
        indices_eval_validation=indices_curva_validation,
        batch_size_eval=batch_size_eval,
        n_classes=n_classes,
        device=device,
        random_state=config.random_state,
        verbose=verbose,
    )

    metrics_train = evaluar_modelo_torch(
        model,
        X_fuente,
        y_fuente,
        indices_train,
        batch_size=batch_size_eval,
        n_classes=n_classes,
        device=device,
    )
    metrics_validation = evaluar_modelo_torch(
        model,
        X_fuente,
        y_fuente,
        indices_validation,
        batch_size=batch_size_eval,
        n_classes=n_classes,
        device=device,
    )

    guardar_historial(rutas["history"], historial)
    guardar_modelo(rutas["model"], model, config)
    guardar_metrics(
        rutas["metrics"],
        metrics_train=metrics_train,
        metrics_validation=metrics_validation,
    )
    pd.DataFrame([config_a_fila(config)]).to_csv(rutas["config"], index=False)

    metrics = cargar_metrics(rutas["metrics"])
    return resumir_experimento(config, historial, metrics)


def correr_grilla_experimentos_torch(
    configuraciones: list[ConfigExperimentoTorch],
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices_train: np.ndarray,
    indices_validation: np.ndarray,
    *,
    dir_artifacts: str | Path,
    nombre_grilla: str = "torch_grid",
    force: bool = False,
    tamano_curva_train: int | None = None,
    tamano_curva_validation: int | None = None,
    batch_size_eval: int = 32768,
    n_classes: int = 47,
    device: torch.device | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entrena/carga todos los experimentos PyTorch y guarda la tabla summary."""

    if device is None:
        device = obtener_device_torch()
    filas = []
    rng = np.random.default_rng(321)
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
        fila = correr_experimento_torch(
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
            batch_size_eval=batch_size_eval,
            n_classes=n_classes,
            device=device,
            verbose=verbose,
        )
        filas.append(fila)

    resumen = pd.DataFrame(filas)
    ruta_resumen = Path(dir_artifacts) / f"{nombre_grilla}_experiment_summary.csv"
    resumen.to_csv(ruta_resumen, index=False)
    return resumen


def seleccionar_mejor_experimento_torch(resumen: pd.DataFrame) -> pd.Series:
    """Selecciona el mejor experimento PyTorch por validation cross-entropy."""

    return resumen.sort_values(
        ["validation_loss", "validation_f1_macro", "validation_accuracy"],
        ascending=[True, False, False],
    ).iloc[0]


def config_por_model_id(
    configuraciones: list[ConfigExperimentoTorch],
    model_id: str,
) -> ConfigExperimentoTorch:
    """Devuelve una config de la lista usando model_id."""

    matches = [config for config in configuraciones if config.model_id == model_id]
    if not matches:
        raise ValueError(f"model_id desconocido: {model_id}")
    return matches[0]


def armar_config_m3(
    config_base: ConfigExperimentoTorch,
    *,
    epochs: int = 100,
) -> ConfigExperimentoTorch:
    """Promueve la mejor config del pool a la config final de M3."""

    return replace(
        config_base,
        model_id="M3",
        description=f"M3 final basado en {config_base.model_id}: {config_base.description}",
        epochs=epochs,
        patience_suba_validation=5,
    )
