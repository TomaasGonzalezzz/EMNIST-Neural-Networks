"""Helpers de training y evaluacion PyTorch para el punto 4 del TP3."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from .metrics import (
    accuracy_desde_confusion_matrix,
    f1_macro_desde_confusion_matrix,
)
from .training import iterar_chunks_indices


def obtener_device_torch() -> torch.device:
    """Devuelve el mejor device PyTorch disponible."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_seed_torch(random_state: int) -> None:
    """Fija seeds para reproducibilidad en PyTorch."""

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def preparar_batch_torch(
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    aplanar: bool = True,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selecciona, normaliza y mueve un batch al device PyTorch."""

    X_batch = np.asarray(X_fuente[indices], dtype=np.float32)
    y_batch = np.asarray(y_fuente[indices], dtype=np.int64)

    X_tensor = torch.from_numpy(X_batch).to(device=device, dtype=dtype)
    X_tensor = X_tensor / 255.0
    if noise_std > 0:
        X_tensor = torch.clamp(
            X_tensor + torch.randn_like(X_tensor) * noise_std,
            min=0.0,
            max=1.0,
        )
    if aplanar:
        X_tensor = X_tensor.view(X_tensor.shape[0], -1)

    y_tensor = torch.from_numpy(y_batch).to(device=device)
    return X_tensor, y_tensor


def learning_rate_para_epoch(
    epoch: int,
    *,
    initial_lr: float,
    schedule: str,
    total_epochs: int,
    final_lr: float | None = None,
    decay_rate: float | None = None,
    min_lr: float | None = None,
) -> float:
    """Calcula el learning rate para una epoch numerada desde 1."""

    schedule = schedule.lower()
    if schedule == "constant":
        return initial_lr
    if schedule == "linear":
        lr_objetivo = initial_lr if final_lr is None else final_lr
        if total_epochs <= 1:
            return lr_objetivo
        progress = min((epoch - 1) / (total_epochs - 1), 1.0)
        return initial_lr + progress * (lr_objetivo - initial_lr)
    if schedule == "exponential":
        tasa = 0.95 if decay_rate is None else decay_rate
        piso = 0.0 if min_lr is None else min_lr
        return max(initial_lr * (tasa ** (epoch - 1)), piso)
    if schedule == "cosine":
        lr_objetivo = 0.0 if final_lr is None else final_lr
        progress = min((epoch - 1) / max(total_epochs - 1, 1), 1.0)
        coseno = 0.5 * (1.0 + np.cos(np.pi * progress))
        return lr_objetivo + (initial_lr - lr_objetivo) * coseno
    raise ValueError(f"Schedule desconocido: {schedule}")


def fijar_learning_rate_optimizer(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Actualiza el mismo learning rate en todos los parameter groups."""

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def armar_optimizer(
    model: nn.Module,
    *,
    nombre_optimizer: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Crea el optimizer PyTorch pedido."""

    nombre_optimizer = nombre_optimizer.lower()
    if nombre_optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if nombre_optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if nombre_optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name debe ser 'adam', 'adamw' o 'sgd'.")


def evaluar_modelo_torch(
    model: nn.Module,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 32768,
    n_classes: int = 47,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    noise_std: float = 0.0,
    random_state: int = 42,
) -> dict[str, float | int | np.ndarray]:
    """Evalua un model PyTorch con loss, accuracy, F1 macro y confusion matrix."""

    if device is None:
        device = next(model.parameters()).device
    indices = np.asarray(indices)
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    suma_loss = 0.0
    n_muestras = len(indices)

    estaba_en_training = model.training
    model.eval()
    if noise_std > 0:
        fijar_seed_torch(random_state)

    with torch.no_grad():
        for indices_chunk in iterar_chunks_indices(indices, tamano_chunk=batch_size):
            X_batch, y_batch = preparar_batch_torch(
                X_fuente,
                y_fuente,
                indices_chunk,
                device=device,
                dtype=dtype,
                noise_std=noise_std,
            )
            logits = model(X_batch)
            loss = nn.functional.cross_entropy(logits, y_batch, reduction="sum")
            predicciones = torch.argmax(logits, dim=1)

            suma_loss += float(loss.detach().cpu())
            y_np = y_batch.detach().cpu().numpy()
            pred_np = predicciones.detach().cpu().numpy()
            np.add.at(matriz, (y_np, pred_np), 1)

    if estaba_en_training:
        model.train()

    return {
        "loss": suma_loss / n_muestras,
        "accuracy": accuracy_desde_confusion_matrix(matriz),
        "f1_macro": f1_macro_desde_confusion_matrix(matriz),
        "confusion_matrix": matriz,
        "n_samples": n_muestras,
    }


def copiar_state_dict_a_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Devuelve una copia detached en CPU del state_dict."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def entrenar_modelo_torch(
    model: nn.Module,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices_train: np.ndarray,
    indices_validation: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    nombre_optimizer: str = "adam",
    schedule: str = "constant",
    final_lr: float | None = None,
    decay_rate: float | None = None,
    min_lr: float | None = None,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    momentum: float = 0.9,
    early_stopping: bool = False,
    patience: int = 8,
    min_delta: float = 5e-4,
    restore_best: bool = True,
    patience_suba_validation: int | None = 5,
    delta_suba_validation: float = 0.0,
    indices_eval_train: np.ndarray | None = None,
    indices_eval_validation: np.ndarray | None = None,
    batch_size_eval: int = 32768,
    n_classes: int = 47,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    random_state: int = 42,
    verbose: bool = True,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Entrena un model PyTorch guardando diagnosticos de validation."""

    fijar_seed_torch(random_state)
    if device is None:
        device = obtener_device_torch()
    model.to(device)

    indices_train = np.asarray(indices_train)
    indices_validation = np.asarray(indices_validation)
    indices_eval_train = indices_train if indices_eval_train is None else np.asarray(indices_eval_train)
    indices_eval_validation = (
        indices_validation
        if indices_eval_validation is None
        else np.asarray(indices_eval_validation)
    )

    optimizer = armar_optimizer(
        model,
        nombre_optimizer=nombre_optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    funcion_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    rng = np.random.default_rng(random_state)

    historial: dict[str, list[float]] = {
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_f1_macro": [],
        "validation_f1_macro": [],
        "elapsed_seconds": [],
        "mean_batch_loss": [],
    }

    mejor_validation_loss = np.inf
    mejor_estado: dict[str, torch.Tensor] | None = None
    epochs_sin_mejora = 0
    subas_validation_consecutivas = 0
    validation_loss_anterior: float | None = None
    motivo_stop: str | None = None

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        lr_actual = learning_rate_para_epoch(
            epoch,
            initial_lr=learning_rate,
            schedule=schedule,
            total_epochs=epochs,
            final_lr=final_lr,
            decay_rate=decay_rate,
            min_lr=min_lr,
        )
        fijar_learning_rate_optimizer(optimizer, lr_actual)

        indices_mezclados = indices_train.copy()
        rng.shuffle(indices_mezclados)

        model.train()
        losses_batch: list[float] = []
        for batch_indices in iterar_chunks_indices(indices_mezclados, tamano_chunk=batch_size):
            X_batch, y_batch = preparar_batch_torch(
                X_fuente,
                y_fuente,
                batch_indices,
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = funcion_loss(logits, y_batch)
            loss.backward()
            optimizer.step()
            losses_batch.append(float(loss.detach().cpu()))

        metrics_train = evaluar_modelo_torch(
            model,
            X_fuente,
            y_fuente,
            indices_eval_train,
            batch_size=batch_size_eval,
            n_classes=n_classes,
            device=device,
            dtype=dtype,
        )
        metrics_validation = evaluar_modelo_torch(
            model,
            X_fuente,
            y_fuente,
            indices_eval_validation,
            batch_size=batch_size_eval,
            n_classes=n_classes,
            device=device,
            dtype=dtype,
        )
        tiempo = time.perf_counter() - start_time
        validation_loss = float(metrics_validation["loss"])

        fila = {
            "epoch": float(epoch),
            "learning_rate": float(lr_actual),
            "train_loss": float(metrics_train["loss"]),
            "validation_loss": validation_loss,
            "train_accuracy": float(metrics_train["accuracy"]),
            "validation_accuracy": float(metrics_validation["accuracy"]),
            "train_f1_macro": float(metrics_train["f1_macro"]),
            "validation_f1_macro": float(metrics_validation["f1_macro"]),
            "elapsed_seconds": tiempo,
            "mean_batch_loss": float(np.mean(losses_batch)),
        }
        for key, value in fila.items():
            historial[key].append(value)

        mejoro = validation_loss < mejor_validation_loss - min_delta
        if mejoro:
            mejor_validation_loss = validation_loss
            mejor_estado = copiar_state_dict_a_cpu(model)
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if validation_loss_anterior is not None:
            subio = validation_loss > validation_loss_anterior + delta_suba_validation
            if subio:
                subas_validation_consecutivas += 1
            else:
                subas_validation_consecutivas = 0
        validation_loss_anterior = validation_loss

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"lr={lr_actual:.6f} | "
                f"train CE={fila['train_loss']:.4f} | "
                f"val CE={fila['validation_loss']:.4f} | "
                f"train acc={fila['train_accuracy']:.4f} | "
                f"val acc={fila['validation_accuracy']:.4f} | "
                f"{tiempo:.1f}s"
            )

        if on_epoch_end is not None:
            on_epoch_end(fila)

        if (
            patience_suba_validation is not None
            and subas_validation_consecutivas >= patience_suba_validation
        ):
            motivo_stop = (
                "validation_loss subio durante "
                f"{patience_suba_validation} epochs consecutivas"
            )
            if verbose:
                print(f"Early stopping: {motivo_stop}.")
            break

        if early_stopping and epochs_sin_mejora >= patience:
            motivo_stop = (
                "validation_loss no mejoro durante "
                f"{patience} epochs"
            )
            if verbose:
                print(f"Early stopping: {motivo_stop}.")
            break

    if restore_best and mejor_estado is not None:
        model.load_state_dict(mejor_estado)
        historial["restored_best_epoch"] = [
            historial["epoch"][int(np.argmin(historial["validation_loss"]))]
        ]
    if motivo_stop is not None:
        historial["stop_reason"] = [motivo_stop]

    return historial
