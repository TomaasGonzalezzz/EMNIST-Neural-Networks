"""Helpers de training para neural networks en NumPy."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .metrics import (
    accuracy_desde_confusion_matrix,
    f1_macro_desde_confusion_matrix,
)
from .models import ClasificadorMLP


@dataclass
class EstadoAdam:
    """Estimaciones de momentos usadas por Adam."""

    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]
    t: int = 0


def schedule_lineal(
    epoch: int,
    *,
    initial_lr: float,
    final_lr: float,
    total_epochs: int,
) -> float:
    """Learning rate schedule lineal con saturacion en ``final_lr``."""

    if total_epochs <= 1:
        return final_lr
    progress = min(epoch / (total_epochs - 1), 1.0)
    return initial_lr + progress * (final_lr - initial_lr)


def schedule_exponencial(
    epoch: int,
    *,
    initial_lr: float,
    decay_rate: float,
    min_lr: float = 0.0,
) -> float:
    """Learning rate schedule exponencial con piso opcional."""

    return max(initial_lr * (decay_rate**epoch), min_lr)


def schedule_decay_lineal(
    epoch: int,
    *,
    initial_lr: float,
    final_lr: float,
    total_epochs: int,
) -> float:
    """Schedule lineal usando epochs numeradas desde 1."""

    return schedule_lineal(
        epoch - 1,
        initial_lr=initial_lr,
        final_lr=final_lr,
        total_epochs=total_epochs,
    )


def schedule_decay_exponencial(
    epoch: int,
    *,
    initial_lr: float,
    decay_rate: float,
    min_lr: float = 0.0,
) -> float:
    """Schedule exponencial usando epochs numeradas desde 1."""

    return schedule_exponencial(
        epoch - 1,
        initial_lr=initial_lr,
        decay_rate=decay_rate,
        min_lr=min_lr,
    )


def penalizacion_l2(model: ClasificadorMLP, *, l2_lambda: float) -> float:
    """Calcula el termino de penalizacion L2 solo sobre matrices de pesos."""

    if l2_lambda <= 0:
        return 0.0
    penalizacion = 0.0
    for key, value in model.parametros.items():
        if key.startswith("W"):
            penalizacion += float(np.sum(value * value))
    return 0.5 * l2_lambda * penalizacion


def sumar_gradientes_l2_inplace(
    model: ClasificadorMLP,
    gradientes: dict[str, np.ndarray],
    *,
    l2_lambda: float,
) -> None:
    """Suma gradientes de L2 weight decay in place."""

    if l2_lambda <= 0:
        return
    for key in gradientes:
        if key.startswith("W"):
            gradientes[key] += l2_lambda * model.parametros[key]


def clip_gradients_por_norma_global(
    gradientes: dict[str, np.ndarray],
    *,
    max_norm: float | None,
    eps: float = 1e-12,
) -> float:
    """Aplica gradient clipping por norma global y devuelve la norma original."""

    norma_cuadrada = 0.0
    for gradiente in gradientes.values():
        norma_cuadrada += float(np.sum(gradiente * gradiente))

    norma_global = float(np.sqrt(norma_cuadrada))
    if max_norm is not None and norma_global > max_norm:
        scale = max_norm / (norma_global + eps)
        for key in gradientes:
            gradientes[key] *= scale
    return norma_global


def inicializar_estado_adam(model: ClasificadorMLP) -> EstadoAdam:
    """Inicializa buffers de momentos en cero para Adam."""

    return EstadoAdam(
        m=model.gradientes_en_cero(),
        v=model.gradientes_en_cero(),
        t=0,
    )


def aplicar_update_adam(
    model: ClasificadorMLP,
    gradientes: dict[str, np.ndarray],
    estado: EstadoAdam,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Aplica un step del optimizer Adam."""

    estado.t += 1
    for key in model.parametros:
        estado.m[key] = beta1 * estado.m[key] + (1.0 - beta1) * gradientes[key]
        estado.v[key] = beta2 * estado.v[key] + (1.0 - beta2) * (gradientes[key] ** 2)
        m_hat = estado.m[key] / (1.0 - beta1**estado.t)
        v_hat = estado.v[key] / (1.0 - beta2**estado.t)
        model.parametros[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def iterar_chunks_indices(indices: np.ndarray, *, tamano_chunk: int) -> list[np.ndarray]:
    """Divide un array de indices en chunks contiguos."""

    return [
        indices[start : start + tamano_chunk]
        for start in range(0, len(indices), tamano_chunk)
    ]


def preparar_chunk_features(
    X_fuente: np.ndarray,
    indices: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    normalizar: bool = True,
    aplanar: bool = True,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Selecciona un chunk de imagenes y lo prepara como input del model."""

    X_chunk = np.asarray(X_fuente[indices], dtype=dtype)
    if normalizar:
        X_chunk /= np.array(255.0, dtype=dtype)
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_std, size=X_chunk.shape).astype(dtype)
        X_chunk = np.clip(X_chunk + noise, 0.0, 1.0)
    if aplanar:
        X_chunk = X_chunk.reshape(X_chunk.shape[0], -1)
    return X_chunk


def calcular_gradientes_full_batch(
    model: ClasificadorMLP,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices: np.ndarray,
    *,
    tamano_chunk: int = 8192,
    dtype: np.dtype = np.float32,
    normalizar: bool = True,
    aplanar: bool = True,
    eps: float = 1e-12,
) -> tuple[dict[str, np.ndarray], float]:
    """Calcula gradientes full-dataset usando chunks cuidados de memoria.

    Esto sigue siendo gradient descent estandar: los chunks solo son una
    estrategia interna de memoria. El model se actualiza una vez despues de
    acumular todos los gradientes.
    """

    indices = np.asarray(indices)
    n_muestras = len(indices)
    gradientes = model.gradientes_en_cero()
    suma_loss = 0.0

    for indices_chunk in iterar_chunks_indices(indices, tamano_chunk=tamano_chunk):
        X_chunk = preparar_chunk_features(
            X_fuente,
            indices_chunk,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )
        y_chunk = np.asarray(y_fuente[indices_chunk], dtype=int)

        probabilidades, caches = model.forward(X_chunk)
        probabilidades_clase_real = probabilidades[np.arange(y_chunk.size), y_chunk]
        suma_loss += float(-np.sum(np.log(np.clip(probabilidades_clase_real, eps, 1.0))))

        gradientes_chunk = model.backward(
            probabilidades,
            y_chunk,
            caches,
            normalizador=n_muestras,
        )
        model.sumar_gradientes_inplace(gradientes, gradientes_chunk)

    return gradientes, suma_loss / n_muestras


def evaluar_clasificador(
    model: ClasificadorMLP,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices: np.ndarray,
    *,
    tamano_chunk: int = 16384,
    n_classes: int | None = None,
    dtype: np.dtype = np.float32,
    normalizar: bool = True,
    aplanar: bool = True,
    noise_std: float = 0.0,
    random_state: int = 42,
    eps: float = 1e-12,
) -> dict[str, float | int | np.ndarray]:
    """Evalua loss, accuracy, F1 macro y confusion matrix por chunks."""

    indices = np.asarray(indices)
    if n_classes is None:
        n_classes = int(np.max(y_fuente) + 1)

    matriz = np.zeros((n_classes, n_classes), dtype=int)
    suma_loss = 0.0
    n_muestras = len(indices)
    rng = np.random.default_rng(random_state) if noise_std > 0 else None

    for indices_chunk in iterar_chunks_indices(indices, tamano_chunk=tamano_chunk):
        X_chunk = preparar_chunk_features(
            X_fuente,
            indices_chunk,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
            noise_std=noise_std,
            rng=rng,
        )
        y_chunk = np.asarray(y_fuente[indices_chunk], dtype=int)

        probabilidades = model.predecir_probabilidades(X_chunk)
        predicciones = np.argmax(probabilidades, axis=1)
        probabilidades_clase_real = probabilidades[np.arange(y_chunk.size), y_chunk]
        suma_loss += float(-np.sum(np.log(np.clip(probabilidades_clase_real, eps, 1.0))))
        np.add.at(matriz, (y_chunk, predicciones), 1)

    loss = suma_loss / n_muestras
    return {
        "loss": loss,
        "accuracy": accuracy_desde_confusion_matrix(matriz),
        "f1_macro": f1_macro_desde_confusion_matrix(matriz),
        "confusion_matrix": matriz,
        "n_samples": n_muestras,
    }


def entrenar_full_batch_gradient_descent(
    model: ClasificadorMLP,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices_train: np.ndarray,
    indices_validation: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    tamano_chunk_gradiente: int = 8192,
    tamano_chunk_eval: int = 16384,
    n_classes: int | None = None,
    dtype: np.dtype = np.float32,
    normalizar: bool = True,
    aplanar: bool = True,
    verbose: bool = True,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Entrena un MLP con full-batch gradient descent."""

    historial: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_f1_macro": [],
        "validation_f1_macro": [],
        "gradient_loss": [],
        "elapsed_seconds": [],
    }

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        gradientes, gradient_loss = calcular_gradientes_full_batch(
            model,
            X_fuente,
            y_fuente,
            indices_train,
            tamano_chunk=tamano_chunk_gradiente,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )
        model.aplicar_gradientes(gradientes, learning_rate=learning_rate)

        metrics_train = evaluar_clasificador(
            model,
            X_fuente,
            y_fuente,
            indices_train,
            tamano_chunk=tamano_chunk_eval,
            n_classes=n_classes,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )
        metrics_validation = evaluar_clasificador(
            model,
            X_fuente,
            y_fuente,
            indices_validation,
            tamano_chunk=tamano_chunk_eval,
            n_classes=n_classes,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )
        tiempo = time.perf_counter() - start_time

        fila = {
            "epoch": float(epoch),
            "train_loss": float(metrics_train["loss"]),
            "validation_loss": float(metrics_validation["loss"]),
            "train_accuracy": float(metrics_train["accuracy"]),
            "validation_accuracy": float(metrics_validation["accuracy"]),
            "train_f1_macro": float(metrics_train["f1_macro"]),
            "validation_f1_macro": float(metrics_validation["f1_macro"]),
            "gradient_loss": float(gradient_loss),
            "elapsed_seconds": tiempo,
        }
        for key, value in fila.items():
            historial[key].append(value)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train CE={fila['train_loss']:.4f} | "
                f"val CE={fila['validation_loss']:.4f} | "
                f"train acc={fila['train_accuracy']:.4f} | "
                f"val acc={fila['validation_accuracy']:.4f} | "
                f"{tiempo:.1f}s"
            )

        if on_epoch_end is not None:
            on_epoch_end(fila)

    return historial


def entrenar_mini_batch(
    model: ClasificadorMLP,
    X_fuente: np.ndarray,
    y_fuente: np.ndarray,
    indices_train: np.ndarray,
    indices_validation: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str = "sgd",
    lr_schedule: Callable[[int], float] | None = None,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
    restore_best: bool = True,
    patience_suba_validation: int | None = None,
    delta_suba_validation: float = 0.0,
    norma_clip_gradiente: float | None = None,
    indices_eval_train: np.ndarray | None = None,
    indices_eval_validation: np.ndarray | None = None,
    tamano_chunk_eval: int = 32768,
    n_classes: int | None = None,
    dtype: np.dtype = np.float32,
    normalizar: bool = True,
    aplanar: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Entrena un MLP con mini-batches y Adam/L2/early stopping opcionales."""

    optimizer = optimizer.lower()
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer debe ser 'sgd' o 'adam'.")

    rng = np.random.default_rng(random_state)
    indices_train = np.asarray(indices_train)
    indices_validation = np.asarray(indices_validation)
    indices_eval_train = indices_train if indices_eval_train is None else np.asarray(indices_eval_train)
    indices_eval_validation = (
        indices_validation
        if indices_eval_validation is None
        else np.asarray(indices_eval_validation)
    )
    estado_adam = inicializar_estado_adam(model) if optimizer == "adam" else None

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
        "train_objective": [],
        "validation_objective": [],
        "mean_gradient_norm": [],
    }

    mejor_validation_loss = np.inf
    mejores_parametros: dict[str, np.ndarray] | None = None
    epochs_sin_mejora = 0
    subas_validation_consecutivas = 0
    validation_loss_anterior: float | None = None
    motivo_stop: str | None = None

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        lr_actual = lr_schedule(epoch) if lr_schedule is not None else learning_rate
        indices_mezclados = indices_train.copy()
        rng.shuffle(indices_mezclados)

        losses_batch: list[float] = []
        normas_gradiente: list[float] = []

        for batch_indices in iterar_chunks_indices(indices_mezclados, tamano_chunk=batch_size):
            X_batch = preparar_chunk_features(
                X_fuente,
                batch_indices,
                dtype=dtype,
                normalizar=normalizar,
                aplanar=aplanar,
            )
            y_batch = np.asarray(y_fuente[batch_indices], dtype=int)
            probabilidades, caches = model.forward(X_batch)

            probabilidades_clase_real = probabilidades[np.arange(y_batch.size), y_batch]
            loss_batch = float(
                -np.mean(np.log(np.clip(probabilidades_clase_real, 1e-12, 1.0)))
            )
            loss_batch += penalizacion_l2(model, l2_lambda=l2_lambda)
            losses_batch.append(loss_batch)

            gradientes = model.backward(
                probabilidades,
                y_batch,
                caches,
                normalizador=y_batch.size,
            )
            sumar_gradientes_l2_inplace(model, gradientes, l2_lambda=l2_lambda)
            normas_gradiente.append(
                clip_gradients_por_norma_global(
                    gradientes,
                    max_norm=norma_clip_gradiente,
                )
            )

            if optimizer == "sgd":
                model.aplicar_gradientes(gradientes, learning_rate=lr_actual)
            else:
                assert estado_adam is not None
                aplicar_update_adam(
                    model,
                    gradientes,
                    estado_adam,
                    learning_rate=lr_actual,
                )

        metrics_train = evaluar_clasificador(
            model,
            X_fuente,
            y_fuente,
            indices_eval_train,
            tamano_chunk=tamano_chunk_eval,
            n_classes=n_classes,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )
        metrics_validation = evaluar_clasificador(
            model,
            X_fuente,
            y_fuente,
            indices_eval_validation,
            tamano_chunk=tamano_chunk_eval,
            n_classes=n_classes,
            dtype=dtype,
            normalizar=normalizar,
            aplanar=aplanar,
        )

        train_loss = float(metrics_train["loss"])
        validation_loss = float(metrics_validation["loss"])
        penalizacion_regularizacion = penalizacion_l2(model, l2_lambda=l2_lambda)
        tiempo = time.perf_counter() - start_time

        fila = {
            "epoch": float(epoch),
            "learning_rate": float(lr_actual),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": float(metrics_train["accuracy"]),
            "validation_accuracy": float(metrics_validation["accuracy"]),
            "train_f1_macro": float(metrics_train["f1_macro"]),
            "validation_f1_macro": float(metrics_validation["f1_macro"]),
            "elapsed_seconds": tiempo,
            "mean_batch_loss": float(np.mean(losses_batch)),
            "train_objective": train_loss + penalizacion_regularizacion,
            "validation_objective": validation_loss + penalizacion_regularizacion,
            "mean_gradient_norm": float(np.mean(normas_gradiente)),
        }

        for key, value in fila.items():
            historial[key].append(value)

        mejoro = validation_loss < mejor_validation_loss - min_delta
        if mejoro:
            mejor_validation_loss = validation_loss
            mejores_parametros = model.copiar_parametros()
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
                f"Epoch {epoch:02d}/{epochs} | "
                f"lr={lr_actual:.5f} | "
                f"train CE={float(metrics_train['loss']):.4f} | "
                f"val CE={float(metrics_validation['loss']):.4f} | "
                f"train acc={float(metrics_train['accuracy']):.4f} | "
                f"val acc={float(metrics_validation['accuracy']):.4f} | "
                f"{tiempo:.1f}s"
            )

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

    if restore_best and mejores_parametros is not None:
        model.fijar_parametros(mejores_parametros)
        historial["restored_best_epoch"] = [
            historial["epoch"][int(np.argmin(historial["validation_loss"]))]
        ]
    if motivo_stop is not None:
        historial["stop_reason"] = [motivo_stop]

    return historial
