"""Corre experimentos PyTorch del punto 4 y guarda artifacts reutilizables."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.data import cargar_arrays_emnist, indices_split_estratificado
from src.experiments import cargar_modelo_desde_parametros
from src.torch_experiments import (
    config_por_model_id,
    grilla_experimentos_torch_por_defecto,
    rutas_artifacts_experimento,
    cargar_modelo,
    config_experimento_m2,
    armar_config_m3,
    correr_experimento_torch,
    correr_grilla_experimentos_torch,
    seleccionar_mejor_experimento_torch,
)
from src.torch_training import evaluar_modelo_torch, obtener_device_torch
from src.training import evaluar_clasificador


DIR_ARTIFACTS = RAIZ_PROYECTO / "artifacts"


def arquitectura_desde_parametros_numpy(path: Path) -> tuple[int, ...]:
    """Infiere la arquitectura de un MLP NumPy desde matrices de pesos guardadas."""

    data = np.load(path)
    weight_keys = sorted(
        [key for key in data.files if key.startswith("W")],
        key=lambda key: int(key[1:]),
    )
    architecture = [int(data[weight_keys[0]].shape[0])]
    architecture.extend(int(data[key].shape[1]) for key in weight_keys)
    return tuple(architecture)


def copiar_artifacts_torch_a_raiz(model_id: str, nombre_grilla: str) -> None:
    """Copia un experimento PyTorch guardado a nombres canonicos en artifacts."""

    rutas = rutas_artifacts_experimento(DIR_ARTIFACTS, model_id, nombre_grilla=nombre_grilla)
    for source_key, target_name in [
        ("history", f"{model_id.lower()}_history.npz"),
        ("model", f"{model_id.lower()}_model.pt"),
        ("metrics", f"{model_id.lower()}_metrics.npz"),
        ("config", f"{model_id.lower()}_config.csv"),
    ]:
        shutil.copy2(rutas[source_key], DIR_ARTIFACTS / target_name)


def evaluar_todos_los_modelos(
    X: np.ndarray,
    y: np.ndarray,
    indices_test: np.ndarray,
    *,
    m2_config,
    m3_config,
    device,
) -> pd.DataFrame:
    """Evalua M0, M1, M2 y M3 sobre el split test."""

    m0_architecture = arquitectura_desde_parametros_numpy(DIR_ARTIFACTS / "m0_parameters.npz")
    m1_architecture = arquitectura_desde_parametros_numpy(DIR_ARTIFACTS / "m1_parameters.npz")
    m0 = cargar_modelo_desde_parametros(DIR_ARTIFACTS / "m0_parameters.npz", m0_architecture)
    m1 = cargar_modelo_desde_parametros(DIR_ARTIFACTS / "m1_parameters.npz", m1_architecture)

    m2_paths = rutas_artifacts_experimento(DIR_ARTIFACTS, "M2", nombre_grilla="torch_m2")
    m3_paths = rutas_artifacts_experimento(DIR_ARTIFACTS, "M3", nombre_grilla="torch_final")
    m2 = cargar_modelo(m2_paths["model"], m2_config, device=device)
    m3 = cargar_modelo(m3_paths["model"], m3_config, device=device)

    filas = []
    matrices = {}
    for model_id, model, evaluator in [
        ("M0", m0, "numpy"),
        ("M1", m1, "numpy"),
        ("M2", m2, "torch"),
        ("M3", m3, "torch"),
    ]:
        if evaluator == "numpy":
            metrics = evaluar_clasificador(
                model,
                X,
                y,
                indices_test,
                n_classes=47,
                tamano_chunk=32768,
            )
        else:
            metrics = evaluar_modelo_torch(
                model,
                X,
                y,
                indices_test,
                n_classes=47,
                batch_size=32768,
                device=device,
            )
        filas.append(
            {
                "model_id": model_id,
                "test_loss": float(metrics["loss"]),
                "test_accuracy": float(metrics["accuracy"]),
                "test_f1_macro": float(metrics["f1_macro"]),
                "n_samples": int(metrics["n_samples"]),
            }
        )
        matrices[f"{model_id.lower()}_confusion_matrix"] = metrics["confusion_matrix"]

    np.savez_compressed(DIR_ARTIFACTS / "point4_test_confusion_matrices.npz", **matrices)
    resumen = pd.DataFrame(filas)
    resumen.to_csv(DIR_ARTIFACTS / "point4_test_summary.csv", index=False)
    return resumen


def evaluar_robustness_noise(
    X: np.ndarray,
    y: np.ndarray,
    indices_test: np.ndarray,
    *,
    m2_config,
    m3_config,
    device,
) -> pd.DataFrame:
    """Evalua los cuatro models con niveles crecientes de Gaussian noise."""

    m0_architecture = arquitectura_desde_parametros_numpy(DIR_ARTIFACTS / "m0_parameters.npz")
    m1_architecture = arquitectura_desde_parametros_numpy(DIR_ARTIFACTS / "m1_parameters.npz")
    m0 = cargar_modelo_desde_parametros(DIR_ARTIFACTS / "m0_parameters.npz", m0_architecture)
    m1 = cargar_modelo_desde_parametros(DIR_ARTIFACTS / "m1_parameters.npz", m1_architecture)

    m2_paths = rutas_artifacts_experimento(DIR_ARTIFACTS, "M2", nombre_grilla="torch_m2")
    m3_paths = rutas_artifacts_experimento(DIR_ARTIFACTS, "M3", nombre_grilla="torch_final")
    m2 = cargar_modelo(m2_paths["model"], m2_config, device=device)
    m3 = cargar_modelo(m3_paths["model"], m3_config, device=device)

    noise_levels = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
    filas = []
    for noise_std in noise_levels:
        print(f"\nNoise std={noise_std:.2f}", flush=True)
        for model_id, model, evaluator in [
            ("M0", m0, "numpy"),
            ("M1", m1, "numpy"),
            ("M2", m2, "torch"),
            ("M3", m3, "torch"),
        ]:
            if evaluator == "numpy":
                metrics = evaluar_clasificador(
                    model,
                    X,
                    y,
                    indices_test,
                    n_classes=47,
                    tamano_chunk=32768,
                    noise_std=noise_std,
                    random_state=123,
                )
            else:
                metrics = evaluar_modelo_torch(
                    model,
                    X,
                    y,
                    indices_test,
                    n_classes=47,
                    batch_size=32768,
                    device=device,
                    noise_std=noise_std,
                    random_state=123,
                )
            filas.append(
                {
                    "model_id": model_id,
                    "noise_std": noise_std,
                    "loss": float(metrics["loss"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1_macro": float(metrics["f1_macro"]),
                }
            )
            print(
                f"{model_id}: CE={float(metrics['loss']):.4f} "
                f"acc={float(metrics['accuracy']):.4f} "
                f"F1={float(metrics['f1_macro']):.4f}",
                flush=True,
            )

    robustness = pd.DataFrame(filas)
    robustness.to_csv(DIR_ARTIFACTS / "point4_noise_robustness.csv", index=False)
    return robustness


def principal() -> None:
    """Genera todos los artifacts del punto 4."""

    DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("Cargando arrays EMNIST y split reproducible...", flush=True)
    X, y = cargar_arrays_emnist(mmap_mode="r")
    particiones = indices_split_estratificado(y, random_state=42)
    device = obtener_device_torch()
    print(f"Device usado: {device}", flush=True)

    print("\nEntrenando/cargando M2...", flush=True)
    m2_config = config_experimento_m2()
    m2_summary_row = correr_experimento_torch(
        m2_config,
        X,
        y,
        particiones.train,
        particiones.validation,
        dir_artifacts=DIR_ARTIFACTS,
        nombre_grilla="torch_m2",
        force=False,
        batch_size_eval=32768,
        device=device,
        verbose=True,
    )
    copiar_artifacts_torch_a_raiz("M2", "torch_m2")
    pd.DataFrame([m2_summary_row]).to_csv(
        DIR_ARTIFACTS / "torch_m2_experiment_summary.csv",
        index=False,
    )

    print("\nEntrenando/cargando pool PyTorch P4A-P4T...", flush=True)
    grid_configs = grilla_experimentos_torch_por_defecto()
    grid_summary = correr_grilla_experimentos_torch(
        grid_configs,
        X,
        y,
        particiones.train,
        particiones.validation,
        dir_artifacts=DIR_ARTIFACTS,
        nombre_grilla="torch_grid_v1",
        force=False,
        tamano_curva_train=60000,
        tamano_curva_validation=None,
        batch_size_eval=32768,
        device=device,
        verbose=True,
    )
    print("\nTop candidates:", flush=True)
    print(
        grid_summary.sort_values("validation_loss")
        .head(5)[
            [
                "model_id",
                "validation_loss",
                "validation_accuracy",
                "validation_f1_macro",
                "completed_epochs",
                "best_epoch",
            ]
        ]
        .to_string(index=False),
        flush=True,
    )

    best_row = seleccionar_mejor_experimento_torch(grid_summary)
    best_config = config_por_model_id(grid_configs, str(best_row["model_id"]))
    m3_config = armar_config_m3(best_config, epochs=100)

    print(f"\nEntrenando/cargando M3 desde {best_config.model_id}...", flush=True)
    m3_summary_row = correr_experimento_torch(
        m3_config,
        X,
        y,
        particiones.train,
        particiones.validation,
        dir_artifacts=DIR_ARTIFACTS,
        nombre_grilla="torch_final",
        force=False,
        batch_size_eval=32768,
        device=device,
        verbose=True,
    )
    copiar_artifacts_torch_a_raiz("M3", "torch_final")
    pd.DataFrame([m3_summary_row]).to_csv(
        DIR_ARTIFACTS / "torch_m3_experiment_summary.csv",
        index=False,
    )

    print("\nEvaluando final models en test...", flush=True)
    test_summary = evaluar_todos_los_modelos(
        X,
        y,
        particiones.test,
        m2_config=m2_config,
        m3_config=m3_config,
        device=device,
    )
    print(test_summary.to_string(index=False), flush=True)

    print("\nEvaluando robustness ante noise...", flush=True)
    evaluar_robustness_noise(
        X,
        y,
        particiones.test,
        m2_config=m2_config,
        m3_config=m3_config,
        device=device,
    )
    print("\nArtifacts del punto 4 listos.", flush=True)


if __name__ == "__main__":
    principal()
