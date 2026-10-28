# TP3 - Machine Learning y Deep Learning

Este repositorio contiene mi entrega del TP3. La idea es que el profesor pueda
abrir primero el notebook, seguir el razonamiento de punta a punta, y despues
mirar el codigo de soporte si quiere revisar como esta implementado cada paso.

La entrega principal es:

```text
Entrega_TP3.ipynb
```

Ese notebook es el relato completo del trabajo: carga de datos, baseline en
NumPy, mejoras de training, seleccion de modelos, parte PyTorch, comparacion en
test y analisis de robustness ante noise.

## Como leer el TP

El orden recomendado es este:

1. Abrir `Entrega_TP3.ipynb`.
2. Leerlo de arriba hacia abajo. El notebook esta pensado como una secuencia
   narrativa, no como celdas sueltas.
3. Los graficos y tablas ya estan ejecutados. No hace falta reentrenar modelos
   para corregir el analisis.
4. Si se quiere revisar implementacion, mirar `src/` siguiendo el mapa de abajo.
5. Si se quiere regenerar los experimentos pesados, mirar la seccion
   "Como volver a entrenar".

Los nombres de funciones y auxiliares estan en espanol. Las palabras propias de
la materia quedan en ingles: `train`, `validation`, `test`, `loss`, `accuracy`,
`F1 macro`, `learning_rate`, `optimizer`, `batch_size`, `Dropout`, `BatchNorm`,
`early_stopping`, `forward`, `backward`, etc.

## Estructura de archivos

```text
.
+-- Entrega_TP3.ipynb
+-- requirements.txt
+-- Datasets/
|   +-- X_images.npy
|   +-- y_images.npy
+-- artifacts/
+-- src/
|   +-- data.py
|   +-- metrics.py
|   +-- models.py
|   +-- training.py
|   +-- experiments.py
|   +-- torch_models.py
|   +-- torch_training.py
|   +-- torch_experiments.py
|   +-- plots.py
+-- scripts/
    +-- run_point4_artifacts.py
```

## Que hay en cada carpeta

`Datasets/` tiene los datos originales en formato NumPy:

- `X_images.npy`: imagenes EMNIST ByMerge de `28 x 28`.
- `y_images.npy`: labels numericos de las `47` clases.

`artifacts/` tiene resultados ya guardados para que el notebook sea liviano de
correr. Son archivos compactos: histories, metrics, parametros NumPy y summaries.
Sirven para mostrar graficos y tablas sin volver a entrenar todo.

`src/` tiene el codigo reutilizable:

- `data.py`: carga de datos, resumen del dataset y split estratificado.
- `metrics.py`: `accuracy` y `F1 macro` desde la `confusion matrix`.
- `models.py`: MLP en NumPy (`ClasificadorMLP`) con `forward` y `backward`.
- `training.py`: training NumPy, `full-batch gradient descent`, `mini-batch`,
  `Adam`, `L2`, `gradient clipping` y evaluacion por chunks.
- `experiments.py`: grilla NumPy `MA`-`MP`, guardado/carga de artifacts y
  seleccion del mejor modelo para `M1`.
- `torch_models.py`: MLP flexible en PyTorch.
- `torch_training.py`: training/evaluacion PyTorch.
- `torch_experiments.py`: pool PyTorch `P4A`-`P4T`, `M2`, `M3` y guardado de
  resultados.
- `plots.py`: funciones de graficos usadas por el notebook.

`scripts/` tiene scripts auxiliares. El mas importante es:

- `run_point4_artifacts.py`: regenera los artifacts pesados del punto 4
  (`M2`, pool PyTorch, `M3`, test final y robustness ante noise).

## Como instalar y abrir

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gonzalez_tp3 --display-name "Gonzalez TP3"
jupyter notebook Entrega_TP3.ipynb
```

Si ya existe la `.venv`, alcanza con activarla:

```powershell
.\.venv\Scripts\activate
jupyter notebook Entrega_TP3.ipynb
```

## Como volver a entrenar

Por defecto, el TP viene listo para leer y corregir sin entrenar todo de nuevo.
Si se quiere regenerar resultados, hay que cambiar algunos flags.

### M0

En `Entrega_TP3.ipynb`, buscar:

```python
FORCE_RETRAIN_M0 = False
```

Cambiarlo a:

```python
FORCE_RETRAIN_M0 = True
```

Eso vuelve a entrenar `M0` con `full-batch gradient descent`.

### Grilla NumPy MA-MP

En `Entrega_TP3.ipynb`, buscar:

```python
FORCE_RETRAIN_GRID = False
```

Cambiarlo a:

```python
FORCE_RETRAIN_GRID = True
```

Eso vuelve a correr la grilla NumPy `MA`-`MP` definida en
`src/experiments.py`.

### M1 y finalistas de 100 epochs

El notebook lee estos resultados desde:

```text
artifacts/finalists_100/finalists_100_summary.csv
artifacts/m1_history.npz
artifacts/m1_metrics.npz
artifacts/m1_parameters.npz
```

Esos archivos quedan incluidos porque son la decision final de `M1` ya
consolidada. Para recalcularlos desde cero, hay que usar las configs de
`src/experiments.py`, correr los finalistas elegidos con `force=True` y volver a
guardar los archivos `m1_*` y `finalists_100_summary.csv`. No se hace por defecto
en el notebook para que la correccion no dependa de entrenamientos largos.

### Punto 4: M2, pool PyTorch y M3

Para regenerar el punto 4 completo, correr:

```powershell
.\.venv\Scripts\python.exe scripts\run_point4_artifacts.py
```

Ese script entrena/carga:

- `M2`, equivalente PyTorch de `M1`.
- El pool PyTorch `P4A`-`P4T`.
- `M3`, elegido desde el mejor candidato del pool.
- La comparacion final en `test`.
- La prueba de robustness con Gaussian noise.

Si se quiere obligar el reentrenamiento aunque ya existan artifacts internos,
abrir `scripts/run_point4_artifacts.py` y cambiar los tres `force=False` de las
llamadas principales por `force=True`:

```python
correr_experimento_torch(..., force=True, ...)
correr_grilla_experimentos_torch(..., force=True, ...)
correr_experimento_torch(..., force=True, ...)
```

Al correr ese script pueden aparecer subcarpetas adicionales dentro de
`artifacts/` (`torch_m2/`, `torch_grid_v1/`, `torch_final/`) y archivos `.pt`.
No estan incluidos en la version compacta porque el notebook final solo necesita
los summaries, histories, metrics y matrices guardadas.



