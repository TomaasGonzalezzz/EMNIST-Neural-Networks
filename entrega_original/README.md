# TP3 - Machine Learning y Deep Learning

Entrega del TP3 de I302. El archivo principal es:

```text
Entrega_TP3.ipynb
```

El notebook esta pensado para leerse de principio a fin. Ahi estan el analisis
del dataset, el MLP implementado en NumPy, la seleccion de `M1`, la parte en
PyTorch (`M2` y `M3`), la comparacion final en test y el analisis con ruido.

Los graficos y tablas ya estan ejecutados. Para corregir el trabajo no hace
falta volver a entrenar los modelos pesados: los resultados necesarios estan
guardados en `artifacts/`.

## Estructura

```text
.
+-- Entrega_TP3.ipynb
+-- requirements.txt
+-- Datasets/
+-- artifacts/
+-- src/
+-- scripts/
```

- `Datasets/`: contiene `X_images.npy` y `y_images.npy`.
- `artifacts/`: contiene metricas, histories, parametros y summaries ya
  calculados.
- `src/`: contiene el codigo reutilizable del TP.
- `scripts/`: contiene el script para regenerar los resultados pesados del
  punto 4.

## Como correrlo

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook Entrega_TP3.ipynb
```

Si el entorno ya existe, alcanza con activarlo y abrir el notebook:

```powershell
.\.venv\Scripts\activate
jupyter notebook Entrega_TP3.ipynb
```

## Reentrenamiento

El notebook viene preparado para correr rapido leyendo `artifacts/`.

Si se quiere recalcular:

- `M0`: cambiar `FORCE_RETRAIN_M0 = True` en el notebook.
- Grilla NumPy `MA`-`MP`: cambiar `FORCE_RETRAIN_GRID = True` en el notebook.
- Punto 4 (`M2`, pool PyTorch, `M3`, test y ruido): correr

```powershell
.\.venv\Scripts\python.exe scripts\run_point4_artifacts.py
```

Ese comando reutiliza los artifacts si ya existen. Para forzar el reentrenamiento
completo de PyTorch, correr:

```powershell
.\.venv\Scripts\python.exe scripts\run_point4_artifacts.py --force
```

El reentrenamiento completo puede tardar bastante. Por eso los artifacts
compactos ya quedan incluidos en la entrega.

Los finalistas de `M1` a `100` epochs tambien quedan incluidos como artifacts
compactos (`artifacts/finalists_100/` y archivos `m1_*`). El notebook los lee
directo para no depender de otra corrida larga durante la correccion.

## Uso de la IA

En el desarrollo del trabajo practico se utilizo Codex solamente como apoyo para
la graficacion y la presentacion visual de resultados, principalmente en
`src/plots.py` y en celdas de visualizacion del notebook.

Los archivos de `artifacts/` guardan resultados ya calculados para evitar
repetir entrenamientos largos cada vez que se abre el notebook.
