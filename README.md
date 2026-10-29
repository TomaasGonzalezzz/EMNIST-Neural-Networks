# EMNIST Neural Networks
### De un MLP en NumPy a redes en PyTorch

Reconocimiento de caracteres manuscritos sobre **EMNIST ByMerge**: 809.555 imágenes y 47 clases. El proyecto parte de una implementación de backpropagation en NumPy y compara mejoras de entrenamiento, redes equivalentes en PyTorch y resistencia al ruido.

**Tomás González** · Machine Learning, UdeSA · 2026.

![Comparación de los cuatro modelos en test](assets/comparacion-test.png)

## Experimentos

- M0: MLP base con ReLU, softmax y cross-entropy.
- M1: mini-batches, Adam, regularización L2, schedules y early stopping.
- M2: arquitectura equivalente a M1 en PyTorch.
- M3: selección de arquitectura, activaciones, BatchNorm y Dropout.

| Modelo | Accuracy test | Macro F1 test |
| --- | ---: | ---: |
| M0 · NumPy base | 0,5762 | 0,3328 |
| M1 · NumPy mejorado | 0,8888 | 0,8586 |
| M2 · PyTorch equivalente | 0,8879 | 0,8605 |
| M3 · PyTorch seleccionado | 0,8993 | 0,8753 |

Son resultados guardados de la entrega. También se compara la degradación ante ruido gaussiano.

## Versiones y ejecución

Hay dos versiones conservadas: la [publicada inicialmente](Entrega_TP3.ipynb) y la [entrega archivada](entrega_original/Entrega_TP3.ipynb). La tabla anterior corresponde a la segunda. Usar siempre notebook, código y artifacts de una misma versión.

El [dataset está en Releases](https://github.com/TomaasGonzalezzz/TP3_Machine_Learning/releases/tag/tp3-datasets-2026). Extraer los arrays en **Datasets/**, junto al notebook de la versión elegida.

Desde esa carpeta, con un entorno virtual activo:

~~~sh
python -m pip install -r requirements.txt
python -m notebook Entrega_TP3.ipynb
~~~

La entrega registra Python 3.14.3 y fija dependencias, incluida una compilación CPU de PyTorch. Esa instalación completa no fue verificada al preparar el portfolio.

Los artifacts incluyen métricas y parámetros NumPy. No hay checkpoints completos de M2/M3: sus predicciones requieren reentrenamiento.

[Detalle de versiones, instalación y comprobaciones](DOCUMENTACION.md) · [Consigna](docs/consigna.pdf).
