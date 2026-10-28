"""Definiciones de modelos para la implementacion NumPy."""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Activacion ReLU."""

    return np.maximum(0, z)


def derivada_relu(z: np.ndarray) -> np.ndarray:
    """Derivada de ReLU respecto de su input."""

    return (z > 0).astype(z.dtype, copy=False)


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable."""

    desplazado = z - np.max(z, axis=1, keepdims=True)
    valores_exp = np.exp(desplazado)
    return valores_exp / np.sum(valores_exp, axis=1, keepdims=True)


class ClasificadorMLP:
    """MLP classifier implementado solo con NumPy."""

    def __init__(
        self,
        layer_sizes: list[int],
        *,
        random_state: int = 42,
        weight_scale: float | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes debe incluir input y output sizes.")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.dtype = np.dtype(dtype)
        self.rng = np.random.default_rng(random_state)
        self.parametros = self._inicializar_parametros(weight_scale=weight_scale)

    def _inicializar_parametros(
        self,
        *,
        weight_scale: float | None,
    ) -> dict[str, np.ndarray]:
        parametros: dict[str, np.ndarray] = {}
        for layer_idx, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:]),
            start=1,
        ):
            # He initialization funciona bien como default para hidden layers con ReLU.
            scale = np.sqrt(2.0 / fan_in) if weight_scale is None else weight_scale
            parametros[f"W{layer_idx}"] = self.rng.normal(
                loc=0.0,
                scale=scale,
                size=(fan_in, fan_out),
            ).astype(self.dtype)
            parametros[f"b{layer_idx}"] = np.zeros((1, fan_out), dtype=self.dtype)
        return parametros

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Ejecuta forward pass y devuelve probabilidades mas caches."""

        activation = np.asarray(X, dtype=self.dtype)
        caches: list[dict[str, np.ndarray]] = []

        for layer_idx in range(1, self.n_layers + 1):
            W = self.parametros[f"W{layer_idx}"]
            b = self.parametros[f"b{layer_idx}"]
            z = activation @ W + b
            next_activation = softmax(z) if layer_idx == self.n_layers else relu(z)
            caches.append({"a_prev": activation, "z": z, "a": next_activation})
            activation = next_activation

        return activation, caches

    def backward(
        self,
        probabilidades: np.ndarray,
        y: np.ndarray,
        caches: list[dict[str, np.ndarray]],
        *,
        normalizador: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Devuelve gradientes para backpropagation con softmax cross-entropy."""

        y = np.asarray(y, dtype=int)
        if normalizador is None:
            normalizador = y.size

        gradientes = self.gradientes_en_cero()
        dz = probabilidades.astype(self.dtype, copy=True)
        dz[np.arange(y.size), y] -= 1.0

        for layer_idx in range(self.n_layers, 0, -1):
            cache = caches[layer_idx - 1]
            a_prev = cache["a_prev"]
            gradientes[f"W{layer_idx}"] = (a_prev.T @ dz) / normalizador
            gradientes[f"b{layer_idx}"] = np.sum(dz, axis=0, keepdims=True) / normalizador

            if layer_idx > 1:
                W = self.parametros[f"W{layer_idx}"]
                z_anterior = caches[layer_idx - 2]["z"]
                dz = (dz @ W.T) * derivada_relu(z_anterior)

        return gradientes

    def gradientes_en_cero(self) -> dict[str, np.ndarray]:
        """Crea un diccionario de gradientes en cero con la forma de los parametros."""

        return {
            key: np.zeros_like(value, dtype=self.dtype)
            for key, value in self.parametros.items()
        }

    def copiar_parametros(self) -> dict[str, np.ndarray]:
        """Devuelve una copia profunda de los parametros del model."""

        return {key: value.copy() for key, value in self.parametros.items()}

    def fijar_parametros(self, parametros: dict[str, np.ndarray]) -> None:
        """Reemplaza los parametros del model usando copias."""

        self.parametros = {
            key: np.asarray(value, dtype=self.dtype).copy()
            for key, value in parametros.items()
        }

    @staticmethod
    def sumar_gradientes_inplace(
        acumulado: dict[str, np.ndarray],
        actualizacion: dict[str, np.ndarray],
    ) -> None:
        """Acumula gradientes in place."""

        for key in acumulado:
            acumulado[key] += actualizacion[key]

    def aplicar_gradientes(
        self,
        gradientes: dict[str, np.ndarray],
        *,
        learning_rate: float,
    ) -> None:
        """Aplica un update de gradient descent."""

        for key in self.parametros:
            self.parametros[key] -= learning_rate * gradientes[key]

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Devuelve probabilidades por class."""

        probabilidades, _ = self.forward(X)
        return probabilidades
