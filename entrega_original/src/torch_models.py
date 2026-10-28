"""Definiciones de modelos PyTorch para el punto 4 del TP3."""

import torch
from torch import nn


def capa_activacion(name: str) -> nn.Module:
    """Devuelve una activation layer de PyTorch por nombre."""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Activacion desconocida: {name}")


class MLPTorch(nn.Module):
    """MLP flexible para clasificacion de EMNIST."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        capas: list[nn.Module] = []
        dim_anterior = input_dim

        for hidden_dim in hidden_sizes:
            capas.append(nn.Linear(dim_anterior, hidden_dim))
            if batch_norm:
                capas.append(nn.BatchNorm1d(hidden_dim))
            capas.append(capa_activacion(activation))
            if dropout > 0:
                capas.append(nn.Dropout(dropout))
            dim_anterior = hidden_dim

        capas.append(nn.Linear(dim_anterior, output_dim))
        self.net = nn.Sequential(*capas)
        self.config = {
            "input_dim": input_dim,
            "hidden_sizes": hidden_sizes,
            "output_dim": output_dim,
            "activation": activation,
            "dropout": dropout,
            "batch_norm": batch_norm,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits para cada class."""

        return self.net(x)


def contar_parametros(model: nn.Module) -> int:
    """Cuenta parametros entrenables."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
