from typing import Any, Protocol, runtime_checkable

import torch
# pyright: reportExplicitAny=none

@runtime_checkable
class CallableDenoiser(Protocol):
    def __call__(
        self,
        # Latent tensor, named differently in different architectures
        # MUST be called as positional argument
        latent: torch.Tensor,
        # All remaining parameters MUST be called as keyword argument
        # since their order are not guaranteed in implementations
        timestep: torch.LongTensor | torch.Tensor | int | float,
        **kwargs: dict[str, Any],
    ):
        ...