from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from .grid import GridSpec


def _ensure_shape(actual: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{name} has shape {actual}, expected {expected}")


@dataclass
class CenteredField:
    """Scalar or vector field stored at cell centers."""

    grid: GridSpec
    data: wp.array

    def __post_init__(self) -> None:
        if hasattr(self.data, "shape"):
            shape = tuple(self.data.shape)
            if len(shape) < 2 or shape[0] != self.grid.nx or shape[1] != self.grid.ny:
                raise ValueError(f"CenteredField expects leading shape {(self.grid.nx, self.grid.ny)} but got {shape}")

    @classmethod
    def zeros(
        cls,
        grid: GridSpec,
        *,
        channels: Optional[int] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "CenteredField":
        device = device or "cpu"
        shape = grid.shape if channels is None else grid.shape + (channels,)
        return cls(grid, wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=requires_grad))

    @classmethod
    def from_numpy(
        cls,
        grid: GridSpec,
        data: np.ndarray,
        *,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "CenteredField":
        device = device or "cpu"
        array = np.asarray(data, dtype=np.float32)
        return cls(grid, wp.array(array, dtype=wp.float32, device=device, requires_grad=requires_grad))

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data.numpy(), dtype=np.float32)


@dataclass
class MACField:
    """Velocity field on a 2D MAC grid."""

    grid: GridSpec
    u: wp.array
    v: wp.array

    def __post_init__(self) -> None:
        if hasattr(self.u, "shape"):
            _ensure_shape(tuple(self.u.shape), self.grid.shape_u, "u")
        if hasattr(self.v, "shape"):
            _ensure_shape(tuple(self.v.shape), self.grid.shape_v, "v")

    @classmethod
    def zeros(
        cls,
        grid: GridSpec,
        *,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "MACField":
        device = device or "cpu"
        return cls(
            grid=grid,
            u=wp.zeros(grid.shape_u, dtype=wp.float32, device=device, requires_grad=requires_grad),
            v=wp.zeros(grid.shape_v, dtype=wp.float32, device=device, requires_grad=requires_grad),
        )

    @classmethod
    def from_numpy(
        cls,
        grid: GridSpec,
        u: np.ndarray,
        v: np.ndarray,
        *,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "MACField":
        device = device or "cpu"
        u_arr = np.asarray(u, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        _ensure_shape(tuple(u_arr.shape), grid.shape_u, "u")
        _ensure_shape(tuple(v_arr.shape), grid.shape_v, "v")
        return cls(
            grid=grid,
            u=wp.array(u_arr, dtype=wp.float32, device=device, requires_grad=requires_grad),
            v=wp.array(v_arr, dtype=wp.float32, device=device, requires_grad=requires_grad),
        )

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.u.numpy(), dtype=np.float32),
            np.asarray(self.v.numpy(), dtype=np.float32),
        )
