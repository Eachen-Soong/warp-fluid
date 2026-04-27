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
            leading = tuple(shape[: self.grid.ndim])
            if len(shape) < self.grid.ndim or leading != self.grid.shape:
                raise ValueError(f"CenteredField expects leading shape {self.grid.shape} but got {shape}")

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
    """Velocity field on a MAC grid."""

    grid: GridSpec
    u: wp.array
    v: wp.array
    w: Optional[wp.array] = None

    def __post_init__(self) -> None:
        if hasattr(self.u, "shape"):
            _ensure_shape(tuple(self.u.shape), self.grid.shape_u, "u")
        if hasattr(self.v, "shape"):
            _ensure_shape(tuple(self.v.shape), self.grid.shape_v, "v")
        if self.grid.is_3d:
            if self.w is None:
                raise ValueError("3D MACField expects a w component.")
            if hasattr(self.w, "shape"):
                _ensure_shape(tuple(self.w.shape), self.grid.shape_w, "w")
        elif self.w is not None:
            raise ValueError("2D MACField must not define a w component.")

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
            w=(
                wp.zeros(grid.shape_w, dtype=wp.float32, device=device, requires_grad=requires_grad)
                if grid.is_3d
                else None
            ),
        )

    @classmethod
    def from_numpy(
        cls,
        grid: GridSpec,
        u: np.ndarray,
        v: np.ndarray,
        w: Optional[np.ndarray] = None,
        *,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "MACField":
        device = device or "cpu"
        u_arr = np.asarray(u, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        _ensure_shape(tuple(u_arr.shape), grid.shape_u, "u")
        _ensure_shape(tuple(v_arr.shape), grid.shape_v, "v")
        if grid.is_3d:
            if w is None:
                raise ValueError("3D grids require a w-face array.")
            w_arr = np.asarray(w, dtype=np.float32)
            _ensure_shape(tuple(w_arr.shape), grid.shape_w, "w")
        else:
            if w is not None:
                raise ValueError("2D grids must not pass a w-face array.")
            w_arr = None
        return cls(
            grid=grid,
            u=wp.array(u_arr, dtype=wp.float32, device=device, requires_grad=requires_grad),
            v=wp.array(v_arr, dtype=wp.float32, device=device, requires_grad=requires_grad),
            w=(
                wp.array(w_arr, dtype=wp.float32, device=device, requires_grad=requires_grad)
                if w_arr is not None
                else None
            ),
        )

    def numpy(self) -> tuple[np.ndarray, ...]:
        values = [
            np.asarray(self.u.numpy(), dtype=np.float32),
            np.asarray(self.v.numpy(), dtype=np.float32),
        ]
        if self.w is not None:
            values.append(np.asarray(self.w.numpy(), dtype=np.float32))
        return tuple(values)
