from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, init=False)
class GridSpec:
    """Uniform 2D Cartesian grid."""

    nx: int
    ny: int
    dx: float
    dy: float
    origin: Tuple[float, float] = (0.0, 0.0)

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: Optional[float] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dx if dy is None else dy)
        object.__setattr__(self, "origin", origin)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("Grid dimensions must be positive.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.dy <= 0.0:
            raise ValueError("dy must be positive.")
        object.__setattr__(self, "dy", float(self.dy))
        object.__setattr__(self, "dx", float(self.dx))
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))

    @classmethod
    def from_extent(
        cls,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "GridSpec":
        return cls(nx=nx, ny=ny, dx=float(lx) / nx, dy=float(ly) / ny, origin=origin)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nx, self.ny

    @property
    def shape_u(self) -> Tuple[int, int]:
        return self.nx + 1, self.ny

    @property
    def shape_v(self) -> Tuple[int, int]:
        return self.nx, self.ny + 1

    @property
    def extent(self) -> Tuple[float, float]:
        return self.nx * self.dx, self.ny * self.dy

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    @property
    def x1(self) -> float:
        return self.x0 + self.extent[0]

    @property
    def y1(self) -> float:
        return self.y0 + self.extent[1]

    @property
    def inv_dx(self) -> float:
        return 1.0 / self.dx

    @property
    def inv_dy(self) -> float:
        return 1.0 / self.dy
