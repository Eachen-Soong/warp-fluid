from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, init=False)
class GridSpec:
    """Uniform Cartesian grid supporting 2D and 3D domains."""

    nx: int
    ny: int
    nz: Optional[int]
    dx: float
    dy: float
    dz: Optional[float]
    origin: Tuple[float, ...] = (0.0, 0.0)

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: Optional[float] = None,
        *,
        nz: Optional[int] = None,
        dz: Optional[float] = None,
        origin: Tuple[float, ...] = (0.0, 0.0),
    ) -> None:
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", None if nz is None else int(nz))
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dx if dy is None else dy)
        object.__setattr__(self, "dz", None if nz is None else (float(dx) if dz is None else dz))
        object.__setattr__(self, "origin", origin)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("Grid dimensions must be positive.")
        if self.nz is not None and self.nz < 1:
            raise ValueError("nz must be positive when provided.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.dy <= 0.0:
            raise ValueError("dy must be positive.")
        if self.nz is not None and (self.dz is None or self.dz <= 0.0):
            raise ValueError("dz must be positive for 3D grids.")
        object.__setattr__(self, "dy", float(self.dy))
        object.__setattr__(self, "dx", float(self.dx))
        if self.nz is None:
            if len(self.origin) != 2:
                raise ValueError("2D grids require a 2D origin tuple.")
            object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))
            object.__setattr__(self, "dz", None)
        else:
            if len(self.origin) == 2:
                origin = (float(self.origin[0]), float(self.origin[1]), 0.0)
            elif len(self.origin) == 3:
                origin = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
            else:
                raise ValueError("3D grids require a 2D or 3D origin tuple.")
            object.__setattr__(self, "origin", origin)
            object.__setattr__(self, "dz", float(self.dz))

    @classmethod
    def from_extent(
        cls,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        origin: Tuple[float, ...] = (0.0, 0.0),
        *,
        nz: Optional[int] = None,
        lz: Optional[float] = None,
    ) -> "GridSpec":
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=float(lx) / nx,
            dy=float(ly) / ny,
            dz=None if nz is None else float(lz if lz is not None else lx) / nz,
            origin=origin,
        )

    @property
    def ndim(self) -> int:
        return 2 if self.nz is None else 3

    @property
    def is_3d(self) -> bool:
        return self.nz is not None

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.nz is None:
            return self.nx, self.ny
        return self.nx, self.ny, self.nz

    @property
    def shape_u(self) -> Tuple[int, ...]:
        if self.nz is None:
            return self.nx + 1, self.ny
        return self.nx + 1, self.ny, self.nz

    @property
    def shape_v(self) -> Tuple[int, ...]:
        if self.nz is None:
            return self.nx, self.ny + 1
        return self.nx, self.ny + 1, self.nz

    @property
    def shape_w(self) -> Tuple[int, ...]:
        if self.nz is None:
            raise AttributeError("2D grids do not define a w-face shape.")
        return self.nx, self.ny, self.nz + 1

    @property
    def extent(self) -> Tuple[float, ...]:
        if self.nz is None:
            return self.nx * self.dx, self.ny * self.dy
        return self.nx * self.dx, self.ny * self.dy, self.nz * float(self.dz)

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    @property
    def z0(self) -> float:
        return 0.0 if self.nz is None else float(self.origin[2])

    @property
    def x1(self) -> float:
        return self.x0 + self.extent[0]

    @property
    def y1(self) -> float:
        return self.y0 + self.extent[1]

    @property
    def z1(self) -> float:
        if self.nz is None:
            raise AttributeError("2D grids do not define z1.")
        return self.z0 + float(self.extent[2])

    @property
    def inv_dx(self) -> float:
        return 1.0 / self.dx

    @property
    def inv_dy(self) -> float:
        return 1.0 / self.dy

    @property
    def inv_dz(self) -> float:
        if self.nz is None or self.dz is None:
            raise AttributeError("2D grids do not define inv_dz.")
        return 1.0 / self.dz
