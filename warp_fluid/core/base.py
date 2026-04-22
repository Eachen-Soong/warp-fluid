"""Backward-compatible aliases for legacy experiments."""

from .field import CenteredField, MACField
from .grid import GridSpec

CollocatedGrid = CenteredField

__all__ = ["CenteredField", "CollocatedGrid", "GridSpec", "MACField"]

    
