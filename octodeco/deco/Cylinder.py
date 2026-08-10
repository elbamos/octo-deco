# Please see LICENSE.md
"""Cylinders (tanks) and gas-volume bookkeeping."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Gas import Gas


class Cylinder:
    def __init__(self, name: str, size_L: float, max_pressure_bar: float):
        assert size_L > 0.0
        assert max_pressure_bar > 0.0
        self.name = name
        self.size_L = size_L
        self.max_pressure_bar = max_pressure_bar
        self.contains_L = size_L * max_pressure_bar

    def liters_to_bars(self, liters: float) -> float:
        """Pressure drop (bar) corresponding to using `liters` of gas."""
        return liters / self.size_L

    def liters_used_to_perc(self, liters: float) -> float:
        """Percentage of the full cylinder used after `liters` of gas."""
        return 100.0 * liters / self.contains_L

    def __repr__(self) -> str:
        return f'{self.name}[{self.size_L}L, {self.max_pressure_bar}bar]'

    def __str__(self) -> str:
        return self.name


def Guess(liters_used: float, gas: Gas) -> Cylinder:
    """Guess a plausible cylinder from the gas mix and the amount used."""
    if gas['fO2'] > 0.82:
        return Cylinder('cf40', 5.55, 200.0)
    if gas['fO2'] > 0.52:
        return Cylinder('Alu7', 7.0, 200.0)
    if gas['fO2'] > 0.42:
        return Cylinder('cf80', 11.1, 200.0)
    if liters_used < 1500:
        return Cylinder('cf80', 11.1, 200.0)
    return Cylinder('D12', 24, 200.0)
