# Please see LICENSE.md
"""Breathing gases.

A Gas is a dict mapping 'fO2', 'fN2', 'fHe' to the respective gas fractions
(floats summing to 1). Note that TissueStateCython may add a derived
'cython_array' entry as a cache; it is not part of the gas definition.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


class Gas(dict):
    def __repr__(self) -> str:
        if self['fHe'] == 0:
            if self['fO2'] == 0.21:
                return 'Air'
            return f'Nx{int(100 * self["fO2"])}'
        return f'Tx{int(100 * self["fO2"])}/{int(100 * self["fHe"])}'

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self['fO2']) + 3 * hash(self['fHe'])

    def __eq__(self, other: object) -> bool:
        # Compare on gas fractions only, so a cached 'cython_array' entry
        # never affects equality.
        if self is None or other is None:
            return self is None and other is None
        return all(self[f] == other[f] for f in ('fO2', 'fN2', 'fHe'))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def partial_pressures(self, p_amb: float) -> tuple[float, float, float]:
        """Partial pressures (pO2, pN2, pHe) at ambient pressure p_amb."""
        return self['fO2'] * p_amb, self['fN2'] * p_amb, self['fHe'] * p_amb


def Air() -> Gas:
    return Gas({'fO2': 0.21, 'fN2': 0.79, 'fHe': 0.0})


def Nitrox(percO2: float) -> Gas:
    return Gas({'fO2': percO2 / 100, 'fN2': 1 - percO2 / 100, 'fHe': 0.0})


def Trimix(percO2: float, percHe: float) -> Gas:
    return Gas({'fO2': percO2 / 100,
                'fN2': 1 - percO2 / 100 - percHe / 100,
                'fHe': percHe / 100})


def from_string(s: str | None) -> Gas | None:
    """Parse 'air', 'nx50', 'tx18/45' (case-insensitive); None if unparseable."""
    if s is None:
        return None
    s = s.strip().lower()
    m = re.match('(?i)(air|nx[0-9][0-9]|tx[0-9][0-9]/[0-9][0-9])', s)
    if m is None:
        return None
    # Ok, at this point we know it's reasonably safe to parse.
    # If you're reading this and you disagree, let me know :)
    if s == 'air':
        return Air()
    if s.startswith('nx'):
        return Nitrox(int(s[2:4]))
    if s.startswith('tx'):
        return Trimix(int(s[2:4]), int(s[5:7]))
    return None


def many_from_string(s: str) -> list[Gas]:
    """Parse a comma-separated list of gases, silently dropping bad entries."""
    gases = (from_string(part) for part in s.split(','))
    return [g for g in gases if g is not None]


def best_gas(gases: Iterable[Gas], p_amb: float, max_pO2: float) -> Gas:
    """The best deco gas at this ambient pressure: the highest-O2 (then
    highest-He) gas whose pO2 does not exceed max_pO2. If none qualifies,
    pick from all gases."""
    gases = list(gases)
    suitable = [gas for gas in gases if p_amb * gas['fO2'] <= max_pO2]
    if len(suitable) == 0:
        suitable = gases
    return max(suitable, key=lambda g: (g['fO2'], g['fHe']))
