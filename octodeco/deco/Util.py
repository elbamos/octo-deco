# Please see LICENSE.md
"""Depth/pressure conversions and small formatting helpers.

Conventions used throughout the package:
* time is in minutes,
* depth is in meters,
* pressure is in bar.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .Gas import Gas

# Water pressure per EN 13319: density 1020 kg/m^3.
# Pressure is measured in Pa = N/m^2; 1 bar = 10^5 Pa.
# 1020 kg/m^3 * 9.80 m/s^2 = 9996 Pa/m = 0.09996 bar/m,
# so 10 meters of water (at EN 13319) is 0.9996 bar.
#
# For comparison:
# 10 m salt/sea water = 1.00693064 bar
# 10 m fresh water    = 0.980638 bar
# So that seems to work out just about right.

SURFACE_PRESSURE = 1.01325           # bar
BAR_PER_METER = 1020 * 9.80 * 1e-5   # = 0.09996
METER_PER_BAR = 1 / BAR_PER_METER    # = 10.0040

# A single decompression stop: (depth, duration, gas breathed at the stop),
# optionally extended with a fourth element: the ascent speed (m/min) to use
# when leaving the stop, ie for the travel from this stop to the next,
# shallower one. Without it, ascents happen at the dive's normal ascent speed.
Stop = tuple[float, float, 'Gas'] | tuple[float, float, 'Gas', float]


def Pamb_to_depth(p_amb: float) -> float:
    """Convert ambient pressure (bar) to depth (m); 0 m at surface pressure."""
    return METER_PER_BAR * (p_amb - SURFACE_PRESSURE)


def depth_to_Pamb(depth: float) -> float:
    """Convert depth (m) to ambient pressure (bar)."""
    return depth * BAR_PER_METER + SURFACE_PRESSURE


def Pamb_to_Pamb_stop(p_amb: float,
                      direction: Literal['down', 'up'] = 'down',
                      last_stop_depth: float = 3) -> float:
    """Round an ambient pressure to the nearest deco stop pressure.

    Stops live on multiples of 3 m; 'down' rounds to the next deeper stop,
    'up' to the next shallower one. Depths between the surface and
    last_stop_depth snap to last_stop_depth ('down') or the surface ('up').
    """
    depth = Pamb_to_depth(p_amb)
    if (depth % 3) > 0.01:
        if direction == 'down':
            depth = 3 * (math.floor(depth / 3) + 1)
        elif direction == 'up':
            depth = 3 * (math.ceil(depth / 3) - 1)
    if depth < 0:
        depth = 0
    elif 0 < depth < last_stop_depth - 0.1:
        if direction == 'down':
            depth = last_stop_depth
        elif direction == 'up':
            depth = 0
    return depth_to_Pamb(depth)


def next_stop_Pamb(p_amb: float, last_stop_depth: float = 3) -> float:
    """Ambient pressure of the next (3 m shallower) stop; the surface if that
    would be shallower than the last stop."""
    r = p_amb - 3 * BAR_PER_METER
    if r < depth_to_Pamb(last_stop_depth) - 0.01:
        r = SURFACE_PRESSURE
    elif r < SURFACE_PRESSURE:
        r = SURFACE_PRESSURE
    return r


def stops_to_string(stops: Iterable[Stop]) -> str:
    """Compact human-readable rendering of stops, eg '3@21m 8@9m'."""
    return ' '.join(f'{round(duration):.0f}@{depth:.0f}m'
                    for depth, duration, *_ in stops if duration >= 0.1)


def stops_to_string_precise(stops: Iterable[Stop]) -> str:
    """Precise rendering of stops including gas, eg '2.7@21m[Nx50]'."""
    return ' '.join(f'{duration:.1f}@{depth:.0f}m[{gas}]'
                    for depth, duration, gas, *_ in stops)
