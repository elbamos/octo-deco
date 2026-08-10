# Please see LICENSE.md
"""Factory functions creating DiveProfiles: the demo dive, dives from user
input, and dives imported from CSV files."""
from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from typing import Any

from . import Gas
from .DiveProfile import DiveProfile


#
# Create: Demo dive
#
def create_demo_dive(gf_low: float = 35, gf_high: float = 70) -> DiveProfile:
    dp = DiveProfile(gf_low=gf_low, gf_high=gf_high)
    dp.is_demo_dive = True
    dp.add_gas(Gas.Nitrox(50))
    dp.append_section(20, 45, Gas.Trimix(21, 35))
    dp.append_section(5, 5, gas=Gas.Trimix(21, 35))
    dp.append_section(40, 35, gas=Gas.Trimix(21, 35))
    dp.add_stops_to_surface()
    dp.append_section(0, 30)
    dp.interpolate_points()
    return dp


#
# Create: Dive from user input
#
def _ipt_check_depth_time(depth: Any, time: Any) -> bool:
    """True if depth and time parse to ints within sane bounds."""
    try:
        return 0 <= int(depth) < 200 and 0 < int(time) < 200
    except (TypeError, ValueError):
        return False


def create_dive_by_depth_time_gas(dtgs: Sequence[tuple[Any, Any, Any]],
                                  extragas: str) -> DiveProfile | None:
    """Create a dive from (depth, time, gas) triples plus a comma-separated
    string of extra (deco) gases; None if no triple is valid."""
    assert len(dtgs) <= 11
    result = DiveProfile()
    # Parse the dive steps
    cntok = 0
    for depth, duration, gasname in dtgs:
        gas = Gas.from_string(gasname)
        if gas is not None and _ipt_check_depth_time(depth, duration):
            result.append_section(int(depth), int(duration), gas,
                                  correct_duration_with_transit=True)
            cntok += 1
    if cntok == 0:
        return None
    # Parse any additional gas
    for g in Gas.many_from_string(extragas):
        result.add_gas(g)
    # Add deco stops
    result.add_stops_to_surface()
    result.append_section(0, 10)
    result.interpolate_points()
    return result


#
# Create: Dive from CSV
#
class ParseError(Exception):
    """To raise & catch unexpected CSV format."""


class QNDCache:
    """Quick-and-dirty cache remembering only the most recent call.

    Calling it returns (is_new, result): is_new is False and the cached
    result is reused when the arguments equal the previous call's.
    """

    def __init__(self, fun: Callable[[tuple[Any, ...]], Any]):
        self.lastargs: tuple[Any, ...] | None = None
        self.lastres: Any = None
        self.fun = fun

    def __call__(self, *args: Any) -> tuple[bool, Any]:
        if args != self.lastargs:
            self.lastargs = args
            self.lastres = self.fun(args)
            return True, self.lastres
        return False, self.lastres


def create_from_shearwater_csv(lines: list[str]) -> DiveProfile:
    """Create a dive from a Shearwater dive computer CSV export."""
    # First two lines contain details about settings etc
    hlines = lines[0:2]
    reader = csv.DictReader(hlines)
    row = next(reader)
    try:
        gf_low = int(row['GF Minimum'])
        gf_high = int(row['GF Maximum'])
        divenr = int(row['Dive Number'])
    except KeyError as err:
        raise ParseError(f'Could not parse header row ({err.args[0]})')
    result = DiveProfile(gf_low=gf_low, gf_high=gf_high)
    result.add_custom_desc = f'SW-{divenr}'
    # Rest is actually the dive
    # (Using DictReader is probably not the most efficient, but soit)
    reader = csv.DictReader(lines[2:])
    gas = Gas.Air()
    cc = QNDCache(lambda p: Gas.Trimix(100.0 * p[0], 100.0 * p[1]))
    for row in reader:
        # Extract info
        try:
            t = float(row['Time (ms)']) / 6e4
            d = float(row['Depth'])
            fo2 = float(row['Fraction O2'])
            fhe = float(row['Fraction He'])
        except KeyError as err:
            raise ParseError(f'Could not parse dive point ({err.args[0]})')
        # Gas
        if d > 0.0:
            nw, gas = cc(fo2, fhe)
            if nw:
                result.add_gas(gas)
        # Point
        result._append_point_abstime(t, d, gas)
    # Wrap up
    result.update_deco_info()
    result.update_deco_model_info(result.deco_model(), update_profile=True)
    return result


def create_from_octodeco_csv(lines: list[str]) -> DiveProfile:
    """Create a dive from octo-deco's own CSV export (see
    DivePoint.repr_for_dataframe)."""
    result: DiveProfile | None = None
    reader = csv.DictReader(lines)
    gas = Gas.Air()
    cc = QNDCache(lambda args: Gas.from_string(args[0]))
    for row in reader:
        # Extract info
        try:
            t = float(row['time'])
            d = float(row['depth'])
            g = str(row['gas'])
            ids = int(row['IsDecoStop']) == 1
            iip = int(row['IsInterpolated']) == 1
            iap = int(row.get('IsAscent', 0)) == 1
            if result is None:
                dgflow = int(row['DiveGFLow'])
                dgfhigh = int(row['DiveGFHigh'])
        except KeyError as err:
            raise ParseError(f'Could not parse dive point ({err.args[0]})')
        # First line:
        if result is None:
            result = DiveProfile(gf_low=dgflow, gf_high=dgfhigh)
        # Gas
        if d > 0.0:
            nw, gas = cc(g)
            if nw:
                result.add_gas(gas)
        # Point
        p = result._append_point_abstime(t, d, gas)
        p.is_deco_stop = ids
        p.is_ascent_point = iap
        p.is_interpolated_point = iip
    # Wrap up
    result.add_custom_desc = 'CSV'
    result.update_deco_info()
    result.update_deco_model_info(result.deco_model(), update_profile=True)
    return result


def create_from_csv_file(filename: str,
                         func: Callable[[list[str]], DiveProfile]) -> DiveProfile:
    with open(filename, 'r') as f:
        return func(f.readlines())
