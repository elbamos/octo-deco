# Please see LICENSE.md
"""Ratio decompression.

Dives to 72 m are supported (1:1 and 1:2 ratios); deeper ratios are not yet
implemented.

RatioDeco implements the DecompressionModel interface, so a dive switches to
ratio deco simply by setting its model type:

    dp = DiveProfile()
    dp._deco_model_type = RatioDeco.MODEL_TYPE
    ...
    dp.add_stops_to_surface()

Tissue tracking (inherited from DecompressionModel) keeps running while the
ratio deco profile is followed, so tissue saturation, GF99/SurfaceGF and
integral supersaturation remain available at every point for visualization
and comparison against Bühlmann.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal

from . import Gas, Util
from .DecompressionModel import DecompressionModel
from .Util import Stop

if TYPE_CHECKING:
    from .DivePoint import DivePoint
    from .DiveProfile import DiveProfile


@dataclass
class RatioDecoState:
    """Continuation state for RatioDeco: the schedule committed at the
    moment the ascent began.

    Ratio deco plans from the bottom phase's average depth and time. While
    the diver is still on the bottom, the state stays None and the plan
    updates live as bottom time accrues; once the ascent begins, the plan
    is committed here and only its remaining portion is reported from then
    on — recomputing mid-ascent would let the plan drift.
    """
    stops: List[Stop]


class RatioDeco(DecompressionModel):
    MODEL_TYPE = 'RatioDeco'

    # Min-deco air NDL by average depth (m), from the 5thD-X 2005 outline.
    _NDL_TABLE = {12: 170, 15: 60, 18: 50, 21: 35, 24: 30, 27: 25,
                  30: 20, 33: 15, 36: 10, 39: 5}

    def __init__(self,
                 descent_speed: float = 6,
                 curve_shape: Literal['s-curve', 'exponential'] = 's-curve',
                 max_pO2_deco: float = 1.6, gas_switch_mins: float = .5,
                 last_stop_depth: float = 3):
        super().__init__()
        self.descent_speed = descent_speed
        # In deco, every 3 m of ascent takes 30 seconds; combined with the
        # standard 30-second stops, each 3 m increment costs one minute.
        self.ascent_speed = 6
        self.max_pO2_deco = max_pO2_deco
        self.gas_switch_mins = gas_switch_mins
        self.last_stop_depth = last_stop_depth
        self.curve_shape = curve_shape
        # TODO: ratio deco configuration (eg the ratio, reference depth,
        # ascent shape parameters). Whatever is added here should also be
        # reflected in settings() / for_profile() so it round-trips.

    #
    # The DecompressionModel interface
    #
    @classmethod
    def for_profile(cls, diveprofile: DiveProfile,
                    settings: dict[str, Any]) -> RatioDeco:
        # Ratio deco is a standardized procedure: it brings its own ascent
        # speeds and stop depths rather than taking them from the dive, so
        # only the curve shape is configurable.
        return cls(curve_shape=settings.get('curve_shape', 's-curve'))

    def settings(self) -> dict[str, Any]:
        return {'curve_shape': self.curve_shape}

    def description(self) -> str:
        return f'Ratio deco ({self.curve_shape})'

    def NDL(self, point: DivePoint, state: Any = None) -> float:
        return max(0, self._NDL(point, state))

    def _NDL(self, point: DivePoint, state: Any = None) -> float:
        """No-decompression limit at this point: how long (minutes) the diver
        can stay at this depth, on this gas, and still ascend directly to the
        surface."""

        #  Are we using Air, Nx32, Tmx30/30, or something else?
        if point.gas == Gas.Air():
            gas_multiplier = 1
        elif point.gas == Gas.Nitrox(32):
            gas_multiplier = .8
        elif point.gas == Gas.Trimix(30, 30):
            gas_multiplier = .8
        else:
            return 0

        avg_bottom_depth_meters = point.avg_depth_bottom()
        bottom_time = point.bottomtime()
        effective_avg_depth_meters = avg_bottom_depth_meters * gas_multiplier

        # The 5thD-X/UTD min-deco air table, entered with the effective
        # average depth rounded up (conservative) to the next 3 m row.
        # The 21-39 m rows follow the "100'/30m = 20 min, +/-5 min per
        # 10'/3m" line; the shallower rows are the table's own departures
        # from that line.
        row = max(12, 3 * math.ceil(effective_avg_depth_meters / 3))
        if row > 39:
            return 0
        return self._NDL_TABLE[row] - bottom_time



    def stop_needed(self, point: DivePoint, state: Any = None) -> bool:
        """Whether deco stops should have been made before reaching this
        point: the diver is shallower than the first scheduled stop, and the
        profile does not yet contain any deco stops.

        The second condition is what makes DiveProfile.add_stops terminate:
        the ratio deco schedule is fixed by the bottom phase (unlike
        Bühlmann, where stops off-gas the tissues and thereby clear the
        violation), so once the schedule's stops are in the profile it has
        been executed and no further stops may be inserted.
        """
        p: DivePoint | None = point
        while p is not None:
            if p.is_deco_stop:
                return False
            p = p.prev
        # Gas choice affects only which gas is breathed at the stops, not
        # their depths, so the point's own gas suffices here.
        stops, _, _ = self.compute_deco_profile(point, [point.gas], state=state)
        if len(stops) == 0:
            return False
        return point.depth < stops[0].depth - 0.01

    def _insert_gas_switches(self, stops: List[Stop], from_gas: Gas.Gas) -> List[Stop]:
        """Insert gas-switch stops into a schedule: wherever the gas changes,
        first hold at the switch depth on the old gas for gas_switch_mins;
        the new gas's planned stop (or the ascent, if it has no duration)
        follows."""
        result: List[Stop] = []
        current_gas = from_gas
        for s in stops:
            if s.gas != current_gas:
                result.append(Stop(s.depth, self.gas_switch_mins, current_gas))
                current_gas = s.gas
            result.append(s)
        return result

    def _apply_O2_breaks(self, stops: List[Stop], back_gas: Gas.Gas) -> List[Stop]:
        """RD 1.0 oxygen-break cycling: an O2 stop longer than 20 minutes
        is taken as cycles of 12 minutes on O2 and 6 minutes on backgas;
        the breaks count toward the stop time (a plain 15-20 minute O2
        stop needs no break)."""
        o2 = Gas.Nitrox(99)
        result: List[Stop] = []
        for s in stops:
            if s.gas != o2 or s.duration <= 20:
                result.append(s)
                continue
            remaining = s.duration
            while remaining > 0:
                on = min(12, remaining)
                result.append(Stop(s.depth, on, o2, s.ascent_speed))
                remaining -= on
                if remaining > 0:
                    off = min(6, remaining)
                    result.append(Stop(s.depth, off, back_gas, s.ascent_speed))
                    remaining -= off
        return result

    def compute_deco_profile(self, point: DivePoint, gases: Iterable[Gas.Gas],
                             p_target: float = Util.SURFACE_PRESSURE,
                             add_gas_switch_time: bool = False,
                             state: Any = None) -> tuple[list[Stop], float, Any]:
        """The ratio deco schedule for ascending from this point to p_target.

        This is where the actual ratio deco rules go. The dive so far is
        reachable through the point (point.time, point.depth, and the chain
        of previous points via point.prev — eg for average depth or bottom
        time). Returns (stops, p_ceiling, state): (depth, duration, gas)
        triples, the ceiling as an ambient pressure (may be a nominal value
        if ratio deco has no ceiling concept), and the continuation state
        (None is fine if no state needs to carry over between calls).
        """

        # Once the ascent has begun the plan is fixed: report what remains
        # of the committed schedule at this depth instead of recomputing.
        if isinstance(state, RatioDecoState):
            remaining = [s for s in state.stops if s.depth <= point.depth + 0.01]
            p_ceiling = Util.depth_to_Pamb(remaining[0].depth) if remaining \
                else Util.SURFACE_PRESSURE
            return (remaining, p_ceiling, state)

        if p_target != Util.SURFACE_PRESSURE:
            # A mid-dive ascent target (eg between levels of a multilevel
            # dive): ratio deco only plans the final ascent to the surface,
            # so no stops are required here.
            return ([], p_target, state)

        # deco_info calls this for every point, including the pre-descent
        # surface points: nothing to decompress yet.
        if point.max_depth() == 0:
            return ([], Util.SURFACE_PRESSURE, state)

        # The gas the bottom phase was dived on: the gas breathed at the
        # deepest point (this point itself may already be breathing a deco
        # gas, or air back at the surface).
        bottom_points = point._points_before_deco_to_here()
        bottom_gas = max(bottom_points, key=lambda p: p.depth).gas \
            if bottom_points else point.gas

        def commit(stops: List[Stop]) -> Any:
            """The continuation state to return alongside a freshly computed
            schedule: commit it once the diver has left the bottom (the
            chain-before-deco no longer ends at this point); while still on
            the bottom, stay uncommitted so the plan keeps updating live."""
            ascent_begun = len(bottom_points) == 0 or bottom_points[-1] is not point
            return RatioDecoState(stops) if ascent_begun else state

        def generate_min_stops() -> List[Stop]:
            min_stops = []

            stop_depth = 3 * math.ceil(point.max_depth() / 2 / 3)
            while stop_depth >= self.last_stop_depth:
                gas_to_use = Gas.best_gas(gases, Util.depth_to_Pamb(stop_depth), self.max_pO2_deco)
                ascent_speed = self.ascent_speed if stop_depth > 6 else self.ascent_speed / 2

                new_stop = Stop(stop_depth, 0.5, gas_to_use, ascent_speed)
                min_stops.append(new_stop)

                stop_depth -= 3

            return min_stops

        def generate_s_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            # RD 1.0 S-curve: start from the linear split, halve the two
            # middle stops (rounding up), push the time taken from them to
            # the two deepest stops, and let the shallowest stop absorb the
            # rounding so the segment total stays exact. Matches the worked
            # examples in the 2005/2008 source material:
            # 15 min -> 4/4/2/2/3, 24 min -> 7/7/3/3/4.
            base = math.ceil(duration / 5)
            deep = base + math.floor(base / 2)
            mid = math.ceil(base / 2)
            shallow = max(1, duration - 2 * deep - 2 * mid)
            stops = [deep, deep, mid, mid, shallow]

            return [Stop(depth, duration, Gas.best_gas(gases, Util.depth_to_Pamb(depth), self.max_pO2_deco), self.ascent_speed)
                    for depth, duration in zip(list(range(start_depth, end_depth - 1, -3)), stops)]

        def generate_expo_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            # RD 1.0 exponential shape: each stop longer than the one
            # before, built by halving time off the two deepest stops and
            # moving it to the shallow end (eg 15 min -> 1/2/3/4/5).
            base = math.ceil(duration / 5)
            second = math.ceil(base / 2)
            deepest = math.ceil(second / 2)
            taken = (base - deepest) + (base - second)
            stops = [deepest, second, base,
                     base + math.floor(taken / 2),
                     base + math.ceil(taken / 2)]

            return [Stop(depth, duration, Gas.best_gas(gases, Util.depth_to_Pamb(depth), self.max_pO2_deco), self.ascent_speed)
                    for depth, duration in zip(list(range(start_depth, end_depth - 1, -3)), stops)]

        if self.curve_shape == "s-curve":
            generate_curve = generate_s_curve
        else:
            generate_curve = generate_expo_curve

        def generate_final_stops(duration: int) -> List[Stop]:
            return [
                Stop(6, duration // 2, Gas.best_gas(gases, Util.depth_to_Pamb(6), self.max_pO2_deco), self.ascent_speed / 2),
                Stop(3, duration // 2, Gas.best_gas(gases, Util.depth_to_Pamb(3), self.max_pO2_deco), self.ascent_speed / 2)
            ]

        def generate_deep_stops(gas_switch_depth_m: int) -> List[Stop]:
            # Durations from the 5thD-X deep-stop table, keyed to exposure
            # past the NDL (~ bottom time in the ratio zones, where the
            # table NDL is 0-5 min): 75% stops run 1..5 min and 50% stops
            # 1..10 min, with a 1-minute minimum at each.
            past_ndl = point.bottomtime()

            def duration_75() -> float:
                if past_ndl < 30:
                    return 1
                if past_ndl < 60:
                    return 2
                if past_ndl < 120:
                    return 3
                if past_ndl < 150:
                    return 4
                return 5

            def duration_50() -> float:
                if past_ndl < 30:
                    return 1
                if past_ndl < 60:
                    return 3
                if past_ndl < 90:
                    return 5
                if past_ndl < 120:
                    return 7
                if past_ndl < 150:
                    return 9
                return 10

            stops = []
            first_stop = 3 * math.floor(point.max_depth() * 0.75 / 3 + 0.5)
            if first_stop > gas_switch_depth_m:
                stops.append(Stop(first_stop, duration_75(), point.gas, self.ascent_speed))
                second_stop = 3 * math.floor(point.max_depth() * 0.5 / 3 + 0.5)
                if second_stop > gas_switch_depth_m:
                    stops.append(Stop(second_stop, duration_50(), point.gas, self.ascent_speed))
            return stops

        ndl = self._NDL(point, state=state)
        if ndl > 0:
            # If we're under the NDL, perform min deco
            min_stops = self._insert_gas_switches(generate_min_stops(), bottom_gas)
            p_ceiling = Util.depth_to_Pamb(min_stops[0].depth) if min_stops \
                else Util.SURFACE_PRESSURE
            return (min_stops, p_ceiling, commit(min_stops))
        elif point.max_depth() <= 30:
            # If we have to decompress but we're under 100',
            # we add to the final stop an amount of time equal
            # to how far over the NDL we are.
            # This tool is only endorsed up to 20 minutes past the NDL
            # ("Only add the Normal Min Deco ascent times when exceeding
            # the N.D.L. by 20 min or less").
            if -ndl > 20:
                raise ValueError(f'ratio deco extended deco covers at most 20 '
                                 f'minutes past the NDL; this dive is '
                                 f'{-ndl:.0f} minutes over')
            deco_time_to_distribue = - math.floor(ndl)
            stops = generate_min_stops()
            has_O2 = Gas.best_gas(gases, Util.depth_to_Pamb(3), 1.6) == Gas.Nitrox(99)
            if has_O2:
                deco_time_to_distribue = math.ceil(deco_time_to_distribue / 2)
            if len(stops) == 0:
                # Too shallow for even a min deco stop
                return ([], Util.SURFACE_PRESSURE, commit([]))
            stops[-1].duration += deco_time_to_distribue
            stops = self._insert_gas_switches(stops, bottom_gas)
            return (stops, Util.depth_to_Pamb(stops[0].depth), commit(stops))
        elif point.max_depth() <= 51:
            # Deco at 1:1 ratio (zone gases: 21/35 or 18/45)
            if not (bottom_gas == Gas.Trimix(21, 35) or bottom_gas == Gas.Trimix(18, 45)):
                raise ValueError(f'ratio deco between 30 m and 51 m requires '
                                 f'bottom gas Tx21/35 or Tx18/45; this dive uses {bottom_gas}')

            deco_time = point.bottomtime()
            avg_depth_m = point.avg_depth_bottom()
            diff = avg_depth_m - 45
            n = math.ceil(abs(diff) / 3)
            intervals = n if diff >= 0  else -n
            deco_time += intervals * 5
            time_21_09 = math.ceil(deco_time / 2)
            time_06_03 = math.ceil(deco_time / 2)
            stops = self._insert_gas_switches(
                generate_deep_stops(21) + generate_curve(21, 9, time_21_09)
                + generate_final_stops(time_06_03), bottom_gas)
            stops = self._apply_O2_breaks(stops, bottom_gas)
            return (stops, Util.depth_to_Pamb(21), commit(stops))
        elif point.max_depth() <= 72:
            # Deco at 1:2 ratio
            if not (bottom_gas == Gas.Trimix(18, 45) or bottom_gas == Gas.Trimix(15, 55)):
                raise ValueError(f'ratio deco between 51 m and 72 m requires '
                                 f'bottom gas Tx18/45 or Tx15/55; this dive uses {bottom_gas}')

            deco_time = point.bottomtime() * 2
            avg_depth_m = point.avg_depth_bottom()
            diff = avg_depth_m - 66
            n = math.ceil(abs(diff) / 3)
            intervals = n if diff >= 0 else -n
            deco_time += intervals * 5
            # Half the ratio time at 21-9 m on Nitrox 50, half on O2 at
            # 6 m; on top of that, half the Nx50 segment's time again in
            # the 36-24 m range ("Do 1/2 of nitrox 50 time in 120'/36m -
            # 80'/24m range"). Deep stops only apply above the 36 m
            # segment, which supersedes the 50%-depth stop.
            time_21_09 = math.ceil(deco_time / 2)
            time_o2 = math.ceil(deco_time / 2)
            time_36_24 = math.ceil(time_21_09 / 2)
            o2_stop = [Stop(6, time_o2,
                            Gas.best_gas(gases, Util.depth_to_Pamb(6), self.max_pO2_deco),
                            self.ascent_speed)]
            stops = self._insert_gas_switches(
                generate_deep_stops(36) + generate_curve(36, 24, time_36_24)
                + generate_curve(21, 9, time_21_09) + o2_stop, bottom_gas)
            stops = self._apply_O2_breaks(stops, bottom_gas)
            return (stops, Util.depth_to_Pamb(stops[0].depth), commit(stops))
        else:
            raise NotImplementedError('ratio deco is not implemented for dives beyond 72 m')


    def deco_info(self, point: DivePoint, gases_carried: Iterable[Gas.Gas],
                  state: Any = None) -> dict[str, Any]:
        # Generic assembly: tissue metrics from the shared tracking, the rest
        # from compute_deco_profile / NDL. Works as-is once those are
        # implemented; refine if ratio deco wants to report more.
        result = self.tissue_state_info(point.tissue_state, point.p_amb)
        stops, p_ceiling, state = self.compute_deco_profile(point, gases_carried,
                                                            state=state)
        nontrivialstops = [s for s in stops if s.duration >= .1]
        result['Ceil'] = Util.Pamb_to_depth(p_ceiling)
        result['Stops'] = stops
        result['FirstStop'] = nontrivialstops[0].depth if len(nontrivialstops) > 0 else 0
        result['TTS'] = point.depth / self.ascent_speed + sum(s.duration for s in stops)
        result['NDL'] = self.NDL(point, state=state)
        result['model_state'] = state
        return result
