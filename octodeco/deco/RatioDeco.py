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

        if effective_avg_depth_meters < 24:
            return 110 - (effective_avg_depth_meters * 3) - bottom_time
        elif effective_avg_depth_meters < 30:
            return 20 - bottom_time
        elif effective_avg_depth_meters < 33:
            return 15 - bottom_time
        elif effective_avg_depth_meters < 36:
            return 10 - bottom_time
        elif effective_avg_depth_meters < 39:
            return 5 - bottom_time
        return 0



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

        def add_stop(stops: List[Stop], depth: float, duration: float):
            new_stop = Stop(depth, duration, Gas.best_gas(gases, Util.depth_to_Pamb(depth), self.max_pO2_deco), self.ascent_speed)
            stops.append(new_stop)

        def generate_min_stops() -> List[Stop]:
            min_stops = []

            stop_depth = 3 * math.ceil(point.max_depth() / 2 / 3)
            while stop_depth >= self.last_stop_depth:
                add_stop(min_stops, stop_depth, 0.5)
                stop_depth -= 3

            return min_stops

        def generate_s_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            stops = [math.ceil(duration / 5)] * 5
            time_to_distribute = stops[-2]
            stops[-2] = stops[-3] = math.ceil(stops[-2] / 2)
            stops[0] = stops[1] = math.ceil(stops[0] + (time_to_distribute / 2))

            return [Stop(depth, duration, Gas.best_gas(gases, Util.depth_to_Pamb(depth), self.max_pO2_deco), self.ascent_speed)
                    for depth, duration in zip(list(range(start_depth, end_depth - 1, -3)), stops)]

        def generate_expo_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            stops = [math.ceil(duration / 5)] * 5
            old_stop_1 = stops[1]
            stops[1] = math.ceil(stops[1] / 2)
            time_to_distribute = stops[1] - old_stop_1
            old_stop_0 = stops[0]
            stops[0] = math.ceil(stops[1] / 2)
            time_to_distribute += stops[0] - old_stop_0
            stops[3] = math.ceil(stops[3] + time_to_distribute / 2)
            stops[4] = math.ceil(stops[4] + time_to_distribute / 2)

            return [Stop(depth, duration, Gas.best_gas(gases, Util.depth_to_Pamb(depth), self.max_pO2_deco), self.ascent_speed)
                    for depth, duration in zip(list(range(start_depth, end_depth - 1, -3)), stops)]

        if self.curve_shape == "s-curve":
            generate_curve = generate_s_curve
        else:
            generate_curve = generate_expo_curve

        def generate_final_stops(duration: int) -> List[Stop]:
            # TODO: What's the right ratio for 20 and 10' stops.
            return [
                Stop(6, duration // 2, Gas.best_gas(gases, Util.depth_to_Pamb(6), self.max_pO2_deco), self.ascent_speed),
                Stop(3, duration // 2, Gas.best_gas(gases, Util.depth_to_Pamb(3), self.max_pO2_deco), self.ascent_speed)
            ]

        def generate_deep_stops(gas_switch_depth_m: int) -> List[Stop]:
            duration = point.bottomtime()
            stops = []
            first_stop = 3 * math.floor(point.max_depth() * 0.75 / 3 + 0.5)
            if first_stop > gas_switch_depth_m:
                stop_duration = math.floor(duration / 30)
                if stop_duration > 0:
                    stops.append(Stop(first_stop, stop_duration, point.gas, self.ascent_speed))
                second_stop = 3 * math.floor(point.max_depth() * 0.5 / 3 + 0.5)
                if second_stop > gas_switch_depth_m:
                    stop_duration = 1 + (2 * math.floor(duration / 30))
                    stops.append(Stop(second_stop, stop_duration, point.gas, self.ascent_speed))
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
            # to how far over the NDL we are
            deco_time_to_distribue = - math.ceil(ndl)
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
            # Deco at 1:1 ratio
            if not (bottom_gas == Gas.Trimix(30, 30) or bottom_gas == Gas.Trimix(18, 45)):
                raise ValueError(f'ratio deco between 30 m and 51 m requires '
                                 f'bottom gas Tx30/30 or Tx18/45; this dive uses {bottom_gas}')

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
            time_21_09 = math.ceil(deco_time / 2)
            time_06_03 = math.ceil(deco_time / 2)
            stops = self._insert_gas_switches(
                generate_deep_stops(21) + generate_curve(21, 9, time_21_09)
                + generate_final_stops(time_06_03), bottom_gas)
            return (stops, Util.depth_to_Pamb(21), commit(stops))
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
