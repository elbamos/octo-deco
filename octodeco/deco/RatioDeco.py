# Please see LICENSE.md
"""Ratio decompression.

Dives to 90 m are supported (1:1, 1:2 and 1:3 ratios); deeper ratios are not yet
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

    CurveShape = Literal['exponential', 's_curve_deep', 's_curve_shallow']

    def __init__(self,
                 curve_shape: CurveShape = 's_curve_deep',
                 gas_switch_mins: float = .5,
                 last_stop_depth: Literal[3, 6] = 3,
                 first_deep_stop: Literal[75, 66] | None = 75,
                 second_deep_stop: Literal[50] | None = 50,
                 o2_break_cycle: tuple[Literal[12, 10], Literal[6, 5]] = (12, 6)):
        """The version-dependent options: first_deep_stop / second_deep_stop
        are the deep-stop depths as percentages of max depth (None = no
        such stop); curve_shape 's_curve_deep' pushes the time halved off
        the middle S-curve stops to the two deepest stops (RD 1.0),
        's_curve_shallow' to the shallowest stop (RD 2.0+); o2_break_cycle
        is (minutes on O2, minutes on backgas) for long O2 stops."""
        super().__init__()
        self.descent_speed = 20
        # In deco, every 3 m of ascent takes 30 seconds; combined with the
        # standard 30-second stops, each 3 m increment costs one minute.
        self.ascent_speed = 6
        self.max_pO2_deco = 1.6
        self.gas_switch_mins = gas_switch_mins
        self.last_stop_depth = last_stop_depth
        self.curve_shape = curve_shape
        self.first_deep_stop = first_deep_stop
        self.second_deep_stop = second_deep_stop
        self.o2_break_cycle = o2_break_cycle

    # The options making up each published version of ratio deco: 2.0
    # moved the first deep stop from 75% to 66% of depth and pushes the
    # S-curve's stolen middle time to the shallowest stop; 3.0 drops the
    # prescribed deep stops entirely. Later materials also cycle O2
    # breaks at 10 on / 5 off instead of 12 / 6.
    _VERSION_PRESETS = {
        1: dict(first_deep_stop=75, second_deep_stop=50,
                curve_shape='s_curve_deep', o2_break_cycle=(12, 6)),
        2: dict(first_deep_stop=66, second_deep_stop=50,
                curve_shape='s_curve_shallow', o2_break_cycle=(10, 5)),
        3: dict(first_deep_stop=None, second_deep_stop=None,
                curve_shape='s_curve_shallow', o2_break_cycle=(10, 5)),
    }

    @classmethod
    def for_version(cls, version: Literal[1, 2, 3],
                    curve_shape: CurveShape | None = None,
                    gas_switch_mins: float = .5,
                    last_stop_depth: Literal[3, 6] = 3) -> RatioDeco:
        """A RatioDeco parameterized for Ratio Deco 1.0, 2.0 or 3.0.
        curve_shape None means the version's own S-curve flavor."""
        if version not in cls._VERSION_PRESETS:
            raise ValueError(f'unknown ratio deco version {version!r}; expected 1, 2 or 3')
        preset = dict(cls._VERSION_PRESETS[version])
        if curve_shape is not None:
            preset['curve_shape'] = curve_shape
        return cls(gas_switch_mins=gas_switch_mins,
                   last_stop_depth=last_stop_depth,
                   **preset)

    #
    # The DecompressionModel interface
    #
    @classmethod
    def for_profile(cls, diveprofile: DiveProfile,
                    settings: dict[str, Any]) -> RatioDeco:
        # Ratio deco is a standardized procedure: it brings its own ascent
        # speeds and stop depths rather than taking them from the dive;
        # the configurable options arrive as strings from the UI or as
        # real values from settings(). Anything absent or unrecognized
        # keeps the version preset (legacy settings carry a 'version' key;
        # its curve falls back to that version's own S-curve flavor).
        curve = settings.get('curve_shape')
        if curve not in ('exponential', 's_curve_deep', 's_curve_shallow'):
            curve = None
        m = cls.for_version(settings.get('version') or 1, curve_shape=curve)
        fds_map = {75: 75, '75': 75, 66: 66, '66': 66, None: None, 'none': None}
        if settings.get('first_deep_stop', '') in fds_map:
            m.first_deep_stop = fds_map[settings['first_deep_stop']]
        sds_map = {50: 50, '50': 50, None: None, 'none': None}
        if settings.get('second_deep_stop', '') in sds_map:
            m.second_deep_stop = sds_map[settings['second_deep_stop']]
        o2_map = {(12, 6): (12, 6), '12_6': (12, 6), (10, 5): (10, 5), '10_5': (10, 5)}
        if settings.get('o2_break_cycle', '') in o2_map:
            m.o2_break_cycle = o2_map[settings['o2_break_cycle']]
        lsd_map = {3: 3, '3': 3, 6: 6, '6': 6}
        if settings.get('last_stop_depth', '') in lsd_map:
            m.last_stop_depth = lsd_map[settings['last_stop_depth']]
        return m

    def settings(self) -> dict[str, Any]:
        return {'curve_shape': self.curve_shape,
                'first_deep_stop': self.first_deep_stop,
                'second_deep_stop': self.second_deep_stop,
                'o2_break_cycle': self.o2_break_cycle,
                'last_stop_depth': self.last_stop_depth}

    def _matching_version(self) -> int | None:
        """The published version these options correspond to, if any
        (the curve shape is a free choice and does not count)."""
        for v, preset in self._VERSION_PRESETS.items():
            if (self.first_deep_stop == preset['first_deep_stop']
                    and self.second_deep_stop == preset['second_deep_stop']
                    and self.o2_break_cycle == preset['o2_break_cycle']):
                return v
        return None

    def description(self) -> str:
        v = self._matching_version()
        label = f'{v}.0' if v is not None else 'custom'
        return f'Ratio deco {label} ({self.curve_shape})'

    def NDL(self, point: DivePoint, state: Any = None) -> float:
        return max(0, self._NDL(point, state))

    def _NDL(self, point: DivePoint, state: Any = None) -> float:
        """No-decompression limit as of this point: how much longer (minutes)
        the bottom phase can last and still allow a min-deco ascent."""

        # The NDL belongs to the bottom phase, so it is computed from the
        # bottom gas: the gas breathed at the deepest point so far. The
        # point itself may already be past the bottom on another gas (eg
        # air back at the surface), which would flip the gas multiplier.
        bottom_points = point._points_before_deco_to_here()
        bottom_gas = max(bottom_points, key=lambda p: p.depth).gas \
            if bottom_points else point.gas

        #  Are we using Air, Nx32, Tmx30/30, or something else?
        if bottom_gas == Gas.Air():
            gas_multiplier = 1
        elif bottom_gas == Gas.Nitrox(32):
            gas_multiplier = .8
        elif bottom_gas == Gas.Trimix(30, 30):
            gas_multiplier = .8
        else:
            return 0

        avg_bottom_depth_meters = point.avg_depth_bottom()
        bottom_time = point.bottomtime()
        effective_avg_depth_meters = avg_bottom_depth_meters * gas_multiplier

        # The 5thD-X/UTD min-deco air table, entered with the effective
        # average depth rounded up (conservative) to the next 3 m row.
        # From 21 m down it is the documented rule "100'/30m = 20 min,
        # +/-5 min per 10'/3m"; the three shallow rows are the table's
        # own departures from that line.
        row = 3 * math.ceil(effective_avg_depth_meters / 3)
        if row > 39:
            return 0
        if row >= 21:
            limit = 20 - (row - 30) * 5 / 3
        else:
            limit = {18: 50, 15: 60}.get(row, 170)
        return limit - bottom_time



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

    def _insert_backgas_breaks(self, stops: List[Stop], back_gas: Gas.Gas) -> List[Stop]:
        """The 1:2/1:3 gas switch protocol: before each switch to the next
        deco gas, the diver switches back to the bottom gas for the last
        gas_switch_mins of the current stop (included in the stop's time),
        rides it through the ascent, and picks up the next gas on arrival
        at its own stop. Coming off the bottom the diver is already on
        backgas, so the first switch needs no break."""
        result: List[Stop] = []
        for i, s in enumerate(stops):
            nxt = stops[i + 1] if i + 1 < len(stops) else None
            if nxt is not None and nxt.gas != s.gas and s.gas != back_gas \
                    and s.duration > 0:
                on_gas = s.duration - self.gas_switch_mins
                if on_gas > 0:
                    result.append(Stop(s.depth, on_gas, s.gas, s.ascent_speed))
                result.append(Stop(s.depth, min(s.duration, self.gas_switch_mins),
                                   back_gas, s.ascent_speed))
            else:
                result.append(s)
        return result

    def _apply_O2_breaks(self, stops: List[Stop], back_gas: Gas.Gas) -> List[Stop]:
        """Oxygen-break cycling: when the continuous time on O2 exceeds 20
        minutes it is taken as cycles of o2_break_cycle = (minutes on O2,
        minutes on backgas), the breaks counting toward the stop times.
        The O2 clock runs across consecutive O2 stops (eg a 6 m and a 3 m
        stop back to back), not per stop; a run of up to 20 minutes needs
        no break. RD 1.0 cycles 12/6, later versions 10/5."""
        o2 = Gas.Nitrox(99)
        on_max, off_max = self.o2_break_cycle
        result: List[Stop] = []
        i = 0
        while i < len(stops):
            if stops[i].gas != o2:
                result.append(stops[i])
                i += 1
                continue
            # The run of consecutive O2 stops starting here
            j = i
            while j + 1 < len(stops) and stops[j + 1].gas == o2:
                j += 1
            run = stops[i:j + 1]
            if sum(s.duration for s in run) <= 20:
                result.extend(run)
            else:
                on_left = on_max
                for s in run:
                    remaining = s.duration
                    while remaining > 0:
                        if on_left <= 0:
                            off = min(off_max, remaining)
                            result.append(Stop(s.depth, off, back_gas, s.ascent_speed))
                            remaining -= off
                            on_left = on_max
                        else:
                            on = min(on_left, remaining)
                            result.append(Stop(s.depth, on, o2, s.ascent_speed))
                            remaining -= on
                            on_left -= on
            i = j + 1
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

        def stop_gas(depth: float) -> Gas.Gas:
            """The gas breathed at a stop at this depth. The standard-gas
            switch depths are imperial (120'/70'/20'), and their metric
            stops round a hair shallow - 35/25 at the 36 m stop runs pO2
            1.614 - so allow that smidgen over the limit; the switches
            belong at the doctrinal depths."""
            return Gas.best_gas(gases, Util.depth_to_Pamb(depth),
                                self.max_pO2_deco + 0.02)

        def generate_min_stops() -> List[Stop]:
            min_stops = []

            stop_depth = 3 * math.ceil(point.max_depth() / 2 / 3)
            while stop_depth >= self.last_stop_depth:
                gas_to_use = stop_gas(stop_depth)
                ascent_speed = self.ascent_speed if stop_depth > 6 else self.ascent_speed / 2

                new_stop = Stop(stop_depth, 0.5, gas_to_use, ascent_speed)
                min_stops.append(new_stop)

                stop_depth -= 3

            return min_stops

        def generate_s_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            # S-curve over any number of 3 m stops: start from the linear
            # split and halve (rounding up) every stop between the two
            # deepest and the shallowest. Where the taken time goes is
            # version-dependent: RD 1.0 pushes it to the two deepest stops
            # (matching the worked 5-stop examples in the 2005/2008
            # material, eg 15 min -> 4/4/2/2/3), RD 2.0+ to the shallowest
            # stop (15 min -> 3/3/2/2/5). The last stop absorbs the
            # rounding so the segment total stays exact.
            depths = list(range(start_depth, end_depth - 1, -3))
            n = len(depths)
            base = math.ceil(duration / n)
            if n < 3:
                durations = [base] * (n - 1) + [max(1, duration - base * (n - 1))]
            else:
                mid = math.ceil(base / 2)
                n_mid = n - 3
                taken = n_mid * (base - mid)
                if self.curve_shape == 's_curve_deep':
                    deeps = [base + (taken - taken // 2), base + taken // 2]
                else:
                    deeps = [base, base]
                durations = deeps + [mid] * n_mid \
                    + [max(1, duration - sum(deeps) - mid * n_mid)]

            return [Stop(depth, d, stop_gas(depth), self.ascent_speed)
                    for depth, d in zip(depths, durations)]

        def generate_expo_curve(start_depth: int, end_depth: int, duration: int) -> List[Stop]:
            # RD 1.0 exponential shape over any number of 3 m stops: each
            # stop no shorter than the one before, built by halving time
            # off the two deepest stops and moving it to the two
            # shallowest (eg 15 min over 5 stops -> 1/2/3/4/5).
            depths = list(range(start_depth, end_depth - 1, -3))
            n = len(depths)
            base = math.ceil(duration / n)
            if n < 4:
                durations = [base] * (n - 1) + [max(1, duration - base * (n - 1))]
            else:
                second = math.ceil(base / 2)
                deepest = math.ceil(second / 2)
                taken = (base - deepest) + (base - second)
                durations = [deepest, second] + [base] * (n - 4) \
                    + [base + math.floor(taken / 2), base + math.ceil(taken / 2)]

            return [Stop(depth, d, stop_gas(depth), self.ascent_speed)
                    for depth, d in zip(depths, durations)]

        if self.curve_shape == 'exponential':
            generate_curve = generate_expo_curve
        else:
            generate_curve = generate_s_curve

        def generate_final_stops(duration: int) -> List[Stop]:
            if self.last_stop_depth < 6:
                return [
                    Stop(6, duration // 2, stop_gas(6), self.ascent_speed / 2),
                    Stop(3, duration // 2, stop_gas(3), self.ascent_speed / 2)
                ]
            else:
                return [
                    Stop(6, duration, stop_gas(6), self.ascent_speed / 2)
                ]

        def generate_deep_stops(next_segment_depth_m: int) -> List[Stop]:
            # Deep stops at the version's depth fractions (None = version
            # prescribes none). Durations from the 5thD-X deep-stop table,
            # keyed to exposure past the NDL (~ bottom time in the ratio
            # zones, where the table NDL is 0-5 min): per 30-minute block
            # of exposure, the first stop grows 1 minute (1..5) and the
            # second 2 minutes (1, 3, 5, 7, 9, capped at 10).
            stops = []
            if self.first_deep_stop is None:
                return stops
            blocks = math.floor(point.bottomtime() / 30)
            first_stop = 3 * math.floor(point.max_depth() * self.first_deep_stop / 100 / 3 + 0.5)
            if first_stop > next_segment_depth_m:
                stops.append(Stop(first_stop, min(5, 1 + blocks),
                                  point.gas, self.ascent_speed))
                if self.second_deep_stop is not None:
                    second_stop = 3 * math.floor(point.max_depth() * self.second_deep_stop / 100 / 3 + 0.5)
                    if second_stop > next_segment_depth_m:
                        stops.append(Stop(second_stop, min(10, 1 + 2 * blocks),
                                          point.gas, self.ascent_speed))
            return stops

        def lost_gas_factor(depth: float) -> int:
            """RD 1.0 lost-gas rule: a deco segment whose deco gas is not
            carried is done on backgas for twice the time. (The gas
            consumption analysis relies on this: its lost-gas scenarios
            replan the dive with a deco gas removed.)"""
            return 2 if stop_gas(depth) == bottom_gas else 1

        def calculate_ratio_deco_time(ratio: Literal[1, 2, 3], set_point: int, avg_depth: float, bottom_time: float) -> float:
            deco_time = bottom_time * ratio
            diff = avg_depth - set_point
            n = math.ceil(abs(diff) / 3)
            intervals = n if diff >= 0 else -n
            deco_time += intervals * 5
            return deco_time

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
            has_O2 = stop_gas(3) == Gas.Nitrox(99)
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

            deco_time = calculate_ratio_deco_time(1, 45, point.avg_depth_bottom(), point.bottomtime())

            time_21_09 = math.ceil(deco_time / 2) * lost_gas_factor(21)
            time_06_03 = math.ceil(deco_time / 2) * lost_gas_factor(6)
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

            deco_time = calculate_ratio_deco_time(2, 66, point.avg_depth_bottom(), point.bottomtime())

            # Half the ratio time at 21-9 m on Nitrox 50, half on O2 in
            # the final stops (all at 6 m per the source, or split 6/3
            # when the last stop is 3 m); on top of that, half the Nx50
            # segment's time again in the 36-24 m range ("Do 1/2 of
            # nitrox 50 time in 120'/36m - 80'/24m range"). Deep stops
            # only apply above the 36 m segment, which supersedes the
            # 50%-depth stop.
            time_36_24 = math.ceil(deco_time / 4)
            time_21_09 = math.ceil(deco_time / 2) * lost_gas_factor(21)
            time_06_03 = math.ceil(deco_time / 2) * lost_gas_factor(6)
            stops = self._insert_backgas_breaks(
                generate_deep_stops(36) + generate_curve(36, 24, time_36_24)
                + generate_curve(21, 9, time_21_09)
                + generate_final_stops(time_06_03), bottom_gas)
            stops = self._apply_O2_breaks(stops, bottom_gas)
            return (stops, Util.depth_to_Pamb(stops[0].depth), commit(stops))
        elif point.max_depth() <= 90:
            # Deco at 1:3 ratio
            if not (bottom_gas == Gas.Trimix(15, 55) or bottom_gas == Gas.Trimix(10, 70)):
                raise ValueError(f'ratio deco between 72 m and 90 m requires '
                                 f'bottom gas Tx15/55 or Tx10/70; this dive uses {bottom_gas}')

            deco_time = calculate_ratio_deco_time(3, 81, point.avg_depth_bottom(), point.bottomtime())

            # 40% of ratio time on oxygen, 40% on Nitrox 50, and 20% on 35/25
            time_57_39 = math.ceil(deco_time * .1)
            time_36_24 = math.ceil(deco_time * .2) * lost_gas_factor(36)
            time_21_09 = math.ceil(deco_time * .4) * lost_gas_factor(21)
            time_06_03 = math.ceil(deco_time * .4) * lost_gas_factor(6)
            stops = self._insert_backgas_breaks(
                generate_deep_stops(57) + generate_curve(57, 39, time_57_39)
                + generate_curve(36, 24, time_36_24)
                + generate_curve(21, 9, time_21_09)
                + generate_final_stops(time_06_03), bottom_gas)
            stops = self._apply_O2_breaks(stops, bottom_gas)
            return (stops, Util.depth_to_Pamb(stops[0].depth), commit(stops))
        else:
            raise NotImplementedError('ratio deco is not implemented for dives beyond 90 m')

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
