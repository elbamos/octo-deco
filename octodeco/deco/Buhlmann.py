# Please see LICENSE.md
"""The Bühlmann ZHL-16 decompression model with gradient factors.

Buhlmann implements the DecompressionModel interface; the heavy lifting is
driven through TissueStateCython. The model's continuation state (see
DecompressionModel) is an AmbientToGF: the gradient factor line, fixed once
an ascent is under way.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from . import Gas, Util
from .DecompressionModel import DecompressionModel
from .Util import Stop

if TYPE_CHECKING:
    from . import TissueStateCython
    from .DivePoint import DivePoint


class AmbientToGF:
    """The gradient factor line: allowed supersaturation as a function of
    ambient pressure.

    At the first stop, allowed supersaturation is gf_low % (eg 35%); at
    surfacing it is gf_high % (eg 70%). In between we interpolate linearly;
    outside those bounds the line is flat.

    Actually, for smoothness, we use p_ceiling instead of p_first_stop.
    """

    def __init__(self, p_first_stop: float, p_target: float,
                 gf_low: float, gf_high: float):
        self.p_surface = Util.SURFACE_PRESSURE
        self.p_first_stop = max(p_first_stop, self.p_surface)
        self.gf_low = gf_low
        self.gf_high = gf_high
        self.p_target = p_target


class Buhlmann(DecompressionModel):
    """All essential logic for the Bühlmann deco model."""

    def __init__(self,
                 gf_low: float, gf_high: float,
                 descent_speed: float, ascent_speed: float,
                 max_pO2_deco: float, gas_switch_mins: float,
                 last_stop_depth: float):
        super().__init__()
        self.gf_low = gf_low
        self.gf_high = gf_high
        self.max_pO2_deco = max_pO2_deco
        self.descent_speed = descent_speed
        self.ascent_speed = ascent_speed
        self.gas_switch_mins = gas_switch_mins
        self.last_stop_depth = last_stop_depth
        self.stop_length_precision = 0.08
        # After 8 halftimes, residual is less than 0.5%, so that's infinite enough.
        self._stop_length_infinity = 8 * self._constants.N2_HALFTIMES[-1]

    def set_gf(self, gf_low: float, gf_high: float) -> None:
        self.gf_low = gf_low
        self.gf_high = gf_high

    #
    # The DecompressionModel interface. These convert the model-agnostic
    # (DivePoint, state) arguments to Bühlmann-specific ones (tissue state,
    # gradient factor line) and delegate to the internal implementations.
    #
    def description(self) -> str:
        return f'ZHL-16C GF {self.gf_low}/{self.gf_high}'

    def NDL(self, point: DivePoint, state: Any = None) -> float:
        amb_to_gf = self._get_ambtogf(point.tissue_state, point.p_amb,
                                      Util.SURFACE_PRESSURE, state)
        return self._ndl(point.tissue_state, amb_to_gf, point.p_amb, point.gas)

    def stop_needed(self, point: DivePoint, state: Any = None) -> bool:
        # When a GF line is under way, judge against it as-is; _get_ambtogf
        # would potentially replace it and thereby move the goalposts.
        amb_to_gf = state if state is not None else \
            self._get_ambtogf(point.tissue_state, point.p_amb, Util.SURFACE_PRESSURE)
        return point.tissue_state.max_over_supersat(amb_to_gf, point.p_amb) > 0.01

    def deco_info(self, point: DivePoint, gases_carried: Iterable[Gas.Gas],
                  state: Any = None) -> dict[str, Any]:
        return self._deco_info(point.tissue_state, point.depth, point.gas,
                               gases_carried, amb_to_gf=state)

    def compute_deco_profile(self, point: DivePoint, gases: Iterable[Gas.Gas],
                             p_target: float = Util.SURFACE_PRESSURE,
                             add_gas_switch_time: bool = False,
                             state: Any = None) -> tuple[list[Stop], float, AmbientToGF]:
        return self._compute_deco_profile(point.tissue_state, point.p_amb,
                                          point.gas, gases,
                                          p_target=p_target,
                                          add_gas_switch_time=add_gas_switch_time,
                                          amb_to_gf=state)

    #
    # NDL, deco stop, deco profile computations
    #
    def _ndl(self, tissue_state: TissueStateCython.TissueState,
             amb_to_gf: AmbientToGF, p_amb: float, gas: Gas.Gas) -> float:
        """No-decompression limit: how long (minutes) we can stay at this
        depth and still ascend directly to the surface."""
        # Binary search on t, the time we still stay at this depth:
        #   t0 < h < t1
        #   max_over_supersat(t0) < 0
        #   max_over_supersat(t1) > 0

        def max_over_supersat(t: float) -> float:
            ts2 = tissue_state.updated_state(t, p_amb, gas)
            ts3 = self._update_tissue_state_travel(ts2, p_amb, Util.SURFACE_PRESSURE, gas)
            return ts3.max_over_supersat(amb_to_gf, Util.SURFACE_PRESSURE)

        t0 = 0.0
        t1 = self._stop_length_infinity
        if max_over_supersat(t0) > 0:
            # Already in deco
            return t0
        if max_over_supersat(t1) < 0:
            # We can stay here indefinitely
            return t1

        while t1 - t0 > 0.01:
            h = t0 + (t1 - t0) / 2
            if max_over_supersat(h) < 0:
                t0 = h
            else:
                t1 = h

        return t0

    def _best_deco_gas(self, p_amb: float, gases: Iterable[Gas.Gas]) -> Gas.Gas:
        return Gas.best_gas(gases, p_amb, self.max_pO2_deco)

    def _gas_switch_p_amb(self, gas: Gas.Gas) -> float:
        """The (stop-aligned) ambient pressure at which to switch to this gas."""
        p_amb = self.max_pO2_deco / gas['fO2']
        return Util.Pamb_to_Pamb_stop(p_amb, direction='up',
                                      last_stop_depth=self.last_stop_depth)

    def _time_to_stay_at_stop(self, p_amb: float, p_amb_next_stop: float,
                              tissue_state: TissueStateCython.TissueState,
                              gas: Gas.Gas, amb_to_gf: AmbientToGF) -> float:
        """Minutes to wait at this stop before the next stop is allowed.

        Straight computation is possible (even for Trimix), but comes down to
        solving a pretty messy quadratic equation, and I'm too lazy for that.
        Binary search is fast enough and more robust (and again, lazy).
        """
        def max_over_supersat(t: float) -> float:
            ts2 = tissue_state.updated_state(t, p_amb, gas)
            return ts2.max_over_supersat(amb_to_gf, p_amb_next_stop)

        # Binary search:
        #   t0 <= t <= t1, max_over_supersat(t) = 0.
        #   max_over_supersat(t0) > 0
        #   max_over_supersat(t1) < 0
        t0 = 0.0
        t1 = 5.0
        if max_over_supersat(t0) < 0:
            # Already OK
            return 0.0
        # For t1, infinity is too long: staying at this stop depth forever may
        # lead to too high loading for next stop (in the slow tissues). So
        # there is a certain point in time where the faster tissues are fine
        # (because of the stop) and the slower are still fine.
        while max_over_supersat(t1) > 0 and t1 < self._stop_length_infinity:
            t1 *= 2.0
        if max_over_supersat(t1) > 0:
            # Still on-gassing
            return 0.0
        while t1 - t0 > self.stop_length_precision:
            h = t0 + (t1 - t0) / 2
            if max_over_supersat(h) > 0:
                t0 = h
            else:
                t1 = h
        return t1

    def _get_ambtogf(self, tissue_state: TissueStateCython.TissueState,
                     p_amb: float, p_target: float,
                     amb_to_gf: AmbientToGF | None = None) -> AmbientToGF:
        """Determine the GF line (ceiling, allowed supersaturation per level).

        An existing amb_to_gf may be passed in when we recompute part of the
        deco after it has already started; it is considered void if the
        ceiling has meanwhile dropped below the original first stop.
        """
        if amb_to_gf is None:
            p_ceiling = tissue_state.p_ceiling_for_gf_now(self.gf_low)
            return AmbientToGF(p_ceiling, p_target, self.gf_low, self.gf_high)
        p_ceiling = tissue_state.p_ceiling_for_amb_to_gf(amb_to_gf)
        if p_ceiling > amb_to_gf.p_first_stop:
            return self._get_ambtogf(tissue_state, p_amb, p_target)
        return amb_to_gf

    def _update_tissue_state_travel(self, state: TissueStateCython.TissueState,
                                    p_amb: float, p_new_amb: float,
                                    gas: Gas.Gas) -> TissueStateCython.TissueState:
        """Tissue state after ascending/descending from p_amb to p_new_amb,
        approximating the travel as time spent at the average pressure."""
        time = 0.0
        if p_amb > p_new_amb:
            time = (p_amb - p_new_amb) / (self.ascent_speed * Util.BAR_PER_METER)
        elif p_amb < p_new_amb:
            time = (p_new_amb - p_amb) / (self.descent_speed * Util.BAR_PER_METER)
        if time == 0.0:
            return state
        assert time > 0.0
        p_avg = (p_amb + p_new_amb) / 2.0
        return state.updated_state(time, p_avg, gas)

    def _deco_profile_p_amb_next_stop(self, p_now: float, p_first_stop: float,
                                      current_gas: Gas.Gas,
                                      gases: Iterable[Gas.Gas],
                                      add_gas_switch_time: bool = True
                                      ) -> tuple[float, Gas.Gas]:
        """The ambient pressure of the next stop (possibly a gas switch stop)
        and the gas to use there."""
        if p_now > p_first_stop:
            p_amb_next_stop = p_first_stop
        else:
            p_amb_next_stop = Util.next_stop_Pamb(p_now, last_stop_depth=self.last_stop_depth)
        # Do we need a gas switch?
        new_gas = self._best_deco_gas(p_amb_next_stop, gases)
        if new_gas != current_gas and p_first_stop > Util.SURFACE_PRESSURE and add_gas_switch_time:
            p_amb_gas_switch = self._gas_switch_p_amb(new_gas)
            if p_amb_next_stop - 0.01 < p_amb_gas_switch < p_now - 0.01:
                p_amb_next_stop = p_amb_gas_switch
        return p_amb_next_stop, new_gas

    def _compute_deco_profile(self, tissue_state: TissueStateCython.TissueState,
                              p_amb: float,
                              current_gas: Gas.Gas, gases: Iterable[Gas.Gas],
                              p_target: float = Util.SURFACE_PRESSURE,
                              add_gas_switch_time: bool = False,
                              amb_to_gf: AmbientToGF | None = None
                              ) -> tuple[list[Stop], float, AmbientToGF]:
        """Compute the decompression profile from p_amb up to p_target.

        Returns (stops, p_ceiling, amb_to_gf), where stops is a list of
        (depth, duration, gas) triples.
        """
        amb_to_gf = self._get_ambtogf(tissue_state, p_amb, p_target, amb_to_gf)
        p_ceiling = tissue_state.p_ceiling_for_amb_to_gf(amb_to_gf)
        assert p_ceiling < 100.0  # Otherwise something very weird is happening
        p_first_stop = Util.Pamb_to_Pamb_stop(p_ceiling, last_stop_depth=self.last_stop_depth)
        # Start at deepest point from first_stop at ambient
        p_now = max(p_first_stop, p_amb)
        # If we don't need to add gas_switch_time, instantly switch to the best gas
        if add_gas_switch_time:
            gas_now = current_gas
        else:
            gas_now = self._best_deco_gas(p_now, gases)
        # 'walk up'
        result: list[Stop] = []
        while p_now > p_target + 0.01:
            p_amb_next_stop, gas_next_stop = self._deco_profile_p_amb_next_stop(
                p_now, p_first_stop, gas_now, gases,
                add_gas_switch_time=add_gas_switch_time)
            stoplength = self._time_to_stay_at_stop(p_now, p_amb_next_stop,
                                                    tissue_state, gas_now, amb_to_gf)
            tissue_state = tissue_state.updated_state(stoplength, p_now, gas_now)
            if stoplength > self.stop_length_precision:
                result.append((Util.Pamb_to_depth(p_now), stoplength, gas_now))
            # Travel to next stop
            tissue_state = self._update_tissue_state_travel(tissue_state, p_now,
                                                            p_amb_next_stop, gas_now)
            # Consider adding gas switch
            if gas_now != gas_next_stop and add_gas_switch_time:
                result.append((Util.Pamb_to_depth(p_amb_next_stop), self.gas_switch_mins, gas_now))
                tissue_state = tissue_state.updated_state(self.gas_switch_mins,
                                                          p_amb_next_stop, gas_now)
                result.append((Util.Pamb_to_depth(p_amb_next_stop), 0.0, gas_next_stop))
            # Onto the next stop!
            p_now = p_amb_next_stop
            gas_now = gas_next_stop
        return result, p_ceiling, amb_to_gf

    def _deco_info(self, tissue_state: TissueStateCython.TissueState,
                   depth: float, gas: Gas.Gas,
                   gases_carried: Iterable[Gas.Gas],
                   amb_to_gf: AmbientToGF | None = None) -> dict[str, Any]:
        """All decompression info for one point in a dive: ceilings, gradient
        factors, stops, time to surface, and no-decompression limit."""
        p_amb = Util.depth_to_Pamb(depth)
        result = self.tissue_state_info(tissue_state, p_amb)

        # Below is about computing the decompression profile
        stops, p_ceiling, amb_to_gf = self._compute_deco_profile(
            tissue_state, p_amb, gas, gases_carried, amb_to_gf=amb_to_gf)
        nontrivialstops = [s for s in stops if s[1] >= .1]
        result['Ceil'] = Util.Pamb_to_depth(p_ceiling)
        result['Stops'] = stops
        result['FirstStop'] = nontrivialstops[0][0] if len(nontrivialstops) > 0 else 0
        result['TTS'] = depth / self.ascent_speed + sum(s[1] for s in stops)
        result['NDL'] = self._ndl(tissue_state, amb_to_gf, p_amb, gas)
        # The continuation state, both under the interface-level key and,
        # for Bühlmann-aware consumers (eg the GF line plot), the old name.
        result['model_state'] = amb_to_gf
        result['amb_to_gf'] = amb_to_gf

        return result
