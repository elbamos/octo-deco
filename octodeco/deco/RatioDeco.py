# Please see LICENSE.md
"""Ratio decompression.

STUB — the model scaffolding is in place, the actual ratio deco rules are
still to be written (see the NotImplementedError methods below).

RatioDeco implements the DecompressionModel interface, so once the rules are
filled in, a dive switches to ratio deco simply by setting its model type:

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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from . import Util
from .DecompressionModel import DecompressionModel
from .Util import Stop

if TYPE_CHECKING:
    from . import Gas
    from .DivePoint import DivePoint
    from .DiveProfile import DiveProfile


class RatioDeco(DecompressionModel):
    MODEL_TYPE = 'RatioDeco'

    def __init__(self,
                 descent_speed: float, ascent_speed: float,
                 max_pO2_deco: float, gas_switch_mins: float,
                 last_stop_depth: float):
        super().__init__()
        self.descent_speed = descent_speed
        self.ascent_speed = ascent_speed
        self.max_pO2_deco = max_pO2_deco
        self.gas_switch_mins = gas_switch_mins
        self.last_stop_depth = last_stop_depth
        # TODO: ratio deco configuration (eg the ratio, reference depth,
        # ascent shape parameters). Whatever is added here should also be
        # reflected in settings() / for_profile() so it round-trips.

    #
    # The DecompressionModel interface
    #
    @classmethod
    def for_profile(cls, diveprofile: DiveProfile,
                    settings: dict[str, Any]) -> RatioDeco:
        # TODO: pass the ratio deco configuration from `settings` once there
        # is some (see settings()).
        return cls(diveprofile._descent_speed, diveprofile._ascent_speed,
                   diveprofile._max_pO2_deco, diveprofile._gas_switch_mins,
                   diveprofile._last_stop_depth)

    def settings(self) -> dict[str, Any]:
        # TODO: return the ratio deco configuration; must round-trip through
        # for_profile().
        return {}

    def description(self) -> str:
        # TODO: include the configuration, eg 'Ratio deco 1:1'.
        return 'Ratio deco'

    def NDL(self, point: DivePoint, state: Any = None) -> float:
        """No-decompression limit at this point: how long (minutes) the diver
        can stay at this depth, on this gas, and still ascend directly to the
        surface."""
        raise NotImplementedError

    def stop_needed(self, point: DivePoint, state: Any = None) -> bool:
        """Whether this point violates the model's ascent limits, ie deco
        stops should have been made before reaching it.

        DiveProfile.add_stops calls this on each point of the raw profile;
        answering True triggers compute_deco_profile to generate the stops
        leading up to it.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def deco_info(self, point: DivePoint, gases_carried: Iterable[Gas.Gas],
                  state: Any = None) -> dict[str, Any]:
        # Generic assembly: tissue metrics from the shared tracking, the rest
        # from compute_deco_profile / NDL. Works as-is once those are
        # implemented; refine if ratio deco wants to report more.
        result = self.tissue_state_info(point.tissue_state, point.p_amb)
        stops, p_ceiling, state = self.compute_deco_profile(point, gases_carried,
                                                            state=state)
        nontrivialstops = [s for s in stops if s[1] >= .1]
        result['Ceil'] = Util.Pamb_to_depth(p_ceiling)
        result['Stops'] = stops
        result['FirstStop'] = nontrivialstops[0][0] if len(nontrivialstops) > 0 else 0
        result['TTS'] = point.depth / self.ascent_speed + sum(s[1] for s in stops)
        result['NDL'] = self.NDL(point, state=state)
        result['model_state'] = state
        return result
