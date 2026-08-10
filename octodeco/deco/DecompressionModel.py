# Please see LICENSE.md
"""The interface every decompression model implements.

DiveProfile and DivePoint drive decompression computations through this
interface only, so that models (Bühlmann, ratio deco, ...) are
interchangeable.

Computations are anchored on a DivePoint rather than on a bare depth or
pressure: a pressure alone does not identify a moment in the dive (a
multilevel dive visits the same depth many times), while a point carries the
time, depth, gas, tissue state and — through `prev` — the entire dive so far.

Tissue tracking is shared. Whatever rule set a model uses to generate stops,
tissue loading is tracked with the Bühlmann ZHL-16 compartments: this base
class provides `cleared_tissue_state` and `tissue_state_info` concretely, and
DivePoint advances the state point by point. That is deliberate — it lets us
visualize how tissues saturate and off-gas while following any model's
profile (eg watching compartment loading evolve under a ratio deco ascent),
and compare models against the same physiological reference.

Some models need to carry information from one call to the next while an
ascent is being computed (eg Bühlmann fixes its gradient factor line at the
first stop). This is handled with an opaque continuation token `state`:
callers pass the state produced by the previous call — found under
'model_state' in a deco_info result, or returned by compute_deco_profile —
or None to start fresh. What the token contains is entirely up to the model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from . import BuhlmannConstants, TissueStateCython, Util
from .Util import Stop

if TYPE_CHECKING:
    from .DivePoint import DivePoint
    from .Gas import Gas


class DecompressionModel(ABC):
    # Tissue compartment configuration, shared by all models. See
    # TissueStateCython: only one set of constants is supported per run.
    TISSUE_CONSTANTS = BuhlmannConstants.ZHL_16C_1a
    RQ = 0.9  # Respiratory quotient

    def __init__(self):
        self._constants = self.TISSUE_CONSTANTS
        self._rq = self.RQ
        self.TissueState = TissueStateCython.TissueState

    #
    # Shared tissue tracking
    #
    def cleared_tissue_state(self) -> TissueStateCython.TissueState:
        """The tissue state of a diver who has not been diving recently:
        fully saturated at surface pressure on air."""
        return self.TissueState(self._constants, self._rq)

    def tissue_state_info(self, tissue_state: TissueStateCython.TissueState,
                          p_amb: float) -> dict[str, Any]:
        """Model-independent tissue metrics for display, derived purely from
        the tissue state at ambient pressure p_amb.

        GF99: how do compartment pressure, ambient pressure, tolerance
        compare. The % makes most sense if ambient pressure is between
        compartment pressure and tolerance; if ambient pressure is bigger
        than compartment pressure: on-gassing.
        """
        p_ceiling_99 = tissue_state.p_ceiling_for_gf_now(99.0)
        gf99s, gf99, leading_tissue_i = tissue_state.GF99_all_info(p_amb)
        surfacegf = tissue_state.GF99(Util.SURFACE_PRESSURE)
        return {
            'Ceil99': Util.Pamb_to_depth(p_ceiling_99),
            'GF99': round(gf99, 1),
            'SurfaceGF': round(surfacegf, 1),
            'LeadingTissueIndex': leading_tissue_i,
            'allGF99s': gf99s,
        }

    #
    # The model-specific interface
    #
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the model and its settings."""

    @abstractmethod
    def NDL(self, point: DivePoint, state: Any = None) -> float:
        """No-decompression limit at this point: how long (minutes) the diver
        can stay at this depth, on this gas, and still ascend directly to the
        surface."""

    @abstractmethod
    def stop_needed(self, point: DivePoint, state: Any = None) -> bool:
        """Whether this point violates the model's ascent limits, ie deco
        stops should have been made before reaching it."""

    @abstractmethod
    def deco_info(self, point: DivePoint, gases_carried: Iterable[Gas],
                  state: Any = None) -> dict[str, Any]:
        """All decompression info for this point: at least the keys of
        tissue_state_info() plus 'Ceil', 'Stops', 'FirstStop', 'TTS', 'NDL',
        and 'model_state' (the continuation state to pass to the next
        call)."""

    @abstractmethod
    def compute_deco_profile(self, point: DivePoint, gases: Iterable[Gas],
                             p_target: float = Util.SURFACE_PRESSURE,
                             add_gas_switch_time: bool = False,
                             state: Any = None) -> tuple[list[Stop], float, Any]:
        """The decompression profile for ascending from this point to
        p_target.

        Returns (stops, p_ceiling, state): the (depth, duration, gas) stop
        triples, the current ceiling as an ambient pressure, and the
        continuation state for subsequent calls.
        """
