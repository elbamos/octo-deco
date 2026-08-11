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

import importlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from . import BuhlmannConstants, TissueStateCython, Util
from .Util import Stop

if TYPE_CHECKING:
    from .DivePoint import DivePoint
    from .DiveProfile import DiveProfile
    from .Gas import Gas

# Registry of concrete model types by their MODEL_TYPE name. DiveProfile
# stores the name (not the class) so that pickled dives survive refactors;
# subclasses register themselves automatically via __init_subclass__ when
# their module is imported. _MODEL_MODULES says which module provides which
# type, so model_class() can import it lazily on first lookup — a new model
# needs an entry here.
_MODEL_TYPES: dict[str, type[DecompressionModel]] = {}
_MODEL_MODULES = {
    'Buhlmann': 'Buhlmann',
    'RatioDeco': 'RatioDeco',
}


def model_class(model_type: str) -> type[DecompressionModel]:
    """Look up a registered model class by its MODEL_TYPE name, importing
    the module that provides it if needed."""
    if model_type not in _MODEL_TYPES and model_type in _MODEL_MODULES:
        importlib.import_module(f'.{_MODEL_MODULES[model_type]}', __package__)
    return _MODEL_TYPES[model_type]


class DecompressionModel(ABC):
    # Name under which the model is registered; None for abstract/helper
    # subclasses that should not be constructible from a profile.
    MODEL_TYPE: str | None = None

    # Tissue compartment configuration, shared by all models. See
    # TissueStateCython: only one set of constants is supported per run.
    TISSUE_CONSTANTS = BuhlmannConstants.ZHL_16C_1a
    RQ = 0.9  # Respiratory quotient

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if cls.MODEL_TYPE is not None:
            _MODEL_TYPES[cls.MODEL_TYPE] = cls

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
    @classmethod
    @abstractmethod
    def for_profile(cls, diveprofile: DiveProfile,
                    settings: dict[str, Any]) -> DecompressionModel:
        """Construct the model for a dive: its own configuration comes from
        `settings` (the dict shape settings() returns), dive-level parameters
        (speeds, max deco pO2, last stop depth, ...) from the profile."""

    @abstractmethod
    def settings(self) -> dict[str, Any]:
        """The model's own configuration as a plain dict, eg
        {'gf_low': 35, 'gf_high': 70}. Must round-trip through
        for_profile()."""

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

        Returns (stops, p_ceiling, state): the Stop objects, the current
        ceiling as an ambient pressure, and the continuation state for
        subsequent calls.

        A Stop's optional ascent_speed sets the speed (m/min) for the
        segment leaving that stop — from it to the next, shallower stop, or
        to the surface if it is the last (eg ratio deco's slow final
        ascent). Stops without it ascend at the dive's normal ascent speed.
        """
