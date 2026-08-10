# Please see LICENSE.md
"""The interface every decompression model implements.

DiveProfile and DivePoint drive decompression computations through this
interface only, so that models (Bühlmann, ratio deco, ...) are
interchangeable.

Computations are anchored on a DivePoint rather than on a bare depth or
pressure: a pressure alone does not identify a moment in the dive (a
multilevel dive visits the same depth many times), while a point carries the
time, depth, gas, tissue state and — through `prev` — the entire dive so far.

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

from . import Util
from .Util import Stop

if TYPE_CHECKING:
    from .DivePoint import DivePoint
    from .Gas import Gas


class DecompressionModel(ABC):
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the model and its settings."""

    @abstractmethod
    def cleared_tissue_state(self) -> Any:
        """The tissue state of a diver who has not been diving recently.

        Models that do not track tissues may return None.
        """

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
        """All decompression info for this point: at least the keys 'Ceil',
        'Stops', 'FirstStop', 'TTS', 'NDL', and 'model_state' (the
        continuation state to pass to the next call)."""

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
