# Please see LICENSE.md
"""Scuba diving decompression profile calculations.

The main entry points are DiveProfile (build and analyze a dive) and
CreateDive (factory functions). Decompression models implement the
DecompressionModel interface; Buhlmann (ZHL-16 with gradient factors)
is the default implementation.
"""
# Importing the model modules registers their model types with
# DecompressionModel's registry; the package always initializes before any
# of its submodules, so the registry is complete wherever you import from.
from . import Buhlmann, RatioDeco  # noqa: F401
