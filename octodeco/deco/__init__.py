# Please see LICENSE.md
"""Scuba diving decompression profile calculations.

The main entry points are DiveProfile (build and analyze a dive) and
CreateDive (factory functions). Decompression models implement the
DecompressionModel interface; Buhlmann (ZHL-16 with gradient factors)
is the default implementation.
"""
