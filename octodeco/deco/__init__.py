# Please see LICENSE.md
"""Scuba diving decompression profile calculations.

The main entry points are DiveProfile (build and analyze a dive) and
CreateDive (factory functions). Decompression models implement the
DecompressionModel interface; Buhlmann (ZHL-16 with gradient factors)
is the default implementation.
"""
# Model types register with DecompressionModel's registry when their module
# is imported; DecompressionModel.model_class() imports them lazily on
# lookup, so an unimportable model module (eg mid-edit) only breaks its own
# model type, not the package.
