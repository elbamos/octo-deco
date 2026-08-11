# Please see LICENSE.md
#
# Display-unit conversions (metric -> imperial).
#
# The deco library is strictly metric (meters / bar / liters); everything
# here is for the display layer only. req_args['units'] selects the system.

M_TO_FT = 3.280839895;
L_TO_CUFT = 0.0353146667;
BAR_TO_PSI = 14.503773773;


def is_imperial(req_args):
    return req_args.get('units') == 'imperial';


def depth(m, imperial):
    return m * M_TO_FT if imperial else m;


def depth_unit(imperial):
    return 'ft' if imperial else 'm';


def volume(liters, imperial):
    return liters * L_TO_CUFT if imperial else liters;


def volume_unit(imperial):
    return 'cuft' if imperial else 'L';


def pressure(bar, imperial):
    return bar * BAR_TO_PSI if imperial else bar;


def pressure_unit(imperial):
    return 'psi' if imperial else 'bar';
