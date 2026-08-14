# Please see LICENSE.md
import time;

import pandas;
from flask import (
    g, Blueprint, flash, redirect, url_for, request, abort, jsonify, session
)
from markupsafe import escape;

from . import app, plots, units, user, db_api_dive;
from .util.features import AllowedFeature as uft;

cache = app.get_the_cache();


bp = Blueprint('dive', __name__, url_prefix='/dive')


@bp.before_request
def load_user_details():
    user.get_user_details();


#
# Getting info
#
def _get_arg_multikey(args, tp, keys, default):
    r = None;
    try:
        for k in keys:
            r = args.get(k);
            if r is not None and r != '':
                return tp(r);
    except ValueError:
        pass;
    return default;


def get_gf_args_from_request():
    if 'gf_args' not in g:
        # GFs
        args = dict(request.args);
        args.update(request.form);
        gflow = min(200, max(0, _get_arg_multikey(args, int, [ 'ipttxt_gflow', 'gflow'], 101)));
        gfhigh = min(200, max(0, _get_arg_multikey(args, int, [ 'ipttxt_gfhigh', 'gfhigh' ], 101)));
        # Deco model selection + ratio deco parameters
        model = args.get('model');
        if model not in ('Buhlmann', 'RatioDeco'):
            model = '';    # '' = the dive's own model
        curve = args.get('curve');
        if curve not in ('s_curve_deep', 's_curve_shallow', 'exponential'):
            curve = '';    # '' = the ratio deco version's own default shape
        # Display units; remembered in the session so the choice sticks
        # across dives and page loads.
        u = args.get('units');
        if u in ('metric', 'imperial'):
            session['units'] = u;
        else:
            u = session.get('units', 'metric');
        # Done
        g.gf_args = { 'gflow': gflow, 'gfhigh': gfhigh, 'model': model, 'curve': curve,
                      'units': u };
    return g.gf_args;


# Using this pattern as it enables invalidation of dive cache as a whole
class CachedDiveProfile:
    def __init__(self, dive_id):
        self.dive_id = dive_id;
        self.lastupdate = time.monotonic();

    def __repr__(self):
        return 'CDP-{}-{}'.format(self.dive_id, self.lastupdate);

    @cache.memoize()
    def profile_base(self):
        dp = db_api_dive.get_one_dive(self.dive_id);
        if dp is None:
            return None;
        return dp;

    def _selected_model(self, req_args):
        if req_args.get('model') in ('Buhlmann', 'RatioDeco'):
            return req_args['model'];
        dp = self.profile_base();
        return dp.deco_model_type() if dp is not None else 'Buhlmann';

    def _model_settings_from_args(self, req_args, model_type):
        dp = self.profile_base();
        if model_type == 'RatioDeco':
            return { 'curve_shape': req_args['curve'] };
        gflow = req_args['gflow'];
        gfhigh = req_args['gfhigh'];
        if (gflow, gfhigh) == (101, 101):
            gflow = dp.gf_low_display if dp.gf_low_display is not None else 35;
            gfhigh = dp.gf_high_display if dp.gf_high_display is not None else 70;
        return { 'gf_low': gflow, 'gf_high': gfhigh };

    @cache.memoize()
    def _plan_result(self, req_args, model_type):
        # The dive replanned under the given model: same bottom phase, that
        # model's stops. Returns (profile, None), or (None, error message)
        # when replanning is not possible (eg a gas the model does not
        # support), or (None, None) for dives that should not be replanned
        # at all (imported dives).
        dp = self.profile_base();
        if dp is None or dp.runtimetable() is None:
            return (None, None);
        cp = dp.clean_copy();
        # clean_copy drops database identity; the display layer needs it
        cp.dive_id = dp.dive_id;
        cp.user_id = dp.user_id;
        cp._deco_model_type = model_type;
        cp._model_settings_display = self._model_settings_from_args(req_args, model_type);
        try:
            cp.update_stops();
        except Exception as err:
            return (None, str(err) or type(err).__name__);
        if model_type == 'RatioDeco':
            # In ratio deco there is no surface stop: the plan ends at
            # surfacing. Without this, update_stops re-appends the stored
            # dive's surface section right after the last (3 m) stop, where
            # it reads as deco time spent at 0 m.
            cp.remove_surface_at_end();
        return (cp, None);

    def profile_plan(self, req_args, model_type):
        return self._plan_result(req_args, model_type)[ 0 ];

    def plan_status_html(self, req_args):
        # Short HTML note about plans that could not be computed; empty if
        # all is well.
        msgs = [];
        for model_type, label in ( ('Buhlmann', 'B&uuml;hlmann'), ('RatioDeco', 'Ratio deco') ):
            _, err = self._plan_result(req_args, model_type);
            if err is not None:
                msgs.append(f'No {label} plan: {escape(err)}');
        return '<br/>'.join(msgs);

    @cache.memoize()
    def profile_args(self, req_args):
        # The profile to display: the selected model's plan where possible,
        # otherwise the stored profile with display GFs applied (old
        # behaviour; also what imported dives get).
        dp = self.profile_plan(req_args, self._selected_model(req_args));
        if dp is not None:
            return dp;
        dp = self.profile_base();
        gflow = req_args['gflow'];
        gfhigh = req_args['gfhigh'];
        if dp is None:
            return None;
        if (gflow, gfhigh) == (101, 101):
            gflow, gfhigh = dp.gf_low_display, dp.gf_high_display;
        if (gflow, gfhigh) != (dp.gf_low_display, dp.gf_high_display):
            dp.set_gf(gflow, gfhigh);
        return dp;

    @cache.memoize()
    def user_id(self):
        return self.profile_base().user_id;

    def stored_decotime(self):
        dp = self.profile_base();
        return dp.decotime() if dp is not None else 0.0;

    @cache.memoize()
    def plot_profile(self, req_args):
        dp = self.profile_args(req_args);
        # Overlay the other model's plan depth for comparison
        selected = self._selected_model(req_args);
        other = 'RatioDeco' if selected == 'Buhlmann' else 'Buhlmann';
        other_dp = self.profile_plan(req_args, other);
        other_label = 'Ratio deco' if other == 'RatioDeco' else 'Bühlmann';
        try:
            jp = plots.show_diveprofile(dp, other_profile = other_dp,
                                        other_label = other_label,
                                        imperial = units.is_imperial(req_args));
        except TypeError:
            jp = {};
        return jsonify(jp);

    @cache.memoize()
    def plot_heatmap(self, req_args):
        dp = self.profile_args(req_args);
        try:
            jp = plots.show_heatmap(dp);
        except TypeError:
            jp = {};
        return jsonify(jp);

    @cache.memoize()
    def plot_pressure_graph(self, req_args):
        dp = self.profile_args(req_args);
        try:
            jp = plots.show_pressure_graph(dp);
        except TypeError:
            jp = {};
        return jsonify(jp);

    @cache.memoize()
    def summary_table(self, req_args):
        dp = self.profile_args(req_args)
        ds = dp.dive_summary();
        if units.is_imperial(req_args):
            ds['Last stop'] = '{:.0f} ft'.format(units.depth(dp._last_stop_depth, True));
        dsdf = pandas.DataFrame([ [ k, v ] for k, v in ds.items() ]);
        dsdf_table = dsdf.to_html(classes="smalltable", header="true");
        return dsdf_table;

    @cache.memoize()
    def deco_plan_table(self, req_args):
        # The executed deco stops of the displayed profile: one row per
        # (depth, gas) dwell, aggregated from the profile's deco points.
        dp = self.profile_args(req_args);
        imperial = units.is_imperial(req_args);
        du = units.depth_unit(imperial);
        rows = [];
        for p in dp.points():
            if not p.is_deco_stop or p.depth <= 0 or p.duration <= 0:
                continue;
            if p.prev is None or abs(p.prev.depth - p.depth) > 0.01:
                # Travel between stops, not time spent at one
                continue;
            if len(rows) > 0 and abs(rows[-1]['depth'] - p.depth) < 0.01 \
                    and rows[-1]['gas'] == p.gas:
                rows[-1]['duration'] += p.duration;
            else:
                rows.append({'depth': p.depth, 'duration': p.duration, 'gas': p.gas});
        if len(rows) == 0:
            return 'No decompression stops.';
        dsdf = pandas.DataFrame([
            {'depth ({})'.format(du): '{:.0f}'.format(units.depth(r['depth'], imperial)),
             'duration (min)': '{:.1f}'.format(r['duration']),
             'gas': str(r['gas'])}
            for r in rows]);
        return dsdf.to_html(classes="smalltable", header="true", index=False);

    @cache.memoize()
    def runtime_table(self, req_args):
        dp = self.profile_args(req_args);
        rtt = dp.runtimetable();
        if rtt is None:
            return 'A runtime table is unfortunately not available for this dive.';
        dsdf = pandas.DataFrame(rtt);
        imperial = units.is_imperial(req_args);
        du = units.depth_unit(imperial);
        vu = units.volume_unit(imperial);
        pu = units.pressure_unit(imperial);
        desired_col_seq = [ 'depth', 'time', 'gas', 'gas_usage' ];
        for c in set(desired_col_seq).difference(dsdf.columns):
            dsdf[ c ] = '';
        dsdf = dsdf[ desired_col_seq ].rename(columns = {'depth': 'depth ({})'.format(du),
                                                         'gas_usage': 'gas usage'});
        frm = {
            'depth ({})'.format(du): lambda x: '{:.0f}'.format(units.depth(x, imperial)),
            'time': lambda x: '{:.1f}'.format(x) if not pandas.isnull(x) else '',
            'gas': str,
            'gas usage': lambda d: ', '.join([ '{}: {:.0f}{} used ({:.0f} {} from {})'.format(gas, units.volume(inf['liters_used'], imperial), vu, units.pressure(inf['bars_used'], imperial), pu, inf['cyl_name']) for gas,inf in d.items() ])
        };
        dsdf_table = dsdf.to_html(classes="smalltable", header="true",
                                  formatters=frm, na_rep='');
        return dsdf_table;

    @cache.memoize()
    def gas_consumption_table(self, req_args):
        imperial = units.is_imperial(req_args);
        du = units.depth_unit(imperial);
        vu = units.volume_unit(imperial);
        pu = units.pressure_unit(imperial);
        def format_lost_gas_link(slost):
            if slost is None:
                return 'planned';
            else:
                return '<a href="{}?lostgas={}">lost {}</a>'.\
                    format(url_for('dive.new_ephm_lost_gas', dive_id=self.dive_id), slost, slost);
        def format_emergency(dp, dict_emerg):
            tooltip = 'In the event of an emergency ({:.0f}x, {:.0f}mins) at {:.0f}{} you need {:.0f}% of the bottom gas {} in your {}'\
                .format(dp._gas_consmp_emerg_factor, dp._gas_consmp_emerg_mins,
                        units.depth(dp.max_depth(), imperial), du,
                        dict_emerg['perc_emerg'], dict_emerg['bottom_gas'], dict_emerg['cyl_name']);
            text = '{:.0f}%'.format(dict_emerg['perc_emerg']);
            cl = 'gas_{}'.format(dict_emerg['ok']);
            return '<div class="tooltip {}">{}<span class="tooltiptext">{}</span></div>'.format(cl, text, tooltip);
        def format_gas_usage(gas, inf):
            if inf['liters'] == 0.0:
                return '-';
            text1 = '{:.0f}{}'.format(units.volume(inf['liters'], imperial), vu);
            text2 = '{:.0f}%'.format(inf['perc']);
            tooltip = '{:.0f} {} of {}'.format(units.pressure(inf['bars'], imperial), pu, inf['cyl_name']);
            cl = 'gas_{}'.format(inf['ok']);
            r = '{} [<span class="tooltip {}">{}<span class="tooltiptext">{}</span></span>]'.format(text1, cl, text2, tooltip);
            return r;
        dp = self.profile_base();
        gct = dp.gas_consumption_analysis();
        gct_formatted = {
            format_lost_gas_link(s['lost'])
            :
            { ' deco time': '{:.1f}mins'.format(s[ 'decotime' ]),
              ' emergency': format_emergency(dp, s['emergency']),
              **{str(gas): format_gas_usage(gas, inf)
                 for gas, inf in s[ 'gas_consmp' ].items()}}
            for s in gct };
        dsdf = pandas.DataFrame(gct_formatted);
        dsdf_table = dsdf.to_html(classes="smalltable", na_rep='', escape=False);
        info = 'Computed with bottom: {:.1f}{}/min, deco: {:.1f}{}/min.'.\
            format(units.volume(dp._gas_consmp_bottom, imperial), vu,
                   units.volume(dp._gas_consmp_deco, imperial), vu);
        warning = ' This does not always fully take max pO2 into account.';
        return dsdf_table + '<br/>'+ info + warning;

    @cache.memoize()
    def full_table(self, req_args):
        dp = self.profile_args(req_args);
        df = dp.dataframe();
        if units.is_imperial(req_args):
            # Depth-valued columns; other columns are minutes / bar /
            # percentages, and the Stops string stays metric (library form).
            for c in ('depth', 'FirstStop', 'Ceil', 'Ceil99'):
                df[c] = df[c].map(lambda v: units.depth(v, True)
                                  if isinstance(v, (int, float)) else v);
            df = df.rename(columns = {c: '{} (ft)'.format(c)
                                      for c in ('depth', 'FirstStop', 'Ceil', 'Ceil99')});
        fulldata_table = df.to_html(classes = "bigtable", header = "true");
        return fulldata_table;

    @cache.memoize()
    def gfdeco_table(self, req_args):
        dp = self.profile_args(req_args)
        t0 = time.perf_counter();
        dtt = dp.decotimes_for_gfs();
        t1 = time.perf_counter();
        if dtt is None:
            return 'Such a table is unfortunately not available for this dive.';
        # Format
        url = url_for('dive.show', dive_id=self.dive_id);
        dtt2 = { gflow: {
                gfhigh: (val,
                         '<a href="{}?gflow={:d}&gfhigh={:d}">{:.1f}</a>'.format(url, gflow, gfhigh, val),
                         gflow == dp.gf_low_profile and gfhigh == dp.gf_high_profile)
                for gfhigh, val in r.items() } for gflow, r in dtt.items() };
        minv = min(min([ v for v in r.values() ] for r in dtt.values()));
        maxv = max(max([ v for v in r.values() ] for r in dtt.values()));
        def style_map(v):
            try:
                op = (v[0]-minv)/(maxv-minv);
            except ZeroDivisionError:
                op = 0.5;
            # base color varies between 150,150,150 and 230,230,230
            r = 'background-color:rgba({0},{0},{0},0.35);'.format(150 + round((1-op)*80));
            if v[2]:
                # Matches current GF's.
                r += 'border:2px dashed black';
            return r;
        def format_map(v):
            return v[1];
        df = pandas.DataFrame(dtt2).transpose();
        # Styler.render()/applymap() were removed/renamed in modern pandas
        styled_df = df.style\
            .set_table_attributes('class="dataframe gfdecotable smalltable"')\
            .map(style_map)\
            .format(format_map)\
            .to_html();
        html_comp_time = 'Computation time: {:.2f}s.'.format(t1-t0);
        return styled_df + '<br/>' + html_comp_time;


@cache.memoize()
def get_cached_dive(dive_id: str, user_id: str):
    # Have to have a cache per user/dive combination,
    # since behaviour could/should be different.
    assert user_id == user.get_user_details().user_id();
    cdp = CachedDiveProfile(dive_id);
    if cdp is None or cdp.profile_base() is None:
        session[ 'last_dive_id' ] = None;
        flash(f'There was a problem finding the dive {dive_id}');
        return None;
    if not user.get_user_details().is_allowed(uft.DIVE_VIEW, dive=cdp.profile_base()):
        session[ 'last_dive_id' ] = None;
        flash(f'There was a problem showing the dive {dive_id}');
        return None;
    if not cdp.profile_base().is_ephemeral:
        session[ 'last_dive_id' ] = dive_id;
    return cdp;


def invalidate_cached_dive(dive_id: str):
    user_id = user.get_user_details().user_id();
    cache.delete_memoized(get_cached_dive, dive_id, user_id);


def get_diveprofile_for_display(dive_id: str):
    cdp = get_cached_dive(dive_id, user.get_user_details().user_id());
    if cdp is None:
        return None;
    return cdp.profile_args(get_gf_args_from_request());


#
from . import dive_show;
from . import dive_change;
from . import dive_new;
