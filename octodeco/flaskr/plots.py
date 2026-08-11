# Please see LICENSE.md
import json;

import numpy;
import plotly;
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from octodeco.deco import Util;
from . import units;


def show_diveprofile(diveprofile, other_profile = None, other_label = None,
                     imperial = False):
    df = diveprofile.dataframe();
    dconv = units.M_TO_FT if imperial else 1.0;
    du = units.depth_unit(imperial);
    fig = sp.make_subplots(specs = [ [ {"secondary_y": True} ] ])
    # The other deco model's plan, depth only, drawn first so the primary
    # depth line ends up on top
    if other_profile is not None:
        fig.add_trace(go.Scatter(x = [ p.time for p in other_profile.points() ],
                                 y = [ dconv * p.depth for p in other_profile.points() ],
                                 name = f'Depth ({other_label})',
                                 hovertemplate = 'Depth (' + (other_label or '') +
                                                 '): %{y:.1f}' + du + ' @ %{x:.1f}mins<extra></extra>',
                                 line = {'color': 'rgb(150,150,165)', 'dash': 'dash', 'width': 2}));
    # ppO2
    fig.add_trace(go.Scatter(x = df[ "time" ], y = 100 * df[ "ppO2" ], name = 'ppO2',
                             hovertemplate = 'ppO2: %{customdata:.2f} @ %{x:.1f}mins<extra></extra>',
                             customdata = df[ "ppO2" ],
                             line = {'color': 'rgb(176,42,143)', 'dash': 'dot', 'width': 1},
                             visible = "legendonly"),
                  secondary_y = True);
    # CNS
    fig.add_trace(go.Scatter(x = df[ "time" ], y = df[ "CNS" ], name = 'CNS',
                             hovertemplate = 'CNS: %{y:.1f}% @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(176,42,143)', 'width': 2},
                             visible = "legendonly"),
                  secondary_y = True);
    # Integral Supersat (scaled; using customdata)
    fig.add_trace(go.Scatter(x = df[ "time" ], y = 0.3*df[ "IntSuperSat" ], name = 'IntSuperSat',
                             hovertemplate = 'IntSuperSat: %{customdata:.1f} @ %{x:.1f}mins<extra></extra>',
                             customdata = df[ "IntSuperSat" ],
                             line = {'color': 'rgb(176,42,143)', 'width': 1},
                             visible = "legendonly"),
                  secondary_y = True);
    # TTS (scaled to make visible; using customdata)
    fig.add_trace(go.Scatter(x = df[ "time" ], y = 2*df[ "TTS" ], name = 'TTS',
                             hovertemplate = 'TTS: %{customdata:.1f}mins @ %{x:.1f}mins<extra></extra>',
                             customdata = df[ "TTS" ],
                             line = {'color': 'rgb(44,160,174)', 'width': 2},
                             visible = "legendonly"),
                  secondary_y = True);
    # NDL (scaled to make visible; using customdata)
    fig.add_trace(go.Scatter(x = df[ "time" ], y = 2*df[ "NDL" ], name = 'NDL',
                             hovertemplate = 'NDL: %{customdata:.1f}mins @ %{x:.1f}mins<extra></extra>',
                             customdata = df[ "NDL" ],
                             line = {'color': 'rgb(0,173,239)', 'width': 2},
                             visible = "legendonly"),
                  secondary_y = True);
    # Ceil99
    fig.add_trace(go.Scatter(x = df[ "time" ], y = dconv * df[ "Ceil99" ], name = 'Ceil GF99',
                             hovertemplate = 'Ceil GF99: %{y:.1f}' + du + ' @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(251,165,56)', 'dash': 'dot', 'width': 2},
                             visible = "legendonly"));
    # Ceil
    fig.add_trace(go.Scatter(x = df[ "time" ], y = dconv * df[ "Ceil" ], name = 'Ceil',
                             hovertemplate = 'Ceil: %{y:.1f}' + du + ' @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(251,165,56)', 'dash': 'dot', 'width': 2}));
    # SurfaceGF
    fig.add_trace(go.Scatter(x = df[ "time" ], y = df[ "SurfaceGF" ], name = 'SurfaceGF',
                             hovertemplate = 'SurfaceGF: %{y:.1f} @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(251,165,56)', 'width': 3},
                             visible = "legendonly"),
                  secondary_y = True);
    # GF99
    fig.add_trace(go.Scatter(x = df[ "time" ], y = df[ "GF99" ], name = 'GF99',
                             hovertemplate = 'GF99: %{y:.1f} @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(255,255,0)', 'width': 3}),
                  secondary_y = True);
    # Depth
    fig.add_trace(go.Scatter(x = df[ "time" ], y = dconv * df[ "depth" ], name = 'Depth',
                             hovertemplate = 'Depth: %{y:.1f}' + du + ' @ %{x:.1f}mins<extra></extra>',
                             line = {'color': 'rgb(30,7,143)', 'width': 3}));
    # Leading tissue -> not that interesting
    # fig.add_trace( go.Scatter( x=df["time"], y=(100/16)*(df["LeadingTissueIndex"]+2), name='Leading tissue',
    #                            line={'color': 'rgb(251,165,56)', 'dash': 'dot', 'width': 1} ),
    #                secondary_y = True );

    #  Set some axes parameters
    fig.update_yaxes(secondary_y = False, autorange = "reversed",
                     title_text = 'Depth ({})'.format(du));
    fig.update_yaxes(secondary_y = True, showgrid = False, range = [ -1, 140 ], tick0 = 0, dtick = 20);
    fig.update_xaxes(title_text = "Time");
    # Draw
    graphjson = json.dumps( fig, cls=plotly.utils.PlotlyJSONEncoder);
    return graphjson;


def show_heatmap(diveprofile):
    # https://plot.ly/python/heatmaps/
    # Shows compartment inert gas pressure: loading during the bottom phase,
    # off-gassing during deco. (GF99 is unsuitable as the color value: it is
    # negative/zero whenever a tissue is below its M-value, so nothing shows
    # until decompression.) GF99 is still reported in the hover.
    tissue_labels = [ 'T%s' % t for t in diveprofile.deco_model()._constants.N2_HALFTIMES ];
    tissue_labels.reverse();

    points = diveprofile.points();
    n_tissues = len(tissue_labels);
    pressures = [ [ p.tissue_state.p_tissue(i) for i in range(n_tissues) ] for p in points ];
    gf99s = [ list(p.deco_info[ 'allGF99s' ]) for p in points ];
    for v in pressures:
        v.reverse();
    for v in gf99s:
        v.reverse();
    pressures = numpy.transpose(pressures);
    gf99s = numpy.transpose(gf99s);

    times = [ p.time for p in points ]

    fig = go.Figure(data = go.Heatmap(
        z = pressures,
        customdata = gf99s,
        y = tissue_labels,
        x = times,
        hovertemplate = '%{x:.1f}mins, %{y}: %{z:.2f} bar (GF99: %{customdata:.0f}%)<extra></extra>',
        zauto = False, zmin = 0, zmax = float(pressures.max()) ));

    graphjson = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder);
    return graphjson;


def _pg_m_lines(diveprofile, constants):
    def m_line_n2(i):
        a = constants.N2_A_VALUES[i];
        b = constants.N2_B_VALUES[ i ];
        return lambda p_amb : a + p_amb / b;
    def m_line_he(i):
        a = constants.HE_A_VALUES[i];
        b = constants.HE_B_VALUES[ i ];
        return lambda p_amb : a + p_amb / b;
    if any(map(lambda g:g['fHe'] > 0.0, diveprofile.gases_carried())):
        show_m_lines = [ ('N2', m_line_n2), ('He', m_line_he) ];
    else:
        show_m_lines = [ ('N2', m_line_n2) ];
    return show_m_lines;


def _pg_m_value_gf(tissue_state_for_a_b, amb_to_gf, p_amb, i):
    return tissue_state_for_a_b._m_values_gf(amb_to_gf, p_amb)[i];


def show_pressure_graph(diveprofile):
    fig = sp.make_subplots();
    # The lines
    pts = diveprofile.points();
    # Strip the surface section
    while len(pts) > 2 and pts[-2].depth == 0:
        pts.pop(-1);
    # deco model params
    buhlmann = diveprofile.deco_model();
    con = buhlmann._constants;
    n_tissues = con.N_TISSUES;
    n2halftimes = con.N2_HALFTIMES;
    show_m_lines = _pg_m_lines(diveprofile, con);
    # The GF M-line overlay only makes sense for models with a gradient
    # factor line (ie Buhlmann); other models' deco_info has no amb_to_gf.
    pts_for_m_line_gf = [ p for p in pts
                          if p.deco_info is not None and 'amb_to_gf' in p.deco_info ];
    # Get the rest of the reusable info
    colors = px.colors.qualitative.Dark24;
    x = [ p.p_amb for p in pts ];
    max_x = max(x);
    customdata = [ p.time for p in pts ];
    max_y = 0;
    default_tissue_to_show = 3;
    # For each tissue ..
    for i in range(n_tissues):
        # The actual line of inert gas pressures
        y = [ p.tissue_state.p_tissue(i) for p in pts ];
        max_y = max(max_y, max(y));
        name = 'T{:.1f}'.format(n2halftimes[i]);
        customstrs = [ '{:.1f}mins: Ambient:{:.1f}, Comptmt:{:.1f}, GF:{:.1f}%'.\
                           format( p.time, p.p_amb, p.tissue_state.p_tissue(i), p.deco_info['allGF99s'][i])
                       for p in pts ];
        fig.add_trace(go.Scatter(x = x, y = y,
                                 name = name,
                                 mode = 'lines+markers',
                                 legendgroup = name,
                                 customdata = customstrs,
                                 hovertemplate = '%{customdata}<extra>'+name+'</extra>',
                                 line = { 'color' : colors[i] },
                                 marker = { 'symbol': 'circle', 'size' : 4, 'color' : colors[i] },
                                 visible = "legendonly" if i != default_tissue_to_show else True
                                 ));
        # Also add the M-value line(s)
        for t, m_line in show_m_lines:
            p_amb = [0,1,2,max_x];
            p_comp_max = [ m_line(i)(x) for x in p_amb ];
            fig.add_trace(go.Scatter(x = p_amb, y = p_comp_max,
                                     name = name + '_M_line_' + t,
                                     mode = 'lines',
                                     legendgroup = name,
                                     hovertemplate = '<extra>M-line for '+name+' ('+t+')</extra>',
                                     line = {'color': colors[ i ], 'dash' : 'dot', 'width' : 0.8 },
                                     showlegend = False,
                                     visible = "legendonly" if i != default_tissue_to_show else True
                                     ));
        # .. and add the resulting gradient factor line that resulted from the GF's
        if len(pts_for_m_line_gf) > 0:
            # Use amb_to_gf and m_line function based on point with largest deco obligation
            # This is not super precise (plotting all points would be), especially if you have multiple
            # decos as in the demo dive, but it will give a more insightful picture, I believe.
            # pts_for_m_line_gf only contains non-interpolated deco points.
            pt = max(pts_for_m_line_gf, key=lambda p: (p.deco_info['SurfaceGF'], p.p_amb));
            amb_to_gf = pt.deco_info['amb_to_gf'];
            p_ambs = [0.0, Util.SURFACE_PRESSURE, amb_to_gf.p_first_stop, max_x ];
            p_comp_max = [ _pg_m_value_gf(pt.tissue_state, amb_to_gf, p, i) for p in p_ambs ];
            hovertemplate = '<extra>GF M-line ({}/{}, first stop:{:.1f}m)</extra>'.\
                format(diveprofile.gf_low_display, diveprofile.gf_high_display,
                       Util.Pamb_to_depth(amb_to_gf.p_first_stop));
            fig.add_trace(go.Scatter(x = p_ambs, y = p_comp_max,
                                     name = name + '_M_line_GF',
                                     mode = 'lines+markers',
                                     legendgroup = name,
                                     hovertemplate = hovertemplate,
                                     line = {'color': colors[ i ], 'dash' : 'dash', 'width' : 1.5 },
                                     showlegend = False,
                                     visible = "legendonly" if i != default_tissue_to_show else True
                                     ));
    # The extra stuff
    max_x = max_x+0.5;
    max_y = max_y+0.5;
    # Ambient = compartment
    fig.add_trace(go.Scatter(x = [0,max(max_x,max_y)],y=[0,max(max_x,max_y)],
                             line = {'color': '#a0a0a0', 'dash' : 'dot', 'width': 1.0},
                             showlegend=False));
    # Labels etc
    fig.update_yaxes(secondary_y = False, title_text = "Compartment pressure",
                     range = [ 0, max_y ], tick0 = 0, dtick = 1);
    fig.update_xaxes(title_text = "Ambient pressure",
                     range = [ 0, max_x ], tick0 = 0, dtick = 1);
    # Draw
    graphjson = json.dumps( fig, cls=plotly.utils.PlotlyJSONEncoder);
    return graphjson;

