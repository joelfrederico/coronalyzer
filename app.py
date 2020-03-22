#!/usr/bin/env python3
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import scipy.optimize as spopt
import json

app = dash.Dash(__name__)

server = app.server

with open('data/los_angeles.json') as f:
    data = json.load(f)

new_case_data = data['New Cases'][1:]
date_times = pd.to_datetime(data['Date'][1:])

total_dat = np.empty_like(new_case_data)
total = 0
for (i, val) in enumerate(new_case_data):
    total += val
    total_dat[i] = total

new_cases = pd.Series(new_case_data, index=date_times)

total_cases = pd.Series(total_dat, index=date_times)
total_fig = go.Figure(data=[go.Scatter(x=total_cases.index, y=total_cases)])
total_fig.update_layout(yaxis_title="Total Cases")


def resid(x, N, t):
    return N - 2**((t - x[1])/x[0])


def get_jac(x, N, t):
    t_sub_t0 = t - x[1]
    exp_part = 2**(t_sub_t0/x[0])
    res = np.array(
            [np.log(2)*t_sub_t0*exp_part/x[0]**2, exp_part*np.log(2)/x[0]])
    return res.transpose()


def get_rate(series):
    t = series.index - pd.to_datetime('1/1/2020')
    t = t.values / pd.Timedelta('1d')
    result = spopt.least_squares(
            resid, [series[0], t[0]], jac=get_jac,
            kwargs={'N': series, 't': t})
    return 1/result.x[0]


window_rate_3d = total_cases.rolling('3d').apply(get_rate)
window_rate_7d = total_cases.rolling('7d').apply(get_rate)
rate_fig = go.Figure()
rate_fig.add_trace(go.Scatter(x=window_rate_3d.index, y=window_rate_3d,
                              name="3-day window"))
rate_fig.add_trace(go.Scatter(x=window_rate_7d.index, y=window_rate_7d,
                              name="7-day window"))
rate_fig.update_layout(
        yaxis_title=r'Doubling Growth Factor',
        showlegend=True)

app.layout = html.Div([
    html.H2('Coronalyzer'),
    dcc.Graph(
        id='total_cases',
        figure=total_fig
    ),
    dcc.Graph(
        id='rate',
        figure=rate_fig
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)
