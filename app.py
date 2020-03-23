#!/usr/bin/env python3
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import scipy.optimize as spopt
import json

app = dash.Dash(__name__)

server = app.server

world_df = pd.read_csv('data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')


def get_label(row):
    label = row.loc['Country/Region']
    state = row.loc['Province/State']
    if pd.notna(state):
        label += f" ({state})"
    return label


dropdown = []
for (i, row) in world_df.iterrows():
    label = get_label(row)
    dropdown.append({'label': label, 'value': i})

us_s = world_df.loc[230][4:]
us_s.index = pd.DatetimeIndex(us_s.index)

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
# total_cases = us_s
total_fig = go.Figure()
total_fig.update_layout(title='Total Cases', yaxis_title="Cases")


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
            resid, [3, t[0]], jac=get_jac,
            kwargs={'N': series, 't': t})
    return 1/result.x[0]


rate_fig = go.Figure()
rate_fig.update_layout(
        title='3 Day Fit',
        yaxis_title=r'Doubling Growth Factor',
        showlegend=True)

rate_fig_7 = go.Figure()
rate_fig_7.update_layout(
        title='7 Day Fit',
        yaxis_title=r'Doubling Growth Factor',
        showlegend=True)

app.layout = html.Div([
    html.H2('Coronalyzer'),
    html.Div([
        dcc.Dropdown(
            id='dropdown',
            options=dropdown,
            value=[230, 140, 65],
            multi=True)
        ]),
    dcc.Graph(
        id='total-cases-1',
        figure=total_fig
    ),
    dcc.Graph(
        id='rate',
        figure=rate_fig
    ),
    dcc.Graph(
        id='rate7',
        figure=rate_fig_7
    ),
    html.Div(id='output')
])


def row_to_series(row_df):
    series = row_df[4:]
    series.index = pd.DatetimeIndex(series.index)
    series = series[series > 0]
    return series


def create_fig_data(row):
    row_df = world_df.loc[row]
    series = row_to_series(row_df)
    return dict(x=series.index, y=series,
                type='scatter', name=get_label(row_df))


def create_rate_data(row, window):
    row_df = world_df.loc[row]
    series = row_to_series(row_df)
    series = series.rolling(window).apply(get_rate)
    return dict(x=series.index, y=series,
                type='scatter', name=get_label(row_df))


@app.callback(
        Output('total-cases-1', 'figure'),
        [Input('dropdown', 'value')],
        [State('total-cases-1', 'figure')])
def update_output(values, fig):
    data = []
    for value in values:
        data.append(create_fig_data(value))

    fig['data'] = data
    return fig


@app.callback(
        Output('rate', 'figure'),
        [Input('dropdown', 'value')],
        [State('rate', 'figure')])
def update_rate(values, fig):
    data = []
    for value in values:
        data.append(create_rate_data(value, '3d'))

    fig['data'] = data
    return fig


@app.callback(
        Output('rate7', 'figure'),
        [Input('dropdown', 'value')],
        [State('rate7', 'figure')])
def update_rate7(values, fig):
    data = []
    for value in values:
        data.append(create_rate_data(value, '7d'))

    fig['data'] = data
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
