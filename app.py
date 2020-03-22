#!/usr/bin/env python3
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json

app = dash.Dash(__name__)

server = app.server

la_df = pd.read_json('data/los_angeles.json')
new_cases_s = la_df['New Cases']
total_dat = np.empty_like(new_cases_s)

total = 0
for (i, val) in enumerate(new_cases_s):
    total += val
    total_dat[i] = total

la_df['Total Cases'] = total_dat

fig = go.Figure(data=[go.Scatter(x=la_df['Date'], y=total_dat)])

app.layout = html.Div([
    html.H2('Coronalyzer'),
    dcc.Graph(
        id='cases',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
