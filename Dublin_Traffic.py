#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pyodbc
data = pd.read_csv('dublinJ (1).csv')
data['Times'] = pd.to_datetime(data['Times'])
data = data.set_index('Times')

# Create the Dash app
app = dash.Dash(__name__)
server = app.server
#Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True
# Set the layout for the app
app.layout = html.Div([
    html.H1("Dublin City Centre Traffic Count Predictions",
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': '24px'}),
    
    html.Head([
        html.Script(
            '''
            <!-- Global site tag (gtag.js) - Google Analytics -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=YOUR_TRACKING_ID"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-7KHT3ZRKTL');
            </script>
            '''
        )
    ]),

    html.Div([
        html.H2("Zone:", style={'margin-right': '2em'}),
        dcc.Dropdown(
            id="zone-dropdown",
            options=[
                {"label": zone, "value": zone} for zone in data["Zone"].unique()
            ],
            placeholder="Select a zone",
            value=None,
            style={'width': '90%', 'padding': '2px', 'font-size': '20px', 'text-align-last': 'center'}
        ),
    ], style={'display': 'flex'}),

    html.Div([
        html.Div([
            html.H2("Count Type:", style={'margin-right': '2em'}),
        ]),
        dcc.RadioItems(
            id="count-type-radio",
            options=[
                {"label": "Incount", "value": "InCount"},
                {"label": "Outcount", "value": "OutCount"}
            ],
            value=None,
            style={'display': 'flex'}
        ),
    ], style={'display': 'inline-block', 'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.H2("Prediction Interval:")
        ]),
        dcc.RadioItems(
            id="prediction-interval-radio",
            options=[
                {"label": "Hourly", "value": "hourly"},
                {"label": "Daily", "value": "daily"},
                {"label": "Weekly", "value": "weekly"},
                {"label": "Monthly", "value": "monthly"},
                {"label": "Quarterly", "value": "quarterly"},
                {"label": "Yearly", "value": "yearly"}
            ],
            value=None,
            style={'display': 'flex'}
        ),
    ], style={'display': 'inline-block', 'margin-bottom': '20px'}),

    html.Button("Generate Graph", id="update-button", n_clicks=0,
                style={'margin': '20px auto', 'display': 'block'}),

    html.Div([
        html.Div([
            html.H3("Trend", style={'textAlign': 'center'})
        ]),
        dcc.Graph(id="trend-graph")
    ]),

    html.Div([
        html.Div([
            html.H3("Seasonal", style={'textAlign': 'center'})
        ]),
        dcc.Graph(id="seasonal-graph")
    ]),

    html.Div([
        html.Div([
            html.H3("Residual", style={'textAlign': 'center'})
        ]),
        dcc.Graph(id="residual-graph")
    ]),

    html.Div([
        html.Div([
            html.H3("ARIMA and SARIMA", style={'textAlign': 'center'})
        ]),
        dcc.Graph(id="arima-sarima-actual-graph")
    ]),
])

# Define a variable to store the directory path for saving results
results_dir = "results"

# Define a function to load or calculate the predictions
def get_predictions(zone, count_type, prediction_interval):
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result_file = f"{results_dir}/{zone}_{count_type}_{prediction_interval}.joblib"

    if os.path.exists(result_file):
        # Load results from file
        trend, seasonal, residual, forecast_arima, forecast_sarima = joblib.load(result_file)
    else:
        # Filter the data based on selected zone and count type
        filtered_data = data[(data['Zone'] == zone) & (data[count_type] > 0)]

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(filtered_data[count_type], model='additive', period=24)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Perform forecasting using ARIMA
        model = ARIMA(filtered_data[count_type], order=(2, 0, 3))
        results = model.fit()
        forecast_arima = results.predict(
            start=filtered_data.index[-1], end=filtered_data.index[-1] + pd.DateOffset(hours=prediction_interval))

        # Perform forecasting using SARIMA
        model_sarima = auto_arima(filtered_data[count_type], seasonal=True, m=24)
        model_sarima.fit(filtered_data[count_type])
        forecast_sarima = model_sarima.predict(n_periods=prediction_interval)

        # Save results to file
        results = (trend, seasonal, residual, forecast_arima, forecast_sarima)
        joblib.dump(results, result_file)

    return trend, seasonal, residual, forecast_arima, forecast_sarima

# Define callback functions for updating the graphs and metrics
@app.callback(
    Output("trend-graph", "figure"),
    Output("seasonal-graph", "figure"),
    Output("residual-graph", "figure"),
    Output("arima-sarima-actual-graph", "figure"),
    Input("update-button", "n_clicks"),
    [State('zone-dropdown', 'value'),
     State('count-type-radio', 'value'),
     State('prediction-interval-radio', 'value')]
)

def update_graphs(n_clicks, zone, count_type, prediction_interval):
    
    filtered_data = data[(data['Zone'] == zone) & (data[count_type] > 0)]
    
    # Prediction interval
    if prediction_interval == "hourly":
        prediction_interval = 1
    elif prediction_interval == "daily":
        prediction_interval = 24
    elif prediction_interval == "weekly":
        prediction_interval = 7 * 24
    elif prediction_interval == "monthly":
        prediction_interval = 30*24
    elif prediction_interval == "quarterly":
        prediction_interval = 3*30*24
    elif prediction_interval == "yearly":
        prediction_interval = 365*24

    trend, seasonal, residual, forecast_arima, forecast_sarima = get_predictions(zone, count_type, prediction_interval)

    # Create trend graph
    trend_graph = go.Figure()
    trend_graph.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[count_type], name="Actual"))
    trend_graph.add_trace(go.Scatter(x=trend.index, y=trend, name="Trend"))
    trend_graph.update_layout(title=f'Trend Analysis- Zone: {zone}, Count Type: {count_type}',
                              xaxis_title="Time",
                              yaxis_title="Count",
                              showlegend=True,
                              plot_bgcolor="rgba(0, 0, 0, 0)",
                              paper_bgcolor="rgba(0, 0, 0, 0)",
                              xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=False))

    # Create seasonal graph
    seasonal_graph = go.Figure()
    seasonal_graph.add_trace(go.Scatter(x=seasonal.index, y=seasonal, name="Seasonality"))
    seasonal_graph.update_layout(title=f'Seasonal Analysis- Zone: {zone}, Count Type: {count_type}',
                                 xaxis_title="Time",
                                 yaxis_title="Count",
                                 showlegend=True,
                                 plot_bgcolor="rgba(0, 0, 0, 0)",
                                 paper_bgcolor="rgba(0, 0, 0, 0)",
                                 xaxis=dict(showgrid=False),
                                 yaxis=dict(showgrid=False))

    # Create residual graph
    residual_graph = go.Figure()
    residual_graph.add_trace(go.Scatter(x=residual.index, y=residual, name="Residuals"))
    residual_graph.update_layout(title=f'Residual Analysis - Zone: {zone}, Count Type: {count_type}',
                                 xaxis_title="Time",
                                 yaxis_title="Count",
                                 showlegend=True,
                                 plot_bgcolor="rgba(0, 0, 0, 0)",
                                 paper_bgcolor="rgba(0, 0, 0, 0)",
                                 xaxis=dict(showgrid=False),
                                 yaxis=dict(showgrid=False))

    # Create ARIMA and SARIMA actual vs predicted graph
    actual_graph = go.Figure()
    actual_graph.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[count_type], name="Actual"))
    actual_graph.add_trace(go.Scatter(x=forecast_arima.index, y=forecast_arima, name="ARIMA Forecast"))
    actual_graph.add_trace(go.Scatter(x=forecast_sarima.index, y=forecast_sarima, name="SARIMA Forecast"))
    actual_graph.update_layout(title=f'SARIMA and ARIMA Forecast - Zone: {zone}, Count Type: {count_type}, Prediction Interval: {prediction_interval} hrs',
                               xaxis_title="Time",
                               yaxis_title="Count",
                               showlegend=True,
                               plot_bgcolor="rgba(0, 0, 0, 0)",
                               paper_bgcolor="rgba(0, 0, 0, 0)",
                               xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=False))

    return trend_graph, seasonal_graph, residual_graph, actual_graph

if __name__ == '__main__':
    app.run_server(debug=True)



# In[ ]:


# In[ ]:




