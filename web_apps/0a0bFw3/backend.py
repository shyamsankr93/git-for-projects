import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import dataiku
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from tzlocal import get_localzone

news_real_time = dataiku.Dataset("news_today_scored")
news_real_time_df = news_real_time.get_dataframe()

all_news_scored = dataiku.Dataset("news_data_all_scored")
all_news_scored_df = all_news_scored.get_dataframe()

tickers = dataiku.Dataset("tickers_information")
tickers_df = tickers.get_dataframe()

stock_prices = dataiku.Dataset("stock_prices_all")
stock_prices_df = stock_prices.get_dataframe()

news_today = dataiku.Dataset("news_today_grouped_scored_name")
news_today_df = news_today.get_dataframe()

anomalies = dataiku.Dataset("anomalies")
anomalies_df = anomalies.get_dataframe()

stock_prices_anomaly = pd.merge(stock_prices_df, anomalies_df, how='left', left_on=['ticker', 'date'], right_on=['ticker', 'date'])
stock_prices_anomaly['is_anomaly'] = [True if anomaly==True else False for anomaly in stock_prices_anomaly['is_anomaly']]

app.config.external_stylesheets = [dbc.themes.BOOTSTRAP]

app.layout = dbc.Container(
    [
        html.H1("Stock Alert System"),
        html.Hr(),
        dcc.Tabs(id='tab-id', value='real-time-tab', 
             children=[
                dcc.Tab(label='Real Time News Scoring', value='real-time-tab'),
                dcc.Tab(label='Case Study', value='histo-anomaly'),
                dcc.Tab(label='Historical Price Anomaly Detection', value='histo-price-tab'),
                dcc.Tab(label='Historical News Scoring', value='histo-news-tab')
             ]),
        dbc.Container(id='tab-content')
    ],
    fluid=True)

@app.callback(Output('tab-content', 'children'),
              Input('tab-id', 'value'))
def render_content(tab):
    if tab == 'real-time-tab':
        return layout_real_time
    elif tab == 'histo-anomaly':
        return layout_histo_anomaly
    elif tab == 'histo-price-tab':
        return layout_histo_price
    elif tab == 'histo-news-tab':
        return layout_histo_news

@app.callback(
        Output('top-scores', 'active_cell'),
        Input('stock-focus', 'value'),
)
def update_active_cell(stock):
    scores = news_today_df.sort_values(by='proba_true', ascending=False)
    selected_row = np.where(scores['stock']==stock)[0][0]
    if selected_row is None:
        selected_row = 0
    active_cell = {
      "row": selected_row,
      "column": 0,
      "row_id": stock,
      "column_id": "Stock"
    }
    return active_cell

def top_scores():
    scores = news_today_df.sort_values(by='proba_true', ascending=False)
    table = dash_table.DataTable(
        id="top-scores",
        columns=(
                [{'id': 'Stock', 'name': 'Stock'},
                 {'id': 'Company', 'name': 'Company'},
                 {'id': 'Risk', 'name': 'Volatility Score'}]
             ),
        data=[{'Stock': scores['stock'].iloc[i],
               'Company': scores['company'].iloc[i],
              'Risk': round(scores['proba_true'].iloc[i]*100)}
              for i in range(len(scores))],
    style_header={'fontWeight': 'bold'},
    style_table={'height': '300px',
                 "overflowY": "scroll"},
    style_cell={'textAlign': 'left',
               'padding': '5px',
               'whiteSpace': 'normal',
               'height': 'auto'},
    sort_action="native",
    sort_mode="multi",
    page_action='none',
    editable=False)
    return table

@app.callback(Output('real-time-news-container', 'children'),
              Input('top-scores', 'active_cell'),
              Input('top-scores', 'derived_virtual_data'))
def news_scores(value, data):
    if value:
        stock = data[value['row']]['Stock']
        containStock = [stock in n.split(',') for n in news_real_time_df['stocks']]
        scores = news_real_time_df[containStock].sort_values(by='proba_true', ascending=False)
    else:
        scores = news_real_time_df.sort_values(by='proba_true', ascending=False)
    table = dash_table.DataTable(
        id="real-time-news",
        columns=(
                [{'id': 'Title', 'name': 'Title', 'presentation': 'markdown'},
                 {'id': 'Stocks', 'name': 'Stocks'},
                 {'id': 'Risk', 'name': 'Volatility Score'}]
             ),
        data=[{'Title': '[' + scores['title'].iloc[i] +
                        '](' + scores['url'].iloc[i] + ')',
              'Stocks': scores['stocks'].iloc[i],
              'Risk': round(scores['proba_true'].iloc[i]*100)}
              for i in range(len(scores))],
    style_header={'fontWeight': 'bold'},
    style_table={"overflowY": "scroll"},
    style_cell={'textAlign': 'left',
               'padding': '5px',
               'whiteSpace': 'normal',
                'height': 'auto'},
    sort_action="native",
    sort_mode="multi",
    editable=False)
    return table

layout_real_time = dbc.Row([
                dbc.Container([
                        html.H3('Real Time Alerts'),
                        html.H4('Last Update: ' + datetime.fromtimestamp(news_real_time.get_last_metric_values().raw['metrics'][0]['lastValues'][0]['computed']/1000).astimezone(get_localzone()).strftime('%Y-%m-%d %H:%M:%S')),
                        html.P('A volatility score is computed using the agregated news titles that have been retrieved today. ' +
                               'The value of the score is between 0 and 100, the higher the score, the more likely the stock could exhibit and anomalous behaviour. ' +
                               'A news scoring model learned on historical data outputs a score for each individual stock and they are displayed according to their level in this table.'),
                        dcc.Dropdown(
                            id='stock-focus',
                            options=[
                            {'label': stock, 
                             'value': stock}
                             for stock in tickers_df['ticker']],
                            value=news_today_df.sort_values(by='proba_true', ascending=False).head(1)['stock'].iloc[0]
                            ),
                        top_scores()
                ]),
                dbc.Container([
                        html.H3('Real Time News Score'),
                        html.P('Individual news are also scored using the same model. ' +
                               'Each of them can relate to one or multiple stocks. ' +
                               'When selecting a line in the Real Time Alerts table, the news are filtered to display those regarding this specific stock.'),
                        dbc.Container(id='real-time-news-container')
                ])
])

layout_histo_price = dbc.Container([
            html.H3("Price Evolution"),
            html.P("Daily close prices are plotted for the stock selected. " + 
                   "The time window can be adjusted to focus on a specific period. " +
                   "Vertical lines indicate the detection of an anomaly, there can be none or a few, depending on the stock. " +
                   "An anomaly is defined as a move that is deemed unlikely with respect to the moves of the other stocks and their historical returns."),
            dbc.Row([
                dbc.Col(
            dcc.Dropdown(
                    id='Stock',
                options=[
            {'label': stock, 
             'value': stock}
             for stock in tickers_df['ticker']],
        value=news_today_df.sort_values(by='proba_true', ascending=False).head(1)['stock'].iloc[0]
            ), md=6),
                dbc.Col(
            dcc.DatePickerRange(
                id='price-date-picker-range',
                min_date_allowed=stock_prices_df['date'].min(),
                max_date_allowed=stock_prices_df['date'].max(),
                initial_visible_month=stock_prices_df['date'].max(),
                start_date=stock_prices_df['date'].min(),
                end_date=stock_prices_df['date'].max()
            ), md=6)]),
            dcc.Graph(id="stock-price-evolution")
            ])

def anomalies_table():
    anomalies = anomalies_df.sort_values(by='date', ascending=False)
    table = dash_table.DataTable(
        id="anomalies",
        columns=(
                [{'id': 'Date', 'name': 'Date'},
                 {'id': 'Stock', 'name': 'Stock'}]
             ),
        data=[{'Date': anomalies['date'].iloc[i].strftime('%Y-%m-%d'),
               'Stock': anomalies['ticker'].iloc[i]}
              for i in range(len(anomalies))],
    style_header={'fontWeight': 'bold'},
    style_table={'height': '300px',
                 "overflowY": "scroll"},
    style_cell={'textAlign': 'left',
               'padding': '5px',
               'whiteSpace': 'normal',
               'height': 'auto'},
    sort_action="native",
    sort_mode="multi",
    page_action='none',
    editable=False)
    return table

layout_histo_anomaly = dbc.Container([
    html.H3("Past Anomaly Analysis"),
    html.P("An anomaly is defined as a move that is deemed unlikely with respect to the moves of the other stocks and their historical returns. " + 
           "This table compiles anomalies detected in the historical dataset. " +
           "These are the labels that are processed in the News Scoring Model learning."),
            anomalies_table(),
            dcc.Graph(id="stock-price-focus"),
            dbc.Container(id="news-focus")
            ])

layout_histo_news = dbc.Container([
            html.H3("Most Impactful News"),
    html.P("Historical Individual News are scored using the News Scoring Model. " + 
           "The default sorting is by descending volatility score, but data can be sorted by date for example. " +
           "These news titles are the raw data processed in the News Scoring Model to predict anomalies and compute scores."),
            dcc.Dropdown(
                    id='Stock-News',
                options=[
            {'label': stock, 
             'value': stock}
             for stock in tickers_df['ticker']],
        value=news_today_df.sort_values(by='proba_true', ascending=False).head(1)['stock'].iloc[0]
            ),
            dbc.Container(id="top-scores-news-history")
            ])

@app.callback(Output('top-scores-news-history', 'children'),
              Input('Stock-News', 'value'))
def render_content(stock):
    scores = all_news_scored_df[all_news_scored_df['stock']==stock].sort_values(by='proba_true', ascending=False)
    table = dash_table.DataTable(
        id="historical-news",
        columns=(
                [{'id': 'Date', 'name': 'Date'},
                 {'id': 'Title', 'name': 'Title', 'presentation': 'markdown'},
                 {'id': 'Risk', 'name': 'Volatility Score'}]
             ),
        data=[{'Date': scores['date'].iloc[i].strftime('%Y-%m-%d'),
               'Title': '[' + scores['title'].iloc[i] +
                        '](' + scores['url'].iloc[i] + ')',
              'Risk': round(scores['proba_true'].iloc[i]*100)}
              for i in range(len(scores))],
    style_header={'fontWeight': 'bold'},
    style_table={"overflowY": "scroll"},
    style_cell={'textAlign': 'left',
               'padding': '5px',
               'whiteSpace': 'normal',
                'height': 'auto'},
    sort_action="native",
    sort_mode="multi",
    editable=False)
    return table


@app.callback(Output('news-focus', 'children'),
              Input('anomalies', 'active_cell'),
              Input('anomalies', 'derived_virtual_data')
)
def render_content(value, data):
    data_row = data[value['row']]
    stock = data_row['Stock']
    date = data_row['Date']
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=3)).strftime('%Y-%m-%d')
    scores = all_news_scored_df[all_news_scored_df['stock']==stock][all_news_scored_df['date']>=start_date][all_news_scored_df['date']<=end_date].sort_values(by='date', ascending=True)
    table = dash_table.DataTable(
        id="focus-news",
        columns=(
                [{'id': 'Date', 'name': 'Date'},
                 {'id': 'Title', 'name': 'Title', 'presentation': 'markdown'},
                 {'id': 'Risk', 'name': 'Volatility Score'}]
             ),
        data=[{'Date': scores['date'].iloc[i].strftime('%Y-%m-%d'),
               'Title': '[' + scores['title'].iloc[i] +
                        '](' + scores['url'].iloc[i] + ')',
              'Risk': round(scores['proba_true'].iloc[i]*100)}
              for i in range(len(scores))],
    style_header={'fontWeight': 'bold'},
    style_table={"overflowY": "scroll"},
    style_cell={'textAlign': 'left',
               'padding': '5px',
               'whiteSpace': 'normal',
                'height': 'auto'},
    sort_action="native",
    sort_mode="multi",
    editable=False)
    return table


@app.callback(
    Output("stock-price-focus", "figure"),
    Input('anomalies', 'active_cell'),
    Input('anomalies', 'derived_virtual_data')
)
def make_graph(value, data):
    data_row = data[value['row']]
    stock = data_row['Stock']
    date = data_row['Date']
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10)).strftime('%Y-%m-%d')
    data = stock_prices_anomaly[stock_prices_anomaly['ticker']==stock][stock_prices_anomaly['date']>=start_date][stock_prices_anomaly['date']<=end_date].sort_values(by='date', ascending=True)
    data_anomaly = data[data['is_anomaly']==True]
    fig = px.line(data, x='date', y='price',
                 labels={
                     "date": "Date",
                     "price": "Price"
                 },
                title="Stock Price")
    fig.add_vline(x=date)
    return fig

@app.callback(
    Output("stock-price-evolution", "figure"),
    Input("Stock", "value"),
    Input('price-date-picker-range', 'start_date'),
    Input('price-date-picker-range', 'end_date')
)
def make_graph(stock, start_date, end_date):
    anomalies_stock = anomalies_df[anomalies_df['ticker']==stock][anomalies_df['date']>=start_date][anomalies_df['date']<=end_date]
    data = stock_prices_df[stock_prices_df['ticker']==stock][stock_prices_df['date']>=start_date][stock_prices_df['date']<=end_date]
    fig = px.line(data, x='date', y='price',
                 labels={
                     "date": "Date",
                     "price": "Price"
                 },
                title="Stock Price")
    for d in list(anomalies_stock['date']):
        fig.add_vline(x=d)
    return fig