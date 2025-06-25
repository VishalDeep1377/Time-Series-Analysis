import pandas as pd
import os

forecast_horizons = [7, 30, 90, 180]

def merge_for_ticker_and_horizon(ticker, horizon):
    arima_prophet_path = f'data/model_outputs/{ticker}_arima_prophet_results_{horizon}.csv'
    sarima_path = f'data/model_outputs/{ticker}_sarima_results_{horizon}.csv'
    if not os.path.exists(arima_prophet_path) or not os.path.exists(sarima_path):
        print(f'Skipping {ticker} ({horizon}d): missing ARIMA/Prophet or SARIMA results.')
        return
    arima_prophet = pd.read_csv(arima_prophet_path)
    sarima = pd.read_csv(sarima_path)
    merged = arima_prophet.merge(sarima[['Date', 'SARIMA_Forecast']], on='Date', how='left')
    merged.to_csv(f'data/model_outputs/{ticker}_all_models_results_{horizon}.csv', index=False)
    print(f'Merged model results saved as data/model_outputs/{ticker}_all_models_results_{horizon}.csv')

# Get all tickers from features file
features = pd.read_csv('data/all_stocks_10y_features.csv')
tickers = features['Ticker'].unique()
for ticker in tickers:
    for horizon in forecast_horizons:
        merge_for_ticker_and_horizon(ticker, horizon) 