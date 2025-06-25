import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Load the enhanced dataset
df = pd.read_csv('data/all_stocks_10y_features.csv')
tickers = df['Ticker'].unique()
forecast_horizons = [7, 30, 90, 180]

for ticker in tickers:
    print(f'Processing {ticker}...')
    df_stock = df[df['Ticker'] == ticker][['Date', 'Close']].copy()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values('Date')
    df_stock.set_index('Date', inplace=True)

    for horizon in forecast_horizons:
        print(f'  Forecast horizon: {horizon} days')
        # Split into train/test (last {horizon} days for test)
        train = df_stock.iloc[:-horizon]
        test = df_stock.iloc[-horizon:]

        # --- ARIMA Model ---
        print(f'    Training ARIMA model for {ticker}...')
        model_arima = ARIMA(train['Close'], order=(5,1,0))
        model_arima_fit = model_arima.fit()
        forecast_arima = model_arima_fit.forecast(steps=len(test))

        # --- Prophet Model ---
        print(f'    Training Prophet model for {ticker}...')
        df_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        model_prophet = Prophet(daily_seasonality=True)
        model_prophet.fit(df_prophet)
        future = model_prophet.make_future_dataframe(periods=len(test))
        forecast_prophet = model_prophet.predict(future)

        # --- Plotting ---
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train['Close'], label='Train', color='blue')
        plt.plot(test.index, test['Close'], label='Test', color='black')
        plt.plot(test.index, forecast_arima, label='ARIMA Forecast', color='red', linestyle='--')
        plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast', color='green', linestyle='--', alpha=0.7)
        plt.title(f'{ticker} Close Price Forecast (ARIMA & Prophet, {horizon} days)')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.tight_layout()
        os.makedirs('data/model_outputs', exist_ok=True)
        plt.savefig(f'data/model_outputs/{ticker}_arima_prophet_forecast_{horizon}.png')
        plt.close()

        # --- Save Results for Power BI/Streamlit ---
        results = pd.DataFrame({
            'Date': test.index,
            'Actual': test['Close'].values,
            'ARIMA_Forecast': forecast_arima.values,
            'Prophet_Forecast': forecast_prophet.iloc[-len(test):]['yhat'].values
        })
        results.to_csv(f'data/model_outputs/{ticker}_arima_prophet_results_{horizon}.csv', index=False)
        print(f'    Results saved as data/model_outputs/{ticker}_arima_prophet_results_{horizon}.csv')

    print(f'ARIMA and Prophet modeling complete for {ticker}!')
    print(f'Forecast plot saved as data/model_outputs/{ticker}_arima_prophet_forecast_{horizon}.png')
    print(f'Results saved as data/model_outputs/{ticker}_arima_prophet_results_{horizon}.csv') 