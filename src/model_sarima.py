import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        # --- SARIMA Model ---
        print(f'    Training SARIMA model for {ticker}...')
        model_sarima = SARIMAX(train['Close'], order=(2,1,2), seasonal_order=(1,1,1,5))
        model_sarima_fit = model_sarima.fit(disp=False)
        forecast_sarima = model_sarima_fit.forecast(steps=len(test))

        # --- Plotting ---
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train['Close'], label='Train', color='blue')
        plt.plot(test.index, test['Close'], label='Test', color='black')
        plt.plot(test.index, forecast_sarima, label='SARIMA Forecast', color='magenta', linestyle='--')
        plt.title(f'{ticker} Close Price Forecast (SARIMA, {horizon} days)')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.tight_layout()
        os.makedirs('data/model_outputs', exist_ok=True)
        plt.savefig(f'data/model_outputs/{ticker}_sarima_forecast_{horizon}.png')
        plt.close()

        # --- Save Results for Power BI/Streamlit ---
        results = pd.DataFrame({
            'Date': test.index,
            'Actual': test['Close'].values,
            'SARIMA_Forecast': forecast_sarima.values
        })
        results.to_csv(f'data/model_outputs/{ticker}_sarima_results_{horizon}.csv', index=False)
        print(f'    Results saved as data/model_outputs/{ticker}_sarima_results_{horizon}.csv')

    print(f'SARIMA modeling complete for {ticker}!')
    print(f'Forecast plots saved as data/model_outputs/{ticker}_sarima_forecast_*.png')
    print(f'Results saved as data/model_outputs/{ticker}_sarima_results_*.csv') 