import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the enhanced dataset
DF_PATH = 'data/all_stocks_10y_features.csv'
df = pd.read_csv(DF_PATH)
tickers = df['Ticker'].unique()
forecast_horizons = [7, 30, 90, 180]

# LSTM parameters
time_steps = 30  # window size for input sequence
EPOCHS = 30
BATCH_SIZE = 32

for ticker in tickers:
    print(f'Processing {ticker}...')
    df_stock = df[df['Ticker'] == ticker][['Date', 'Close']].copy()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values('Date')
    df_stock.set_index('Date', inplace=True)
    data = df_stock['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    for horizon in forecast_horizons:
        print(f'  Forecast horizon: {horizon} days')
        train_scaled = data_scaled[:-horizon]
        test_scaled = data_scaled[-(horizon+time_steps):]  # for rolling prediction
        test_dates = df_stock.index[-horizon:]
        # Prepare training sequences
        X_train, y_train = [], []
        for i in range(time_steps, len(train_scaled)):
            X_train.append(train_scaled[i-time_steps:i, 0])
            y_train.append(train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(time_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])
        # Rolling forecast
        inputs = data_scaled[-(horizon+time_steps):].copy()
        X_test = [inputs[i:i+time_steps, 0] for i in range(horizon)]
        X_test = np.array(X_test).reshape((horizon, time_steps, 1))
        lstm_forecast_scaled = model.predict(X_test, verbose=0)
        lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)
        # Save results
        actual = data[-horizon:].flatten()
        results = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'LSTM_Forecast': lstm_forecast.flatten()
        })
        os.makedirs('data/model_outputs', exist_ok=True)
        results.to_csv(f'data/model_outputs/{ticker}_lstm_results_{horizon}.csv', index=False)
        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(df_stock.index[-(horizon+time_steps):-horizon], scaler.inverse_transform(data_scaled[-(horizon+time_steps):-horizon]), label='Train', color='blue')
        plt.plot(test_dates, actual, label='Test', color='black')
        plt.plot(test_dates, lstm_forecast.flatten(), label='LSTM Forecast', color='orange', linestyle='--')
        plt.title(f'{ticker} Close Price Forecast (LSTM, {horizon} days)')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'data/model_outputs/{ticker}_lstm_forecast_{horizon}.png')
        plt.close()
        print(f'    Results saved as data/model_outputs/{ticker}_lstm_results_{horizon}.csv')
        print(f'    Forecast plot saved as data/model_outputs/{ticker}_lstm_forecast_{horizon}.png')
    print(f'LSTM modeling complete for {ticker}!')
