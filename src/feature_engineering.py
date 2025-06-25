import pandas as pd
import numpy as np
import os

# Load the combined stock data
df = pd.read_csv('data/all_stocks_10y.csv')

# --- Feature Engineering Functions ---
def add_technical_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    # Volatility (Rolling Std Dev)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    return df

# Apply indicators for each stock separately
enhanced_df = []
for ticker in df['Ticker'].unique():
    temp = df[df['Ticker'] == ticker].copy()
    temp = add_technical_indicators(temp)
    enhanced_df.append(temp)

df_features = pd.concat(enhanced_df)

# Save the enhanced dataset
os.makedirs('data', exist_ok=True)
df_features.to_csv('data/all_stocks_10y_features.csv', index=False)

print('Feature engineering complete! Enhanced dataset saved as data/all_stocks_10y_features.csv') 