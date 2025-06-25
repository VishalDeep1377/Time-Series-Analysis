import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('Script started.')

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# List of tickers (add/remove as needed)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'JPM', 'NFLX', '^NSEI', '^GSPC']

print('Starting data download...')
try:
    data = yf.download(tickers, start='2014-01-01', end='2024-01-01', group_by='ticker', auto_adjust=True)
    print('Data download complete.')
    print(f'Downloaded data shape: {data.shape}')
    if data.empty:
        print('No data was downloaded. Exiting.')
        exit(1)
except Exception as e:
    print(f'Error during data download: {e}')
    exit(1)

# Save each stock as a separate CSV
for ticker in tickers:
    print(f'Processing {ticker}...')
    try:
        df = data[ticker].copy()
        print(f'{ticker} data shape: {df.shape}')
        df['Ticker'] = ticker
        df.reset_index(inplace=True)
        file_path = f'data/{ticker}_10y.csv'
        df.to_csv(file_path, index=False)
        print(f'Saved {file_path} ({len(df)} rows)')
    except Exception as e:
        print(f'Error saving {ticker}: {e}')

print('Combining all CSVs...')
# Combine all into one DataFrame for Power BI and EDA
try:
    all_data = pd.concat([pd.read_csv(f'data/{ticker}_10y.csv') for ticker in tickers])
    all_data.to_csv('data/all_stocks_10y.csv', index=False)
    print('Combined all stocks into data/all_stocks_10y.csv')
except Exception as e:
    print(f'Error combining CSVs: {e}')
    exit(1)

# --- Initial EDA ---
print('Data Info:')
print(all_data.info())
print('\nSummary Statistics:')
print(all_data.describe())
print('\nTicker Counts:')
print(all_data['Ticker'].value_counts())
print('\nMissing Values:')
print(all_data.isnull().sum())

# Plot closing prices for all stocks
try:
    plt.figure(figsize=(14, 7))
    for ticker in all_data['Ticker'].unique():
        plt.plot(all_data[all_data['Ticker'] == ticker]['Date'],
                 all_data[all_data['Ticker'] == ticker]['Close'], label=ticker)
    plt.legend()
    plt.title('Closing Prices of All Stocks (2014-2024)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.savefig('data/closing_prices_all_stocks.png')
    plt.close()
    print('Saved data/closing_prices_all_stocks.png')
except Exception as e:
    print(f'Error plotting closing prices: {e}')

# Correlation heatmap (last available day for each stock)
try:
    pivot = all_data.pivot_table(index='Date', columns='Ticker', values='Close')
    corr = pivot.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation of Closing Prices')
    plt.tight_layout()
    plt.savefig('data/correlation_heatmap.png')
    plt.close()
    print('Saved data/correlation_heatmap.png')
except Exception as e:
    print(f'Error plotting correlation heatmap: {e}')

print('EDA complete. Check the data/ directory for outputs.')
