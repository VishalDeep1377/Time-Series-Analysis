import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

st.set_page_config(page_title='Advanced Stock Analysis & Forecasting', layout='wide')

# --- Custom CSS for a Professional Look ---
st.markdown('''
    <style>
    body, .stApp {
        background-color: #18191A !important;
        color: #F5F6F7 !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #00BFFF !important;
        color: white !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-weight: 600;
        padding: 0.5em 2em;
        margin: 0.5em 0;
    }
    .stSidebar, .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj, .css-1cypcdb {
        background-color: #22232A !important;
        color: #F5F6F7 !important;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    .stSelectbox>div>div {
        background-color: #242526 !important;
        color: #F5F6F7 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton>button {
        background-color: #242526 !important;
        color: #00BFFF !important;
        border-radius: 8px !important;
        font-weight: 600;
        border: 1px solid #00BFFF !important;
    }
    .stMetric {
        background: #22232A !important;
        border-radius: 8px;
        padding: 1em;
        color: #00BFFF !important;
        font-weight: 700;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #00BFFF !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stPlotlyChart {
        background: #18191A !important;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        padding: 1em;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
''', unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_features():
    return pd.read_csv('data/all_stocks_10y_features.csv')

@st.cache_data
def load_model_results(ticker):
    path = f'data/model_outputs/{ticker}_all_models_results.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def get_available_tickers():
    df = load_features()
    return sorted(df['Ticker'].unique())

# --- Company Name Mapping ---
ticker_to_name = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc. (Facebook)',
    'JPM': 'JPMorgan Chase & Co.',
    'NFLX': 'Netflix, Inc.',
    '^NSEI': 'Nifty 50 Index',
    '^GSPC': 'S&P 500 Index',
}

# --- Sidebar ---
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to:', ['EDA & Indicators', 'Forecasting', 'Summary & Insights', 'How to Read Charts'])

st.sidebar.markdown('---')

# Build a list of display names for the selectbox
available_tickers = get_available_tickers()
ticker_display_names = [f"{ticker_to_name.get(t, t)} ({t})" for t in available_tickers]
selected_display = st.sidebar.selectbox('Select Stock', ticker_display_names, index=0)
# Extract ticker from display name
ticker = available_tickers[ticker_display_names.index(selected_display)]
company_name = ticker_to_name.get(ticker, ticker)

# --- Forecast Horizon Selection ---
st.sidebar.markdown('---')
st.sidebar.markdown('**Select Forecast Horizon (days)**')
forecast_horizons = [7, 30, 90, 180]
horizon = st.sidebar.selectbox('Forecast Horizon', forecast_horizons, index=forecast_horizons.index(30))

# --- Main Content ---
st.title('Advanced Stock Analysis & Forecasting App')
st.markdown('''
This app provides advanced, interactive stock analysis and forecasting using ARIMA, Prophet, and SARIMA models. 
Select a stock and explore technical indicators, model predictions, and actionable insights. 
''')

# --- Load Data ---
df = load_features()
df_stock = df[df['Ticker'] == ticker].copy()

def load_model_results_horizon(ticker, horizon):
    path = f'data/model_outputs/{ticker}_all_models_results_{horizon}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model_results = load_model_results_horizon(ticker, horizon)

# --- EDA & Indicators Page ---
if page == 'EDA & Indicators':
    st.header(f'Exploratory Data Analysis & Technical Indicators: {company_name} ({ticker})')
    st.markdown('''
    This section helps you understand the stock's behavior using popular technical indicators. Each chart includes a tip on what to look for.
    ''')

    # --- Close Price with SMA & EMA ---
    st.subheader('1. Close Price with SMA & EMA')
    st.markdown('''
    **Shows the trend direction.** If the price is above the moving averages, the stock is generally trending up.
    
    **What to look for:**
    - Price crossing above SMA/EMA: possible uptrend (buy signal)
    - Price crossing below: possible downtrend (sell signal)
    ''')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['SMA_50'], mode='lines', name='SMA 50', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['EMA_20'], mode='lines', name='EMA 20', line=dict(color='red')))
    fig.update_layout(title='Close Price with SMA & EMA', xaxis_title='Date', yaxis_title='Price', legend_title='Legend',
                     font=dict(size=14), plot_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- RSI ---
    st.subheader('2. RSI (Relative Strength Index)')
    st.markdown('''
    **Measures if a stock is overbought or oversold.**
    
    **What to look for:**
    - RSI above 70: overbought (stock may decrease soon)
    - RSI below 30: oversold (stock may increase soon)
    ''')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['RSI_14'], mode='lines', name='RSI 14', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)', annotation_position='top left')
    fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)', annotation_position='bottom left')
    fig.update_layout(title='RSI (14)', xaxis_title='Date', yaxis_title='RSI', font=dict(size=14), plot_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- MACD ---
    st.subheader('3. MACD (Moving Average Convergence Divergence)')
    st.markdown('''
    **Shows momentum and trend changes.**
    
    **What to look for:**
    - MACD crossing above Signal Line: possible uptrend (buy signal)
    - MACD crossing below: possible downtrend (sell signal)
    ''')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
    fig.update_layout(title='MACD', xaxis_title='Date', yaxis_title='MACD', font=dict(size=14), plot_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- Bollinger Bands ---
    st.subheader('4. Bollinger Bands')
    st.markdown('''
    **Shows volatility and price extremes.**
    
    **What to look for:**
    - Price touching upper band: stock may be overbought (could decrease)
    - Price touching lower band: stock may be oversold (could increase)
    - Wide bands: high volatility; narrow bands: low volatility
    ''')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='red', dash='dash')))
    # Shaded area between bands
    fig.add_traces([
        go.Scatter(
            x=pd.concat([df_stock['Date'], df_stock['Date'][::-1]]),
            y=pd.concat([df_stock['BB_Upper'], df_stock['BB_Lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(200,200,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        )
    ])
    fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price', font=dict(size=14), plot_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- Volatility ---
    st.subheader('5. Volatility (20-day Rolling Std Dev)')
    st.markdown('''
    **Shows how much the price moves (risk).**
    
    **What to look for:**
    - High volatility = bigger price swings (riskier)
    - Low volatility = stable price
    ''')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['Volatility_20'], mode='lines', name='20-day Volatility', line=dict(color='brown')))
    fig.update_layout(title='20-day Rolling Volatility', xaxis_title='Date', yaxis_title='Volatility', font=dict(size=14), plot_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

# --- Forecasting Page ---
elif page == 'Forecasting':
    st.header(f'Model Forecasts: {company_name} ({ticker})')
    if model_results is not None:
        st.markdown('''
        **Compare model predictions for the selected forecast horizon.**
        - **Actual**: True closing price
        - **ARIMA, SARIMA**: Model forecasts
        ''')
        models = ['ARIMA_Forecast', 'SARIMA_Forecast']
        model_display_names = {'ARIMA_Forecast': 'ARIMA', 'SARIMA_Forecast': 'SARIMA'}
        selected_models = st.multiselect('Select models to display:', [model_display_names[m] for m in models], default=[model_display_names[m] for m in models])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model_results['Date'], y=model_results['Actual'], mode='lines', name='Actual', line=dict(color='black')))
        colors = ['red', 'magenta']
        for i, m in enumerate(models):
            display_name = model_display_names[m]
            if display_name in selected_models and m in model_results.columns:
                fig.add_trace(go.Scatter(x=model_results['Date'], y=model_results[m], mode='lines', name=display_name, line=dict(color=colors[i], dash='dash')))
        fig.update_layout(title='Actual vs. Model Forecasts', xaxis_title='Date', yaxis_title='Close Price')
        st.plotly_chart(fig, use_container_width=True)
        # Downloadable CSV
        st.download_button(
            label='Download Forecast Data as CSV',
            data=model_results.to_csv(index=False).encode('utf-8'),
            file_name=f'{ticker}_forecast_{horizon}d.csv',
            mime='text/csv'
        )
        # Trend signal
        st.subheader('📊 Model Trend Signal')
        last_row = model_results.iloc[-1]
        prev_row = model_results.iloc[-2]
        for i, m in enumerate(models):
            display_name = model_display_names[m]
            if display_name in selected_models and m in model_results.columns:
                trend = '↑' if last_row[m] > prev_row[m] else '↓'
                st.markdown(f"**{display_name}:** {trend} ({'Up' if trend=='↑' else 'Down'})")
    else:
        st.warning('No model results available for this stock.')

# --- Summary & Insights Page ---
elif page == 'Summary & Insights':
    st.header(f'Summary & Insights: {company_name} ({ticker})')
    st.markdown('''
    - **Latest Model Signals:**
    ''')
    if model_results is not None:
        last_row = model_results.iloc[-1]
        prev_row = model_results.iloc[-2]
        models = ['ARIMA_Forecast', 'SARIMA_Forecast']
        model_display_names = {'ARIMA_Forecast': 'ARIMA', 'SARIMA_Forecast': 'SARIMA'}
        for m in models:
            display_name = model_display_names[m]
            if m in model_results.columns:
                trend = '↑' if last_row[m] > prev_row[m] else '↓'
                st.markdown(f"- **{display_name}:** {trend} ({'Up' if trend=='↑' else 'Down'})")
        st.markdown('---')
        st.markdown('**Latest Actual Price:**')
        st.metric('Actual', f"{last_row['Actual']:.2f}")
        st.markdown('**Model Forecasts:**')
        for m in models:
            display_name = model_display_names[m]
            if m in model_results.columns:
                st.metric(display_name, f"{last_row[m]:.2f}")
        # Downloadable CSV
        st.download_button(
            label='Download Forecast Data as CSV',
            data=model_results.to_csv(index=False).encode('utf-8'),
            file_name=f'{ticker}_forecast_{horizon}d.csv',
            mime='text/csv'
        )
    else:
        st.warning('No model results available for this stock.')
    st.markdown('---')
    st.markdown('''
    **How to use:**
    - Use the trend arrows to quickly see if the models expect the stock to go up or down.
    - Use the EDA page to understand why (look for overbought/oversold, volatility, etc).
    - Use the Forecasting page to compare models and see which is most accurate.
    ''')

# --- How to Read Charts Page ---
elif page == 'How to Read Charts':
    st.header('How to Read the Charts')
    st.markdown(open('notebooks/how_to_read_charts.md').read())
    st.info('For more details, see the documentation or ask your team lead!') 