import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from binance.client import Client 
from sklearn.linear_model import LinearRegression 
from PIL import Image

API_KEY = 'your_api_key_here'
API_SECRET = 'your_api_secret_here'

client = Client(API_KEY, API_SECRET, requests_params={'timeout': 30})

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, interval, days=1):
    since = int(time.time() * 1000) - days * 24 * 60 * 60 * 1000
    try:
        candles = client.get_klines(symbol=symbol, interval=interval, startTime=since)
        df = pd.DataFrame(candles, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_regression(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    slope = model.coef_[0]
    return 'Positive' if slope > 0 else 'Negative' if slope < 0 else 'Flat', y_pred

def plot_regression(df, y_pred, height=500):
    if 'timestamp' not in df.columns:
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        else:
            raise KeyError("No timestamp-like column found in dataframe. Expected 'timestamp', 'open_time', or 'time'.")

    fig, ax = plt.subplots(figsize=(12, height / 100))
    ax.plot(df['timestamp'], df['close'], label='BTC Price', color='blue')
    ax.plot(df['timestamp'], y_pred, label='Regression Line', color='red', linestyle='--')
    ax.set_title('BTC Price with Regression Line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid()
    
    return fig

def calculate_price_reaction_velocity(df, ema_col, k=5):
  
    touches = df[(df['close'].shift(1) > df[ema_col]) & (df['close'] <= df[ema_col]) | 
                 (df['close'].shift(1) < df[ema_col]) & (df['close'] >= df[ema_col])]
    velocities = []
    
    for idx in touches.index:
        if idx + k < len(df):
            reaction = abs(df.loc[idx + k, 'close'] - df.loc[idx, 'close'])
            velocities.append(reaction)
            
    return np.mean(velocities) if velocities else 0
def calculate_bounce_efficiency(df, ema_col, k=10, threshold=0.005):
    touches = df[(df['close'].shift(1) > df[ema_col]) & (df['close'] <= df[ema_col]) | 
                 (df['close'].shift(1) < df[ema_col]) & (df['close'] >= df[ema_col])]
    
    significant_bounces = 0
    
    for idx in touches.index:
        if idx + k < len(df):
            reaction = abs(df.loc[idx + k, 'close'] - df.loc[idx, 'close'])
            if reaction > threshold * df.loc[idx, 'close']:
                significant_bounces += 1
                
    return significant_bounces / len(touches) if len(touches) > 0 else 0
def identify_best_ma_ema_and_plot(df):
    results = []

    for period in range(15, 76, 3):
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        velocity = calculate_price_reaction_velocity(df, f'EMA_{period}', k=10)
        efficiency = calculate_bounce_efficiency(df, f'EMA_{period}', k=10, threshold=0.005)
        
        if velocity is None or efficiency is None:
            velocity = 0
            efficiency = 0

        results.append({
            'Period': period,
            'Metric': f'EMA_{period}',
            'Velocity': velocity,
            'Efficiency': efficiency,
        })

    results_df = pd.DataFrame(results)

    
    results_df['Scaled Velocity'] = (results_df['Velocity'] - results_df['Velocity'].min()) / \
                                    (results_df['Velocity'].max() - results_df['Velocity'].min())

    eff_min = results_df['Efficiency'].min()
    eff_max = results_df['Efficiency'].max()

    if eff_max == eff_min:
        results_df['Scaled Efficiency'] = 0.0
    else:
        results_df['Scaled Efficiency'] = (results_df['Efficiency'] - eff_min) / (eff_max - eff_min)
    
    results_df['Combined Score'] = 0.41 * results_df['Scaled Velocity'] + 0.59 * results_df['Scaled Efficiency'].fillna(0)

    # Identify the best EMA
    best = results_df.sort_values('Combined Score', ascending=False).iloc[0]

    # Return best metric and results dataframe (removed plotting loop)
    return best['Metric'], results_df
@st.cache_data(ttl=300)
def calculate_correlation_and_sensitivity(base_df, target_df, decimals=4):
    freq = '15min'

    base_df = base_df.copy()
    target_df = target_df.copy()
    
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if 'timestamp' in base_df.columns:
            base_df.set_index('timestamp', inplace=True)
        elif 'open_time' in base_df.columns:
            base_df['timestamp'] = pd.to_datetime(base_df['open_time'], unit='ms')
            base_df.set_index('timestamp', inplace=True)
        else:
            return 0.0, 0.0, 0.0, 0.0
    
    if not isinstance(target_df.index, pd.DatetimeIndex):
        if 'timestamp' in target_df.columns:
            target_df.set_index('timestamp', inplace=True)
        elif 'open_time' in target_df.columns:
            target_df['timestamp'] = pd.to_datetime(target_df['open_time'], unit='ms')
            target_df.set_index('timestamp', inplace=True)
        else:
            return 0.0, 0.0, 0.0, 0.0

    try:
        base_resampled = base_df['close'].astype(float).resample(freq).ffill()
        target_resampled = target_df['close'].astype(float).resample(freq).ffill()
    except Exception as e:
        print(f"Resampling error: {e}")
        return 0.0, 0.0, 0.0, 0.0

    base_pct_change = base_resampled.pct_change().dropna()
    target_pct_change = target_resampled.pct_change().dropna()

    aligned_data = pd.concat([base_pct_change, target_pct_change], axis=1).dropna()
    aligned_data.columns = ['Base_pct_change', 'Target_pct_change']

    if len(aligned_data) < 2:
        return 0.0, 0.0, 0.0, 0.0

    try:
        correlation = float(aligned_data['Base_pct_change'].corr(aligned_data['Target_pct_change']))
        if pd.isna(correlation):
            correlation = 0.0
    except Exception as e:
        print(f"Correlation error: {e}")
        correlation = 0.0

    try:
        X = aligned_data['Base_pct_change'].values.reshape(-1, 1)
        y = aligned_data['Target_pct_change'].values
        reg = LinearRegression().fit(X, y)
        sensitivity = float(reg.coef_[0])
    except Exception as e:
        print(f"Sensitivity error: {e}")
        sensitivity = 0.0

    try:
        target_close = target_df['close'].astype(float).dropna().values
        if len(target_close) > 1:
            indices = np.arange(len(target_close)).reshape(-1, 1)
            trend_reg = LinearRegression().fit(indices, target_close)
            trend_direction_score = float(trend_reg.coef_[0])
        else:
            trend_direction_score = 0.0
    except Exception as e:
        print(f"Trend error: {e}")
        trend_direction_score = 0.0

    correlation_scaled = (correlation + 1) / 2
    sensitivity_scaled = sensitivity / (abs(sensitivity) + 1) if sensitivity != 0 else 0.5
    
    min_trend = min(trend_direction_score, 0)
    max_trend = max(trend_direction_score, 0)
    denom = max_trend + abs(min_trend)
    
    if denom == 0 or trend_direction_score == 0:
        trend_direction_scaled = 0.5
    else:
        trend_direction_scaled = (trend_direction_score - min_trend) / denom

    combined_score = (
        0.31 * correlation_scaled +
        0.32 * sensitivity_scaled +
        0.37 * trend_direction_scaled
    )

    return round(correlation, decimals), round(sensitivity, decimals), round(trend_direction_score, decimals), round(combined_score, decimals)

def suggest_trades(base_df, target_df, best_metric, trend):
    """Suggest long or short trades based on trend with detailed conditions."""
    target_df['high'] = pd.to_numeric(target_df['high'], errors='coerce')
    target_df['low'] = pd.to_numeric(target_df['low'], errors='coerce')

    high = target_df['high'].max()
    low = target_df['low'].min()
    variance = (high - low) / 6  # Adjusted variance division
    take_profit = variance
    stop_loss = take_profit / 4

    target_df['signal'] = None
    target_df['take_profit'] = None
    target_df['stop_loss'] = None

    latest_row = target_df.iloc[-1]

    if trend == 'Positive' and latest_row['close'] > latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Long'
        target_df.loc[target_df.index[-1], 'take_profit'] = latest_row[best_metric] + take_profit
        target_df.loc[target_df.index[-1], 'stop_loss'] = latest_row[best_metric] - stop_loss
    elif trend == 'Negative' and latest_row['close'] < latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Short'
        target_df.loc[target_df.index[-1], 'take_profit'] = latest_row[best_metric] - take_profit
        target_df.loc[target_df.index[-1], 'stop_loss'] = latest_row[best_metric] + stop_loss

    return target_df
def plot_candlestick_with_signals(df, metric_list, title, plot_positions=True, future_time_minutes=300, height=500):
    future_time = df['timestamp'].iloc[-1] + timedelta(minutes=future_time_minutes)

    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])

    for metric in metric_list:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[metric], mode='lines', name=metric))

    if plot_positions and 'signal' in df.columns:
        last_signal_row = df.iloc[-1]
        if last_signal_row['signal'] == 'Long':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['take_profit'],
                          fillcolor="green", opacity=0.2, line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['stop_loss'],
                          fillcolor="red", opacity=0.2, line_width=0)
        elif last_signal_row['signal'] == 'Short':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['stop_loss'],
                          fillcolor="red", opacity=0.2, line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['take_profit'],
                          fillcolor="green", opacity=0.2, line_width=0)

    # Enable zoom and pan for the y-axis
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        height=height,
        yaxis=dict(
            fixedrange=False,
            rangemode='normal',
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
        )
    )

    return fig
    
st.set_page_config(layout="wide", page_title="Crypto EMA Analysis")
st.markdown("""
<style>
    .main {background-color: #f2f2f2;}
    .stCard {
        border: none;
        text-align: center;
        margin: 5px;
        padding: 15px;
        border-radius: 5px;
    }
    .yellowCard {background-color: #ffffcc;}
    .greenCard {background-color: #ccffcc;}
    .redCard {background-color: #ffcccc;}
    .whiteCard {background-color: #ffffff;}
    .subheader-centered {text-align: center;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("src/__pycache__/Pic1.png", use_container_width=True)
st.sidebar.header("Settings")

days = st.sidebar.number_input("Number of Days to Fetch Data:", min_value=1, max_value=30, value=2)
interval = st.sidebar.selectbox("Select Interval:", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2)
calculate_button = st.sidebar.button("Calculate Now")

st.image("src/__pycache__/Pic2.png", use_container_width=True)
st.title("Best EMA for BTC and Crypto Positions Suggestions", anchor="center")

if calculate_button:
    # Fetch BTC data
    btc_df = fetch_crypto_data("BTCUSDT", interval, days=days)
    btc_trend, y_pred = calculate_regression(btc_df)

    # Find top EMAs
    best_metric, all_metrics_df = identify_best_ma_ema_and_plot(btc_df)
    top_3_metrics = all_metrics_df.sort_values("Combined Score", ascending=False).head(3)["Metric"].tolist()

    # Fetch and calculate correlation and sensitivity for other coins
    coins = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SOLUSDT', 'XRPUSDT',
       'BNBUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'LINKUSDT',
       'UNIUSDT', 'LTCUSDT', 'ARBITUSDT', 'SHIBUSDT', 'DYDXUSDT']

    results = []
    for coin in coins:
        try:
            target_df = fetch_crypto_data(coin, '1h', days=1)
            base_df = fetch_crypto_data('BTCUSDT', '1h', days=1)
            if not target_df.empty and not base_df.empty:
                correlation, sensitivity, trend_direction_score, combined_score = calculate_correlation_and_sensitivity(base_df, target_df, 4)
                results.append({'Coin': coin, 'Correlation': correlation, 'Sensitivity': sensitivity, 'Trend Direction Score': trend_direction_score, 'Combined Score': combined_score})
        except Exception as e:
            print(f"Error processing {coin}: {e}")

    results_df = pd.DataFrame(results)

    if btc_trend == 'Positive':
        best_coin = results_df.sort_values(['Combined Score'], ascending=False).iloc[0]['Coin']
    else:
        best_coin = results_df.sort_values(
            ['Trend Direction Score', 'Correlation', 'Sensitivity'],
            ascending=[True, False, False]
        ).iloc[0]['Coin']

    best_coin_df = fetch_crypto_data(best_coin, interval)
    best_coin_df[best_metric] = best_coin_df['close'].ewm(span=int(best_metric.split('_')[1]), adjust=False).mean()

    best_coin_df = suggest_trades(btc_df, best_coin_df, best_metric, btc_trend)

    target_price = best_coin_df["close"].iloc[-1]
    btc_price = btc_df["close"].iloc[-1]
    price_to_buy_or_sell = best_coin_df.iloc[-1][best_metric]

    take_profit = best_coin_df.iloc[-1].get('take_profit', None)
    if take_profit is None:
        take_profit = 0.0

    stop_loss = best_coin_df.iloc[-1].get('stop_loss', None)
    if stop_loss is None:
        stop_loss = 0.0



    col1, col2, col3, col4, col5, col6 = st.columns(6) 

    with col1:
        st.markdown(f'<div class="stCard yellowCard">Latest BTC Price<br><b>${btc_price:,.0f}</b></div>', unsafe_allow_html=True)

    with col2:
        trend_color = "greenCard" if btc_trend == "Positive" else "redCard"
        st.markdown(f'<div class="stCard {trend_color}">BTC Trend<br><b>{btc_trend}</b></div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="stCard whiteCard">Best Performing Coin<br><b>{best_coin}</b></div>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<div class="stCard greenCard">Price to Buy or Sell<br><b>${price_to_buy_or_sell:,.6f}</b></div>', unsafe_allow_html=True)

    with col5:
        st.markdown(f'<div class="stCard greenCard">Take Profit Price<br><b>${take_profit:,.6f}</b></div>', unsafe_allow_html=True)

    with col6:
        st.markdown(f'<div class="stCard redCard">Stop Loss<br><b>${stop_loss:,.6f}</b></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="subheader-centered"><h3>Scatterplot: BTC Regression Line</h3></div>', unsafe_allow_html=True)
        plot_regression(btc_df, y_pred, height=1120)

    with col2:
        st.markdown('<div class="subheader-centered"><h3>Top 10 EMAs by Combined Score</h3></div>', unsafe_allow_html=True)
        st.dataframe(all_metrics_df[['Period', 'Metric', 'Velocity', 'Efficiency','Combined Score']].sort_values("Combined Score", ascending=False).head(10), height=400)

    with col3:
        st.markdown('<div class="subheader-centered"><h3>BTC Candlestick with EMAs</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(plot_candlestick_with_signals(btc_df, top_3_metrics, "BTC with Top 3 EMAs", height=400), use_container_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="subheader-centered"><h3>Top Coins by Combined Score</h3></div>', unsafe_allow_html=True)
        st.dataframe(results_df.sort_values("Combined Score", ascending=False), height=600)

    with col2:
        st.markdown(f'<div class="subheader-centered"><h3>{best_coin} Candlestick with Suggested Position</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(plot_candlestick_with_signals(best_coin_df, [best_metric], f"{best_coin} with Suggested Position", future_time_minutes=300, height=600), use_container_width=True)
