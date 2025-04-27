import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import requests
from textblob import TextBlob

# Streamlit App Title
st.title("📈 Stock OHLC Prediction App")

# Select stock and date range
stock_symbol = st.text_input("Enter NSE stock symbol (e.g. TCS.NS):", "TCS.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if start_date >= end_date:
    st.warning("⚠️ End date must be after start date.")
    st.stop()

# Download historical data
raw_df = yf.download(stock_symbol, start=start_date, end=end_date + pd.Timedelta(days=1))

if raw_df.empty:
    st.warning("⚠️ No data found. Please check the stock symbol.")
    st.stop()

st.write(f"### Showing data for {stock_symbol}")
st.dataframe(raw_df.tail())

# Feature Engineering
df = raw_df.copy()
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=5).std()
df['RSI'] = 100 - (100 / (1 + df['Returns'].rolling(window=14).mean() / df['Returns'].rolling(window=14).std()))
df['Bollinger Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['Bollinger Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
df.dropna(inplace=True)

# Trend Labels Based on Indicators
df['Trend_MA'] = np.where(df['MA5'] > df['MA10'], 1, 0)
df['Trend_RSI'] = np.where(df['RSI'] > 50, 1, 0)
df = df[df['Bollinger Upper'].notnull() & df['Bollinger Lower'].notnull()]

boll_upper = df['Bollinger Upper'].values.flatten()
boll_lower = df['Bollinger Lower'].values.flatten()
close_vals = df['Close'].values.flatten()

trend_bollinger = np.where(close_vals > boll_upper, 1,
                   np.where(close_vals < boll_lower, 0, 1))

df['Trend_Bollinger'] = trend_bollinger

trend_features = ['Trend_MA', 'Trend_RSI', 'Trend_Bollinger']
df['Overall_Trend'] = df[trend_features].mean(axis=1).round().astype(int)

# Features for ML
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'Returns', 'Volatility', 'RSI']
X = df[feature_columns]

# Targets: Open, High, Low, Close
ohlc_targets = ['Open', 'High', 'Low', 'Close']
y_ohlc = df[ohlc_targets].loc[X.index]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MultiOutput Regressor
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))
split = int(len(X_scaled) * 0.8)
model.fit(X_scaled[:split], y_ohlc.iloc[:split])
y_ohlc_pred = model.predict(X_scaled[split:])

# Predict next day's OHLC
X_last_scaled = scaler.transform(df[feature_columns].iloc[[-1]])
predicted_ohlc = model.predict(X_last_scaled)[0]

# Yesterday's OHLC
yesterday_ohlc = raw_df[ohlc_targets].iloc[-2].values

# Latest OHLC (today's actual)
today_ohlc = raw_df[ohlc_targets].iloc[-1].values

# Display comparison table
st.write("## 🔍 Yesterday vs Today vs Predicted (Next Day) OHLC")
comparison_df = pd.DataFrame({
    'Yesterday': yesterday_ohlc.flatten(),
    'Today': today_ohlc.flatten(),
    'Predicted (Next Day)': predicted_ohlc.flatten()
}, index=ohlc_targets)

st.dataframe(comparison_df.style.format("{:.2f}"))

# Indicator Summary Table
st.write("## 🧭 Indicator Trends Table")
indicator_table = pd.DataFrame({
    'MA Trend': ['Uptrend' if df['Trend_MA'].iloc[-1] else 'Downtrend'],
    'RSI Trend': ['Bullish' if df['Trend_RSI'].iloc[-1] else 'Bearish'],
    'Bollinger Trend': ['Above Upper Band' if df['Trend_Bollinger'].iloc[-1] == 1 else ('Below Lower Band' if df['Trend_Bollinger'].iloc[-1] == 0 else 'Within Band')],
    'ML Trend Prediction': ['Uptrend' if df['Overall_Trend'].iloc[-1] else 'Downtrend']
})
st.dataframe(indicator_table)

# Show current trend based on MA crossover
st.write("### 📊 Current ML-Based Market Trend")
st.write(f"The market is predicted to be in an **{'Uptrend' if df['Overall_Trend'].iloc[-1] else 'Downtrend'}**.")

# Actual vs Predicted Close Plot
actual_close = y_ohlc['Close'].iloc[split:].values.flatten()
pred_close = y_ohlc_pred[:, 3].flatten()

plot_df = pd.DataFrame({
    'Actual': actual_close,
    'Predicted': pred_close
}, index=y_ohlc.iloc[split:].index)

st.write("## 📉 Actual vs Predicted Close Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual Close'))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted Close'))
fig.update_layout(
    title="Actual vs Predicted Close Prices (Backtest)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    legend=dict(x=0, y=1.0),
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# Backtesting Metrics
st.write("## 📈 Backtesting Summary")
mae = mean_absolute_error(actual_close, pred_close)
mse = mean_squared_error(actual_close, pred_close)
rmse = np.sqrt(mse)
r2 = r2_score(actual_close, pred_close)

backtest_metrics = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae, mse, rmse, r2]
})

st.dataframe(backtest_metrics.style.format({"Value": "{:.4f}"}))

# User-Friendly Summary of Backtesting with Accuracy Estimate
st.subheader("📘 What Do These Metrics Mean?")
st.markdown("""
- **MAE (Mean Absolute Error)**: On average, the model's predictions were ₹{:.2f} away from the actual closing prices.
- **RMSE (Root Mean Squared Error)**: This penalizes larger errors more than MAE. A lower value means better accuracy.
- **R² Score**: This tells how well the model explains the price movements. A value closer to 1 means a better fit. 
  - **Accuracy Estimate**: Based on the R² score of {:.4f}, the model is approximately **{:.2f}% accurate** in predicting the closing prices.
""".format(mae, r2, r2 * 100))

# Additional insights and user-friendly summary
st.write("## 🧠 Additional Insights & Sentiment Analysis")

# News Sentiment Analysis
st.subheader("📰 News & Sentiment Analysis")

API_KEY = 'st.secrets["newsapi"]["api_key"]'  # Replace with your actual NewsAPI key

def get_stock_news(stock_symbol, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return [article['title'] for article in news_data['articles']]
    else:
        return []

def analyze_sentiment(news_headlines):
    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for headline in news_headlines:
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment_scores['Positive'] += 1
        elif polarity < 0:
            sentiment_scores['Negative'] += 1
        else:
            sentiment_scores['Neutral'] += 1
    return sentiment_scores

try:
    stock_news = get_stock_news(stock_symbol, API_KEY)
    sentiment = analyze_sentiment(stock_news)
    total = sum(sentiment.values())
    st.write(f"### Sentiment Analysis for {stock_symbol}")
    st.write(f"Positive News: {sentiment['Positive']}")
    st.write(f"Negative News: {sentiment['Negative']}")
    st.write(f"Neutral News: {sentiment['Neutral']}")
    if total > 0:
        overall = 'Positive' if sentiment['Positive'] / total > 0.5 else 'Negative' if sentiment['Negative'] / total > 0.5 else 'Neutral'
        st.write(f"**Overall Sentiment**: {overall}")
    else:
        st.write("No news available to analyze.")
except Exception as e:
    st.error(f"⚠️ Error fetching or analyzing news: {e}")

# Plain Summary for Non-Technical Users
st.subheader("📊 Easy-to-Understand Summary")
st.info("Coming soon: Plain-language insights for non-technical users, including potential risk levels and trend summaries.")
