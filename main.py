import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import streamlit as st
import plotly.graph_objects as go
import asyncio
import aiohttp
from dotenv import load_dotenv
from textblob import TextBlob
from bs4 import BeautifulSoup
import ta  # Technical Analysis Library
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3

# Load environment variables
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def get_stock_data(company_name):
    ticker = yf.Ticker(company_name)
    df = ticker.history(period='6mo')
    df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    return df

async def fetch_news(session, url):
    async with session.get(url) as response:
        return await response.text()

async def get_news_sentiment(company_name):
    url = f'https://news.google.com/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en'
    async with aiohttp.ClientSession() as session:
        html = await fetch_news(session, url)
    soup = BeautifulSoup(html, 'html.parser')
    articles = soup.find_all('article')
    sentiment_scores = [TextBlob(article.text).sentiment.polarity for article in articles[:10]]
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    return "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(30, len(df_scaled)):
        X.append(df_scaled[i-30:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model

def predict_stock_price(df):
    X, y, scaler = prepare_lstm_data(df)
    model_path = "lstm_stock_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = train_lstm(X, y)
        model.save(model_path)
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions), df.index[-len(predictions):]

def send_email(user_email, company_name, sentiment):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = user_email
    msg['Subject'] = f"Stock Alert: {company_name} Sentiment Update"
    msg.attach(MIMEText(f"Sentiment for {company_name} is {sentiment}", 'plain'))
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, user_email, msg.as_string())

def plot_stock(df, company_name, predicted_prices, predicted_dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price', line=dict(dash='dot')))
    st.plotly_chart(fig)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Stock Market Insights")
st.title("📈 Stock Insights - Get Advanced Stock Analysis & Prediction")
company_name = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
user_email = st.text_input("Enter your email for alerts")
if st.button("Analyze"):
    df = get_stock_data(company_name)
    if df is not None:
        st.subheader(f"📊 {company_name} Stock Data")
        st.write(df.tail(5))
        predicted_prices, predicted_dates = predict_stock_price(df)
        plot_stock(df, company_name, predicted_prices, predicted_dates)
        sentiment = asyncio.run(get_news_sentiment(company_name))
        st.subheader("📰 News Sentiment Analysis")
        st.write(f"Overall Sentiment: {sentiment}")
        if user_email:
            send_email(user_email, company_name, sentiment)
            st.success(f"Email alert sent to {user_email}")
