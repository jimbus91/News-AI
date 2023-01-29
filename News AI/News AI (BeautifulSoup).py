import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from datetime import timedelta, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_hist_gradient_boosting

# Ask the user for the stock ticker symbol
stock_ticker = input("Enter the stock ticker symbol: ")

# Get the date 1 week ago
one_week_ago = date.today() - timedelta(weeks=1)

# Download the stock data for the chosen stock
stock_data = yf.download(stock_ticker, start=one_week_ago)

# Check if the 'Date' column is present, if not create it
if 'Date' not in stock_data.columns:
 stock_data['Date'] = stock_data.index

# Convert the date column to a datetime object if it's not in the correct format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Set the date column as the index
stock_data.set_index('Date', inplace=True)

# Sort the data by date
stock_data.sort_index(inplace=True)

# Make the request to the Yahoo Finance website
url = f'https://finance.yahoo.com/quote/{stock_ticker}/news'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the news articles
articles = soup.find_all('li', class_='js-stream-content')

news_data = []

# Print the title, source, and summary of each article
for article in articles:
    title = article.find('h3').text
    source = article.find('span', class_='C($c-fuji-grey-c) Fz(11px)').text
    summary = article.find('p').text
    date = article.find('time').text
    news_data.append({'title': title, 'source': source, 'summary': summary, 'date': date})

# Print the response content of the website    
print(response.content)

# Print the first news article data in the list
print(pd.DataFrame(news_data))

# Convert the news data to a DataFrame
news_data = pd.DataFrame(news_data)

# Convert date to datetime format
news_data['Date'] = pd.to_datetime(news_data['Date'])

# Set the date column as the index
news_data.set_index('date', inplace=True)

# Sort the data by date
news_data.sort_index(inplace=True)

# Merge the stock and news data
data = pd.merge(stock_data, news_data, left_index=True, right_index=True)

# Select the last week's worth of data
last_week = data.iloc[-7:]

# Split the data into X (features) and y (target)
X = last_week.drop(columns=['Close'])
y = last_week['Close']

# Use CountVectorizer to extract features from the news articles
vectorizer = CountVectorizer()
X_news = vectorizer.fit_transform(X['Article'])

# Concatenate the news features with the other features
X = pd.concat([X.drop(columns=['Article']), pd.DataFrame(X_news.toarray())], axis=1)

# Train a random forest regressor on the data
model = RandomForestRegressor()
model.fit(X, y)

# Make predictions for the next 30 days
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
future_data = pd.DataFrame(index=future_dates, columns=X.columns)
predictions = model.predict(future_data)
    
# Calculate the standard deviation of the last year's close prices
std_dev = predictions.std()

# Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

# Add the random values to the predicted prices
predictions += random_values 
predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])
    
# Concatenate the last_year and predictions dataframes
predictions_df = pd.concat([last_year, predictions_df])

# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame(index=future_dates, columns=['Prediction'])
predictions_df['Prediction'] = predictions

# Set the style to dark theme
style.use('dark_background')

# Create the plot
fig, ax = plt.subplots()
for i in range(len(predictions)):
    if predictions[i] < data['Close'].iloc[-1]:
        plt.plot(future_dates[i], predictions[i], '-or', label='Predicted')
    else:
        plt.plot(future_dates[i], predictions[i], '-og', label='Predicted')
plt.plot(data['Close'], label='Actual', color='blue')
plt.title('AAPL Stock Price Predictions')
plt.xlabel('Day')
plt.ylabel('Price (USD)')
plt.legend()

# Set the x-axis tick locator
ax.xaxis.set_major_locator(mdates.DayLocator(interval=55))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Show the plot
plt.show()