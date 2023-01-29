import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from newsapi import NewsApiClient
from datetime import timedelta, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_hist_gradient_boosting

# Ask the user for the stock ticker symbol
stock_ticker = input("Enter the stock ticker symbol: ")

# Get the date 1 Month ago
one_month_ago = date.today() - timedelta(weeks=4)

# Download the stock data for the chosen stock
stock_data = yf.download(stock_ticker, start=one_month_ago)

# Check if the 'Date' column is present, if not create it
if 'Date' not in stock_data.columns:
 stock_data['Date'] = stock_data.index

# Convert the date column to a datetime object if it's not in the correct format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Set the date column as the index
stock_data.set_index('Date', inplace=True)

# Sort the data by date
stock_data.sort_index(inplace=True)

# Define the API endpoint
url = ('https://newsapi.org/v2/everything?'
       'q='+stock_ticker+'&'
       'from='+str(one_month_ago)+'&'
       'sortBy=popularity&'
       'domains=cnbc.com&'
       'language=en&'
       'apiKey=d71f10006d5545de827c10c38cdcfd69')

# Make the API call
response = requests.get(url)

#Create a new dataframe to store the extracted data
df_news = pd.DataFrame(columns=['title', 'description', 'publishedAt', 'content'])

#Define 'Data'
data = response.json()

#Iterate through the list of dictionaries in the 'articles' key
for i in range(len(data['articles'])):

# Extract the values for the desired keys
    title = data['articles'][i]['title']
    description = data['articles'][i]['description']
    publishedAt = data['articles'][i]['publishedAt']
    content = data['articles'][i]['content']

# Append the extracted values to a list
news_list = []
news_list.append({'title': title, 'description': description, 'publishedAt': publishedAt, 'content': content})

# Convert the list to a dataframe
df_news = pd.DataFrame(news_list)

# Print the JSON data
print(json.dumps(response.json(), indent=4))

# Save the dataframe to a csv file
df_news.to_csv('CNBC_'+stock_ticker+'_News.csv', columns=['title', 'description', 'publishedAt', 'content'], index=False)

# Read the csv file and store it in a DataFrame
#news_data = pd.read_csv('C:\\Users\\jimbu\\News AI\\CNBC_AAPL_News.csv') #Windows
news_data = pd.read_csv('/Users/beusse/Desktop/Stocks/News AI/CNBC_AAPL_News.csv') #Mac
news_data = news_data[['title', 'description', 'publishedAt', 'content']] 

# Select the columns containing the articles
articles = news_data[['title', 'description', 'publishedAt', 'content']]

# Merge the stock and news data
merged_data = pd.merge(stock_data, news_data, left_index=True, right_index=True)

# Select the last month's worth of data
last_month = merged_data.iloc[-30:]

# Split the data into X (features) and y (target)
X = last_month.drop(columns=['Close'])
y = last_month['Close']

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=[])

# Extract features from the 'content' column in the X dataframe
X_news = vectorizer.fit_transform(X['content'])

# Concatenate the news features with the other features
X = pd.concat([X.drop(columns=['content']),
pd.DataFrame(X_news.toarray())], axis=1)

# Train a random forest regressor on the data
model = RandomForestRegressor()
model.fit(X, y)

# Make predictions for the next 30 days
future_dates = pd.date_range(start=merged_data.index[-1], periods=30, freq='D')
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
merged_data = pd.concat([merged_data, predictions_df])

# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame(index=future_dates, columns=['Prediction'])
predictions_df['Prediction'] = predictions

# Set the style to dark theme
style.use('dark_background')

# Plot the predictions
fig, ax = plt.subplots()
for i in range(len(predictions)):
    if predictions[i] < merged_data['Close'].iloc[-1]:
        plt.plot(future_dates[i], predictions[i], '-or', label='Predicted')
    else:
        plt.plot(future_dates[i], predictions[i], '-og', label='Predicted')
plt.plot(merged_data['Close'], label='Actual', color='blue')
plt.title(f'{stock_ticker} Stock Price Predictions')
plt.xlabel('Day')
plt.ylabel('Price (USD)')
plt.legend()

# Set the x-axis tick locator
ax.xaxis.set_major_locator(mdates.DayLocator(interval=55))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Show the plot
plt.show()