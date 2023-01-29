import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from datetime import timedelta, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_hist_gradient_boosting

#Ask the user for the stock ticker symbol
stock_ticker = input("Enter the stock ticker symbol: ")

#Get the date 1 week ago
one_week_ago = date.today() - timedelta(weeks=4)

#Download the stock data for the chosen stock
stock_data = yf.download(stock_ticker, start=one_week_ago)

#Check if the 'Date' column is present, if not create it
if 'Date' not in stock_data.columns:
     stock_data['Date'] = stock_data.index

#Convert the date column to a datetime object if it's not in the correct format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

#Set the date column as the index
stock_data.set_index('Date', inplace=True)

#Sort the data by date
stock_data.sort_index(inplace=True)

#Define the API endpoint and parameters
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers='+stock_ticker+'&apikey=X1A22TUICY3L2QZA'

#Make the API call
response = requests.get(url)

#Get the JSON data
data = response.json()

# Iterate through the JSON data
for key in data:
    if key == "OverallSentiment":
        # Print the "OverallSentiment" key and its value
        print(json.dumps(data[key], indent=4))
    elif key == "News":
        for i in range(len(data[key])):
            if data[key][i]["ticker"] == stock_ticker:
                # Print the "News" key and its value that matches the stock user input
                print(json.dumps(data[key][i], indent=4))

# convert dictionary to Dataframe
data_df = pd.DataFrame.from_dict(data)

# Export the data to a CSV file
data_df.to_csv(stock_ticker + "_News.csv", sep=",", index=False)

#Extract the news from the JSON data
articles = data["sentiment"]
news_data = pd.DataFrame(articles)

#Merge the stock and news data
merged_data = pd.merge(stock_data, news_data, left_index=True, right_index=True)

#Select the last month's worth of data
last_month = merged_data.iloc[-30:]

#Split the data into X (features) and y (target)
X = last_month.drop(columns=['Close'])
y = last_month['Close']

#Train a random forest regressor on the data
model = RandomForestRegressor()
model.fit(X, y)

#Make predictions for the next 30 days
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
future_data = pd.DataFrame(index=future_dates, columns=X.columns)
predictions = model.predict(future_data)

#Calculate the standard deviation of the last year's close prices
std_dev = predictions.std()

#Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

# Add the random values to the predicted prices
predictions += random_values 
predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

# Concatenate the last_year and predictions dataframes

# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame(index=future_dates, columns=['Prediction'])
predictions_df['Prediction'] = predictions

# Set the style to dark theme
style.use('dark_background')

# Plot the predictions
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