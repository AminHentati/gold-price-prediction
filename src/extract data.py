import yfinance as yf

# Define the ticker symbol for gold futures
ticker_symbol = 'GC=F'

# Download historical data from 2010 to today
data = yf.download(ticker_symbol, start="2010-01-01", end="2024-12-15", interval="1d")

# Save the data to a CSV file
data.to_csv('gold_data.csv')

print("Historical gold data from 2010 to today saved to 'gold_data_2010_today.csv'")
