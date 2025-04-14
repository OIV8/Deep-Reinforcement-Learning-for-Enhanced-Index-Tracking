import pandas as pd
import yfinance as yf
import datetime

# Load S&P 500 constituent data with start/end dates
csv_url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
sp500_df = pd.read_csv(csv_url, parse_dates=['start_date', 'end_date'])

# Filter stocks that were in S&P 500 for the ENTIRE 2013-2024 period
start = datetime.datetime(2013, 10, 31)
end = datetime.datetime(2024, 10, 31)
mask = (
    (sp500_df['start_date'] <= start) & 
    (sp500_df['end_date'].isna() | (sp500_df['end_date'] >= end))
)
filtered_df = sp500_df[mask]

# Download data only for these "survivor" stocks
unique_tickers = filtered_df['ticker'].unique().tolist()
unique_tickers = [ticker.replace('.', '-') for ticker in unique_tickers]

# Download price/volume data (no NaN/zero checks)
data = yf.download(unique_tickers, start=start, end=end)
price_data = data['Close']
volume_data = data['Volume']

# Remove tickers (columns) with any NaN or 0 values in either "Close" or "Volume"
valid_tickers = price_data.dropna(axis=1).loc[:, (price_data != 0).all(axis=0) & (price_data >= 1).all(axis=0)].columns
valid_tickers = volume_data[valid_tickers].dropna(axis=1).loc[:, (volume_data != 0).all(axis=0)].columns

# Filter both DataFrames to include only valid tickers
cleaned_price_data = price_data[valid_tickers]
cleaned_volume_data = volume_data[valid_tickers].iloc[:,:-2]

# Download S&P 500 and VIX
sp_vix = yf.download(["^GSPC", "^VIX"], start=start, end=end)["Close"]

# Combine data (survivors only)
df = pd.concat([cleaned_volume_data, cleaned_price_data, sp_vix], axis = 1)


train = df.loc["2013-10-31":"2023-10-31"]
test = df.loc["2022-08-31":"2024-10-31"]

train.to_csv('train_df.csv', index = True)
test.to_csv('test_df.csv', index = True)
