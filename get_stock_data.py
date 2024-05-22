# IMPORT NEEDED LIBRARIES
import os
import pandas as pd
import yfinance as yf

# CREATE AN EXCEL FILE WITH DAILY STOCK DATA
# Set working directory.
path = "C:/Users/franc/OneDrive/Escritorio/On portfolio construction"
os.chdir(path)

# Create a list with the paths of the .csv files with the listed stocks.
paths = ['Amsterdam.csv', 'Bruselas.csv', 'Lisboa.csv', 'Milan.csv', 
         'Paris.csv']

# Create a list of ticker endings.
ends = ['.AS', '.BR', '.LS', '.MI', '.PA']

# Create an empty list where tickers will be stored.
euronxt_symbols = []

# Create a loop to store tickers of all the stocks.
for i in range(len(paths)):
    euronxt_df = pd.read_csv(paths[i], sep=';', low_memory=False)
    symbols = euronxt_df['Symbol']
    euronxt_symbols_temp = [str(x) + ends[i] for x in list(symbols)]
    euronxt_symbols.extend(euronxt_symbols_temp)
    
# Download daily data from Yahoo Finance.
historical_data = {}
for ticker in euronxt_symbols:
    stock_data = yf.download(ticker, start='2018-01-01', end='2024-03-14',
                             interval='1d')
    historical_data[ticker] =stock_data['Adj Close']
    
# Concatenate the dictionary in a single dataframe.
data_def = pd.concat(historical_data.values(), keys=euronxt_symbols, axis=1)

# Export the dataframe as an Excel file.
path = path + '/dataset.xlsx'
data_def.to_excel(path, index=True)