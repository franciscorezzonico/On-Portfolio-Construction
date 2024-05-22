# IMPORT NEEDED LIBRARIES
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pmdarima as pm
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dense 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import statistics
from scipy.stats import trim_mean

# Set working directory.
path = 'C:/Users/franc/OneDrive/Escritorio/On portfolio construction/'
os.chdir(path)

# FIT SARIMA MODELS
# Read the dataset with daily data.
path = 'dataset.xlsx'
daily_data = pd.read_excel(path)

# Change the format of 'Date' column.
daily_data['Date'] = daily_data['Date'].dt.date

# Remove missing values and set 'Date' column as index.
daily_data.dropna(inplace=True, axis=1)
daily_data.index = daily_data.pop('Date')

# Create three empty dictionaries and a dataframe to store the goodness-of-fit
# measures and the predictions.
mse_arima, mae_arima, rmse_arima = {}, {}, {}
predictions_arima = pd.DataFrame()

# Fit a SARIMA model for each stock.
tickers = list(daily_data)
for ticker in tickers:
    # Create a new dataframe containing only the data of such stock.
    data = daily_data[ticker].iloc[len(daily_data)-365]
    
    # Split the observations in training and test data.
    train_data = data.iloc[:len(data)-60]
    test_data = data.iloc[len(data)-60:]
    
    # Fit the model.
    model = pm.auto_arima(train_data,
                          m=7,
                          seasonal=True,
                          test='adf',
                          start_p=0, start_q=0,
                          max_p=12, max_q=12,
                          D=1,
                          start_P=1, start_Q=1,
                          trace=True,
                          error_action='ignore',
                          supress_warnings=True,
                          stepwise=True)
    
    # Use the model to get predictions.
    predictions = pd.DataFrame(model.predict(n_periods=len(test_data)))
    predictions.index = test_data.index
    predictions.columns = [ticker]
    predictions_arima = pd.concat([predictions_arima, predictions], axis=1,
                                  join='outer')
    
    # Plot both data samples along with the predictions.
    plt.plot(train_data, color='blue', label='Train Data')
    plt.plot(test_data, color='green', label='Test Data')
    plt.plot(predictions, color='red', label='Predicted Values')
    plt.legend()
    plt.show()
    
    # Store the goodness-of-fit measures of the model.
    mse_arima[ticker] = mean_squared_error(test_data, predictions)
    mae_arima[ticker] = mean_absolute_error(test_data, predictions)
    rmse_arima[ticker] = math.sqrt(mean_squared_error(test_data, predictions))
    
# TRAIN THE CONVOLUTIONAL NEURAL NETWORKS
# Define a function that converts a string to datetime format.
def str_to_datetime(string):
    split = string.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

# Create two functions that are going to be used to transform data.
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    
    target_date = first_date
    
    dates = []
    X, Y = [], []
    
    last_time = False
    while True:
        df_subset = dataframe.iloc[:target_date].tail(n+1)
        
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return
        
        values = df_subset.to_numpy()
        x, y = values[:-1], values[-1]
        
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        
        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
        
        target_date = next_date
        
        if target_date == last_date:
            last_time = True
        
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]
        
    ret_df['Target'] = Y
    
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    
    dates = df_as_np[:, 0]
    
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    
    Y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), Y.astype(np.float32)

# Create three empty dictionaries and a dataframe to store the goodness-of-fit
# measures and the predictions.
mse_conv, mae_conv, rmse_conv = {}, {}, {}
predictions_conv = pd.DataFrame()

# Train the convolutional neural networks.
for ticker in tickers:
    # Create a new dataframe with the data of such stock.
    data = daily_data[ticker]
    
    # Transform the data.
    windowed_data = df_to_windowed_df(data, '2018-02-02', '2024-03-13', n=20)
    dates, X, y = windowed_df_to_date_X_y(windowed_data)
    
    # Split the data into train, validation and test samples.
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
    # Train the CNN.
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimize='adam', loss='mse')
    
    # Fit the model.
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    
    # Get predictions for the test sample.
    test_predictions = model.predict(X_test).flatten()
    predictions = pd.DataFrame(test_predictions, index=dates_test)
    predictions.columns = [ticker]
    predictions_conv = pd.concat([predictions_conv, predictions], axis=1,
                                 join='outer')
    
    # Plot train, validation and test data as well as the predictions.
    plt.plot(dates_train, y_train, label='Train Data', color='yellow')
    plt.plot(dates_val, y_val, label='Validation data', color='blue')
    plt.plot(dates_test, y_test, label='Test data', color='green')
    plt.plot(dates_test, test_predictions, label='Predictions', color='red')
    plt.legend()
    plt.plot()
    
    # Store the goodness-of-fit measures.
    mse_conv[ticker] = mean_squared_error(y_test, test_predictions)
    mae_conv[ticker] = mean_absolute_error(y_test, test_predictions)
    rmse_conv[ticker] = math.sqrt(mean_squared_error(y_test, test_predictions))
    
# TRAIN THE LSTM NEURAL NETWORKS
# Create three empty dictionaries and a dataframe to store the goodness-of-fit
# measures and the predictions.
mse_lstm, mae_lstm, rmse_lstm = {}, {}, {}
predictions_lstm = pd.DataFrame()

# Train the LSTM neural networks.
for ticker in tickers:
    # Create a new dataframe with the data of such stock.
    data = daily_data[ticker]
    
    # Transform the data.
    windowed_data = df_to_windowed_df(data, '2018-02-02', '2024-03-13', n=20)
    dates, X, y = windowed_df_to_date_X_y(windowed_data)
    
    # Split the data into train, validation and test samples.
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
    # Train the LSTM neural network.
    model = Sequential([layers.Input((20, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])
    
    # Fit the model.
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    
    # Get predictions for the test sample.
    test_predictions = model.predict(X_test).flatten()
    predictions = pd.DataFrame(test_predictions, index=dates_test)
    predictions_columns = [ticker]
    predictions_lstm = pd.concat([predictions_lstm, predictions], axis=1,
                                 join='outer')
    
    # Plot train, validation and test samples as well as the predictions.
    plt.plot(dates_train, y_train, label='Train Data', color='yellow')
    plt.plot(dates_val, y_val, label='Validation Data', color='blue')
    plt.plot(dates_test, y_test, label='Test Data', color='green')
    plt.plot(dates_test, test_predictions, label='Predictions', color='red')
    plt.legend()
    plt.show()
    
    # Store the goodness-of-fit measures.
    mse_lstm[ticker] = mean_squared_error(y_test, test_predictions)
    mae_lstm[ticker] = mean_absolute_error(y_test, test_predictions)
    rmse_lstm[ticker] = math.sqrt(mean_squared_error(y_test, test_predictions))
    
# EVALUATE THE GOODNESS OF FIT OF EACH MODEL.
# Calculate the trimmed mean for each goodness-of-fit measure of SARIMA models.
E_mse_arima = trim_mean(list(mse_arima.values()), 0.2)
E_mae_arima = trim_mean(list(mae_arima.values()), 0.2)
E_rmse_arima = trim_mean(list(rmse_arima.values()), 0.2)

# Calculate the trimmed mean for each goodness-of-fit measure of CNNs.
E_mse_conv = trim_mean(list(mse_conv.values()), 0.2)
E_mae_conv = trim_mean(list(mae_conv.values()), 0.2)
E_rmse_conv = trim_mean(list(rmse_conv.values()), 0.2)

# Calculate the trimmed mean for each goodness-of-fit measure of LSTM NNs.
E_mse_lstm = trim_mean(list(mse_lstm.values()), 0.2)
E_mae_lstm = trim_mean(list(mae_lstm.values()), 0.2)
E_rmse_lstm = trim_mean(list(rmse_lstm.values()), 0.2)

# Create three lists with the data.
E_mse_nn = [E_mse_conv, E_mse_lstm]
E_mae_nn = [E_mae_conv, E_mae_lstm]
E_rmse_nn = [E_rmse_conv, E_rmse_lstm]

# Create a list with both NN models.
models = ['Convolutional', 'LSTM']

# Create a dataframe with the data of NNs.
dict_nn = {'Model':models, 
        'Mean Squared Error':E_mse_nn, 
        'Mean Absolute Error':E_mae_nn,
        'Root Mean Squared Error':E_rmse_nn}
goodness_nn = pd.DataFrame(dict_nn)

# Plot.
ax1 = goodness_nn.plot(x = 'Model',
                 kind = 'bar',
                 stacked = False,
                 rot = 0.90)
ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax1.plot()

# Create a dataframe with SARIMA data.
dict = {'Model':'SARIMA', 
        'Mean Squared Error':E_mse_arima,
        'Mean Absolute Error':E_mae_arima,
        'Root Mean Squared Error':E_rmse_arima}
goodness = pd.DataFrame(dict, index=[0])

# Plot.
ax2 = goodness.plot(x = 'Model', 
              kind = 'bar', 
              stacked = False,
              rot = 0.90)
ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax2.plot()

# BUILD PORTFOLIOS AND GET SHARPE RATIO FOR EACH OF THEM.
# Isolate predictions for february.
pred_arima_feb = pd.DataFrame(predictions_arima.loc['2024-02-29',:])
pred_arima_feb.columns = ['2024-02-29']
pred_conv_feb = pd.DataFrame(predictions_conv.loc['2024-02-29',:])
pred_conv_feb.columns = ['2024-02-29']
pred_lstm_feb = pd.DataFrame(predictions_lstm.loc['2024-02-29',:])
pred_lstm_feb.columns = ['2024-02-29']

# Get the real stock prices for the previous month.
actual_prices = pd.DataFrame(daily_data.loc['2024-01-31',:])
actual_prices.columns = ['2024-01-31']

# Combine both dataframes.
exp_returns_arima = pd.concat([actual_prices, pred_arima_feb], axis=1, 
                              join='outer')
exp_returns_conv = pd.concat([actual_prices, pred_conv_feb], axis=1,
                             join='outer')
exp_returns_lstm = pd.concat([actual_prices, pred_lstm_feb], axis=1,
                             join='outer')

# Calculate the expected returns.
exp_returns_arima['Expected Returns'] = exp_returns_arima.iloc[:,1]-exp_returns_arima.iloc[:,0]
exp_returns_conv['Expected Returns'] = exp_returns_conv.iloc[:,1]-exp_returns_conv.iloc[:,0]
exp_returns_lstm['Expected Returns'] = exp_returns_lstm.iloc[:,1]-exp_returns_lstm.iloc[:,0]

# Drop columns that are not required and sort in descending order.
exp_returns_arima = exp_returns_arima['Expected Returns'].sort_values(
    ascending=False)
exp_returns_conv = exp_returns_conv['Expected Returns'].sort_values(
    ascending=False)
exp_returns_lstm = exp_returns_lstm['Expected Returns'].sort_values(
    ascending=False)

# Get the stocks corresponding to the first decile.
index_decile = int(len(exp_returns_arima) * 0.1)
arima_decile = exp_returns_arima.index[:index_decile]
conv_decile = exp_returns_conv.index[:index_decile]
lstm_decile = exp_returns_lstm.index[:index_decile]

# Create a dataframe with the data of the stocks in the first decile.
arima_decile_data = daily_data[arima_decile].loc['2024-01-31':'2024-02-29']
conv_decile_data = daily_data[conv_decile].loc['2024-01-31':'2024-02-29']
lstm_decile_data = daily_data[lstm_decile].loc['2024-01-31':'2024-02-29']

# Normalize the returns.
for column in arima_decile_data.columns:
    arima_decile_data[column] = arima_decile_data[column]/arima_decile_data.iloc[0][column]
for column in conv_decile_data.columns:
    conv_decile_data[column] = conv_decile_data[column]/conv_decile_data.iloc[0][column]
for column in lstm_decile_data.columns:
    lstm_decile_data[column] = lstm_decile_data[column]/lstm_decile_data.iloc[0][column]
    
# Get the allocation assuming equal share.
allocation = 1 / len(arima_decile_data.axes[1])
for column in arima_decile_data.columns:
    arima_decile_data[column] = arima_decile_data[column] * allocation
for column in conv_decile_data.columns:
    conv_decile_data[column] = conv_decile_data[column] * allocation
for column in lstm_decile_data.columns:
    lstm_decile_data[column] = lstm_decile_data[column] * allocation
    
# Calculate the position assuming a portfolio value of 10.000 euros.
for column in arima_decile_data.columns:
    arima_decile_data[column] = arima_decile_data[column] * 10000
for column in conv_decile_data.columns:
    conv_decile_data[column] = conv_decile_data[column] * 10000
for column in lstm_decile_data.columns:
    lstm_decile_data[column] = lstm_decile_data[column] * 10000
    
# Import Euronext 100 index data.
path = 'euronxt_100_data.xlsx'
index_data = pd.read_excel(path)
index_data.index = index_data.pop('Date')

# Normalize the returns.
index_data = index_data/index_data.iloc[0]

# Calculate the position assuming an investment of 10.000 euros.
index_data = index_data * 10000
    
# Plot the evolution of portfolios and index performance.
arima_decile_data['Total Pos'] = arima_decile_data.sum(axis=1)
conv_decile_data['Total Pos'] = conv_decile_data.sum(axis=1)
lstm_decile_data['Total Pos'] = lstm_decile_data.sum(axis=1)

arima_decile_data['Total Pos'].plot(label='ARIMA')
conv_decile_data['Total Pos'].plot(label='CONVOLUTIONAL')
lstm_decile_data['Total Pos'].plot(label='LSTM')
index_data.plot(label='Euronext 100 Index')
plt.xticks(ticks=arima_decile_data.index.values, 
           labels=arima_decile_data.index,
           rotation=270)
plt.xlabel('Date')
plt.ylabel('Euros')
plt.legend()
plt.show()

# Calculate daily returns.
arima_decile_data['Daily Returns'] = arima_decile_data['Total Pos'].pct_change(1)
conv_decile_data['Daily Returns'] = conv_decile_data['Total Pos'].pct_change(1)
lstm_decile_data['Daily Returns'] = lstm_decile_data['Total Pos'].pct_change(1)

# Calculate the Sharpe ratio.
arima_sharpe = (arima_decile_data['Daily Returns'].mean()-0.0253)/arima_decile_data['Daily Returns'].std()
conv_sharpe = (conv_decile_data['Daily Returns'].mean()-0.0253)/conv_decile_data['Daily Returns'].std()
lstm_sharpe = (lstm_decile_data['Daily Returns'].mean()-0.0253)/lstm_decile_data['Daily Returns'].std()

# Plot.
plt.bar(x='SARIMA', height=arima_sharpe)
plt.bar(x='Convolutional', height=conv_sharpe)
plt.bar(x='LSTM', height=lstm_sharpe)
plt.xlabel('Portfolio')
plt.show()