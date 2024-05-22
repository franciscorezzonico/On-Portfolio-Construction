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

