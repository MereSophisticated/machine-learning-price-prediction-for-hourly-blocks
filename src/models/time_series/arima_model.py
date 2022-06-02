from cProfile import label
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from inspect import trace
# from msilib.schema import Error
from re import X
from tracemalloc import start
from turtle import color

import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np
from yaml import serialize

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import sys
sys.path.insert(0, "src/data_analysis")

from data_retrieval import get_intra_day_data_for_region
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pmdarima.arima import ADFTest
from datetime import datetime

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from pmdarima import auto_arima, AutoARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from spectrum import Periodogram, data_cosine

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import signal
import os

from Ignore_warnings import suppress_stdout_stderr

# All dates
# start_date = "2021-11-08"
# end_date = "2022-05-01"

#time_1 is a random specific date
time_1 = "2022-02-03 22:00:00"
pd.set_option('display.expand_frame_repr', False)
df_intra_day = get_intra_day_data_for_region("GERMANY")

df_intra_day = df_intra_day.sort_values(by=['trd_execution_time',
                                            'trd_buy_delivery_area',
                                            'trd_sell_delivery_area'])

# Collects data for a specific hour block (X is the hour time before the end date of the trade) and (y is the lenght of the prediction)
def One_block_X_h(time_1, X=5, y=2):
    df_single_block = df_intra_day[df_intra_day['trd_delivery_time_start'] == time_1]

    df_single_block['diff'] = (df_single_block['trd_delivery_time_start'] - df_single_block['trd_execution_time'])

    all_data = df_single_block[df_single_block['diff'] <= pd.Timedelta(X, unit='H')]
    train_data = all_data[all_data['diff'] >= pd.Timedelta(y, unit='H')]
    last_data = all_data[all_data['diff'] <= pd.Timedelta(y, unit='H')]

    # all_data.iloc[::1, :].plot(x='trd_execution_time', y='trd_price',
    #                                   title= str(time_1),
    #                                   xlabel="Time", ylabel="Price", legend=False)
    # plt.show()
    return all_data, train_data, last_data


####################################################################################################################################################################################################################################################################
#Plot for prediction

def get_intra_day(df_intra_day, start_date='2022-02-02', end_date='2022-03-24'):
    
    # Time to start of block from purchase time (execution time)
    df_intra_day['diff'] = (df_intra_day['trd_delivery_time_start']
                            - df_intra_day['trd_execution_time'])

    df_intra_day = df_intra_day[(df_intra_day['trd_delivery_time_start']
                                 >= start_date)
                                & (df_intra_day['trd_delivery_time_end']
                                   <= f'{end_date} 00:00:00')]
    df_intra_day = df_intra_day[df_intra_day['diff'] <= pd.Timedelta(1, unit='H')]

    df_intra_day = df_intra_day.resample('H', on='trd_delivery_time_start').mean()
    df_intra_day.reset_index(inplace=True)

    # Some products might not be traded in the hour before closing, so drop those
    df_intra_day.dropna(inplace=True)
    return df_intra_day


def difference(start_date='2022-02-02', end_date='2022-03-24'):
    drugi = np.diff(get_intra_day(df_intra_day, start_date, end_date)["trd_price"].values)
    drugi = np.insert(drugi, 0, 0)

    df = get_intra_day(df_intra_day, start_date, end_date)
    df["difference"] = drugi

    dftest = adfuller(df["difference"], autolag='AIC')
    # print("P-Value test is = ", dftest[1])
    
    if dftest[1] <= 0.05:
        stepwise_fit = auto_arima(df["difference"], trace=False, suppress_warnings=True)
        print(stepwise_fit)
        stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
        order = (int(stepwise_fit[0]), 0, int(stepwise_fit[2][0]))
    else: 
        print("ADF test value to high")

    # Plot difference
    df.plot(x='trd_delivery_time_start', y='difference')
    plt.show()

    train = df.iloc[:-24]
    test = df.iloc[-24:]
    return (train, test, drugi, order)


#function that draws graphs for intra day 24h prediction

def difference_predicition(start_date='2022-02-02', end_date='2022-03-24'):
    train, test, drugi, order = difference(start_date, end_date)
    all_data = get_intra_day(df_intra_day, start_date, end_date)[["trd_delivery_time_start", "trd_price"]]
    # all_data.dropna(inplace=True)
    all_data.reset_index(inplace=True)

    mod = sm.tsa.arima.ARIMA(train['difference'].values, order=order)
    res = mod.fit()

    start = len(train)
    end = len(all_data) - 1
    pred = res.predict(start=start, end=end, typ='levels')
    

    plt.plot(pred, label='ARIMA prediction')
    plt.plot(test['difference'].values, label='Data')
    plt.xlabel("Time (Hour Blocks)")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    drugi[0] = (get_intra_day(df_intra_day, start_date, end_date)["trd_price"].values[0])
    result = (drugi.cumsum())

    pred[0] = result[-24]
    prediction = pred.cumsum()

    novi = [None] * len(all_data)
    novi[-24:] = prediction

    
    plt.plot(prediction, label="prediction")
    plt.plot(result[-24:], label="data")
    plt.legend()
    plt.show()

    plt.plot(result, label="Data")
    plt.plot(novi, label="prediciton")
    plt.show()


    ## Plots with dates
    nans = (len(train)) * [np.nan]
    valuess = prediction
    predictionn = np.concatenate((nans, valuess))
    all_data["prediction"] = pd.Series(predictionn)
    
    ax = all_data.plot(x="trd_delivery_time_start", y="prediction", color="red", label="Napoved s pomočjo funkcijo Auto ARIMA")
    all_data.plot(x="trd_delivery_time_start", y="trd_price", color="#1874CD", label="Resnični podatki", ax=ax,  linewidth=1)
    plt.xlabel("Čas")
    plt.ylabel("Vrednosti")
    plt.legend()
    plt.show()


    ### Test how good the prediction is, very big rmse :/
    # # print(test['difference'].values.mean())
    # rmse = sqrt(mean_squared_error(pred, test['difference'].values))
    # print(rmse)


################################################################################################################################################################################
################################################################################################################################################################################

## Export data for R

def data_pandas_all(start_date='2021-11-08', end_date='2022-05-01'):
    drugi = get_intra_day(start_date, end_date)[["trd_price", "trd_delivery_time_start"]]
    drugi.to_csv("All_data.csv")


################################################################################################################################################################################
################################################################################################################################################################################

# Hourly Blocks prediction

def arima_model(time_1, X=5, y=2):
    all_data, train_data, last_data = One_block_X_h(time_1, X, y)
    adf_test = ADFTest(alpha=0.05)

    train_data = train_data[["trd_execution_time", "trd_price"]]
    all_data = all_data[["trd_execution_time", "trd_price"]]
    train_data["diff"] = train_data["trd_price"]
    stevec = 0

    # print(all_data["diff"].values)
    # while adf_test.should_diff(train_data["diff"].values)[1]:
    #     train_data["diff"] = train_data["diff"].diff()
    #     train_data.loc[1,"diff"] = 0
    #     stevec += 1
        # print(stevec)

    train_data.reset_index(inplace=True)
    all_data.reset_index(inplace=True)

    stepwise_fit = auto_arima(train_data["diff"], trace=False, suppress_warnings=True, seasonal=False)
    print(stepwise_fit)

    stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
    order = (int(stepwise_fit[0]), stevec, int(stepwise_fit[2][0]))
    
    mod = ARIMA(train_data["diff"], order=order)
    res = mod.fit()

    all_data["fit"] = res.fittedvalues

    end = len(all_data)
    start = len(train_data)
    pred = res.predict(start=start, end=end, typ='levels', color="red")
    
    nans = (len(train_data)) * [np.nan]
    valuess = pred.values
    predictionn = np.concatenate((nans, valuess))
    all_data["prediction"] = pd.Series(predictionn)
    
    ### Plot

    # ax = all_data.plot(x="trd_execution_time", y="prediction", color="red", label="Prediciton")
    # all_data.plot(x="trd_execution_time", y="trd_price", color="blue", label="data", ax=ax)
    # plt.legend()
    # plt.show()


    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = y * 120 # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))

    df.reset_index(inplace=True)

    test_size = number_of_predictions
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    # stepwise_fit = auto_arima(train_df["trd_price_mean"], trace=False, suppress_warnings=True, seasonal=False)

    # print(stepwise_fit)
    # stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
    # order = (int(stepwise_fit[0]), stevec, int(stepwise_fit[2][0]))
    # print(order)

    mod1 = ARIMA(train_df["trd_price_mean"], order=order)
    res1 = mod1.fit()

    end = len(train_df) + len(test_df)
    start = len(train_df)
    pred1 = res.predict(start=start, end=end, typ='levels', color="red")
    pred2 = res1.predict(start=start, end=end, typ='levels', color="red")

    nans = (len(train_df)) * [np.nan]
    valuess1 = pred1.values
    valuess2 = pred2.values
    predictionn1 = np.concatenate((nans, valuess1))
    predictionn2 = np.concatenate((nans, valuess2))
    df["prediction1"] = pd.Series(predictionn1)
    df["prediction2"] = pd.Series(predictionn2)
    

    ax = df.plot(x="trd_execution_time", y="prediction2", color="red", label="Prediciton on resampled data")
    df.plot(x="trd_execution_time", y="prediction1", color="orange", label="Prediciton on data", ax=ax)
    df.plot(x="trd_execution_time", y="trd_price_mean", color="blue", label="data", ax=ax)
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # One_block_X_h(time_1, X=5, y=2)
    # get_intra_day(df_intra_day, start_date='2022-02-02', end_date='2022-03-24')
    # difference(start_date='2022-02-02', end_date='2022-03-24')
    # difference_predicition(start_date='2022-02-02', end_date='2022-03-24')
    # arima_model(time_1, X=5, y=2)

    pass