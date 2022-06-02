from random import seed
from turtle import done
import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np
from yaml import serialize

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import sys
sys.path.insert(0, "src/data_analysis")

from data_retrieval import get_intra_day_data_for_region, get_intra_day
from arima_model import One_block_X_h
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pmdarima.arima import ADFTest
from datetime import datetime, timedelta

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima, AutoARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from spectrum import Periodogram, data_cosine

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import signal
import os

import orbit
from orbit.utils.dataset import load_iclaims
from orbit.models import ETS, DLT, LGT
from orbit.diagnostics.plot import plot_predicted_data


# import necessary functions that I will use in this demonstartion
## Orbit plot functions for EDA 
from orbit.eda import eda_plot

## Orbit Themed Palette and styles 
import orbit.constants.palette as palette
from orbit.utils.plot import get_orbit_style

# Orbit models and diagnostic plots 
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components

# Orbit Backtest tools 
from orbit.diagnostics.backtest import BackTester, TimeSeriesSplitter
from orbit.diagnostics.plot import plot_bt_predictions
from orbit.diagnostics.metrics import smape, wmape
# Orbit plotting backtest with gif
from orbit.diagnostics.plot import plot_bt_predictions2



from Ignore_warnings import suppress_stdout_stderr


# DLT model
def Orbit_model_DLT(time_1="2021-11-11 00:00:00", X=30, y=2):
    all_data, train_data, last_data = One_block_X_h(time_1, X, y)
    # all_data_yesterday, b, c = One_block_X_h("2021-11-10 23:00:00",X,y)
    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = y * 120 # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))


    test_size = number_of_predictions
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    dlt = DLT(
    response_col='trd_price_mean', date_col="trd_execution_time") #, regressor_col=['prices'], regressor_sign=["="])
    #   regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
    #   seasonality=52) 
    with suppress_stdout_stderr():
        dlt.fit(df=train_df)

    # outcomes data frame
    predicted_df_DLT = dlt.predict(df=test_df)

    plot_predicted_data(
    training_actual_df=train_df, predicted_df=predicted_df_DLT,
    date_col=dlt.date_col, actual_col=dlt.response_col,
    test_actual_df=test_df)

    last_pred_value_DLT = predicted_df_DLT.prediction.values[-1]
    last_all_value = all_data.trd_price.values[-1]



########################################################################################################################################################################################################################
# ETS model
# Predictions are made for this 
def Orbit_model_ETS(sez_ETS, time_1="2021-11-11 00:00:00", X=30, y=2):
    
    stevec = 0 
    stevec_pred = 0
    all_data, train_data, last_data = One_block_X_h(time_1, X, y)
    # yesterday = time_1 - timedelta(1)
    # all_data_yesterday = One_block_X_h("2021-11-10 23:00:00",X,y)
    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = int(y * 120) # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))

    # df.dropna(inplace=True)
    df.reset_index(inplace=True)

    test_size = number_of_predictions
    train_df = df[:-test_size]
    test_df = df[-test_size:]


    ets = ETS(
    response_col='trd_price_mean', date_col="trd_execution_time")

    try:
        with suppress_stdout_stderr():
            ets.fit(df=train_df)
    except IndexError:
        sez_ETS.append("N")
        return sez_ETS
        

    # outcomes data frame
    predicted_df = ets.predict(df=test_df)

    # plot_predicted_data(
    # training_actual_df=train_df, predicted_df=predicted_df,
    # date_col=ets.date_col, actual_col=ets.response_col,
    # test_actual_df=test_df)

    last_pred_value_ETS = predicted_df.prediction.values[-1]
    last_all_value = all_data.trd_price.values[-1]

    ####### ETS
    ### 10 procentov
    if last_pred_value_ETS >= 0:
        last_10_prct_upp_ETS = last_pred_value_ETS * 1.1
        last_10_prct_low_ETS = last_pred_value_ETS * 0.9
    else:
        last_10_prct_upp_ETS = last_pred_value_ETS * 0.9
        last_10_prct_low_ETS = last_pred_value_ETS * 1.1
    if last_10_prct_low_ETS <= last_all_value and last_10_prct_upp_ETS >= last_all_value:
        ten_prct_ETS = 1
    else:
        ten_prct_ETS = 0
    ## 20 procentov
    if last_pred_value_ETS >= 0:
        last_20_prct_upp_ETS = last_pred_value_ETS * 1.2
        last_20_prct_low_ETS = last_pred_value_ETS * 0.8
    else:
        last_20_prct_upp_ETS = last_pred_value_ETS * 0.8
        last_20_prct_low_ETS = last_pred_value_ETS * 1.2
    if last_20_prct_low_ETS <= last_all_value and last_20_prct_upp_ETS >= last_all_value:
        twenty_prct_ETS = 1
    else:
        twenty_prct_ETS = 0

    ## 30 procentov
    if last_pred_value_ETS >= 0:
        last_30_prct_upp_ETS = last_pred_value_ETS * 1.3
        last_30_prct_low_ETS = last_pred_value_ETS * 0.7
    else:
        last_30_prct_upp_ETS = last_pred_value_ETS * 0.7
        last_30_prct_low_ETS = last_pred_value_ETS * 1.3
    if last_30_prct_low_ETS <= last_all_value and last_30_prct_upp_ETS >= last_all_value:
        thirty_prct_ETS = 1
    else:
        thirty_prct_ETS = 0

    ## 50 procentov
    if last_pred_value_ETS >= 0:
        last_50_prct_upp_ETS = last_pred_value_ETS * 1.5
        last_50_prct_low_ETS = last_pred_value_ETS * 0.5
    else:
        last_50_prct_upp_ETS = last_pred_value_ETS * 0.5
        last_50_prct_low_ETS = last_pred_value_ETS * 1.5
    if last_50_prct_low_ETS <= last_all_value and last_50_prct_upp_ETS >= last_all_value:
        fifty_prct_ETS = 1
    else:
        fifty_prct_ETS = 0

    ## 100 procentov
    if last_pred_value_ETS >= 0:
        last_100_prct_upp_ETS = last_pred_value_ETS * 2
        last_100_prct_low_ETS = last_pred_value_ETS * 0
    else:
        last_100_prct_upp_ETS = last_pred_value_ETS * 0
        last_100_prct_low_ETS = last_pred_value_ETS * 2
    if last_100_prct_low_ETS <= last_all_value and last_100_prct_upp_ETS >= last_all_value:
        hundred_prct_ETS = 1
    else:
        hundred_prct_ETS = 0

    Eror_ETS = last_all_value - last_pred_value_ETS
    sez_ETS.append((Eror_ETS, ten_prct_ETS, twenty_prct_ETS, thirty_prct_ETS, fifty_prct_ETS, hundred_prct_ETS))
    print(time_1)
    return sez_ETS



########################################################################################################################################################################################################



# LGT model
def Orbit_model_LGT(time_1="2021-11-11 00:00:00", X=30, y=2):
    all_data, train_data, last_data = One_block_X_h(time_1, X, y)

    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = y * 120 # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))

    # df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df["indexs"] = df.index[:].values

    test_size = number_of_predictions
    train_df = df[:-test_size]
    test_df = df[-test_size:]


    lgt = LGT(
    response_col='trd_price_mean', date_col="trd_execution_time")
    #   regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
    #   seasonality=52) 
    with suppress_stdout_stderr():
        lgt.fit(df=train_df)

    # outcomes data frame
    predicted_df = lgt.predict(df=test_df)

    plot_predicted_data(
    training_actual_df=train_df, predicted_df=predicted_df,
    date_col=lgt.date_col, actual_col=lgt.response_col,
    test_actual_df=test_df)

    last_pred_value = predicted_df.prediction.values[-1]
    last_all_value = all_data.trd_price.values[-1]

    print(last_all_value, last_pred_value)



# All three models
def Orbit_model(date="2021-11-11 00:00:00", X=30, y=2, sez_DLT = [], sez_ETS = [], sez_LGT = []):
    all_data, train_data, last_data = One_block_X_h(date, X, y)

    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = y * 120 # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))

    # df.dropna(inplace=True)
    df.reset_index(inplace=True)

    last_all_value = all_data.trd_price.values[-1]

    test_size = number_of_predictions
    train_df = df[:-test_size]
    test_df = df[-test_size:]

##### DLT Version
    dlt = DLT(
    response_col='trd_price_mean', date_col="trd_execution_time")
    #   regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
    #   seasonality=52) 
    with suppress_stdout_stderr():
        dlt.fit(df=train_df)

    # outcomes data frame
    predicted_df_DLT = dlt.predict(df=test_df)

    # plot_predicted_data(
    # training_actual_df=train_df, predicted_df=predicted_df_DLT,
    # date_col=dlt.date_col, actual_col=dlt.response_col,
    # test_actual_df=test_df)

    last_pred_value_DLT = predicted_df_DLT.prediction.values[-1]
    
##### ETS version
    ets = ETS(
    response_col='trd_price_mean', date_col="trd_execution_time")
    #   regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
    #   seasonality=52) 
    with suppress_stdout_stderr():
        ets.fit(df=train_df)

    # outcomes data frame
    predicted_df_ETS = ets.predict(df=test_df)

    # plot_predicted_data(cted_df_ETS,
    # date_col=ets.date_col, actual_col=ets.response_col,
    # test_actual_df=test_df)
    training_actual_df=train_df, predicted_df=predi

    last_pred_value_ETS = predicted_df_ETS.prediction.values[-1]

###### LGT version
    lgt = LGT(
    response_col='trd_price_mean', date_col="trd_execution_time", )
    #   regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
    #   seasonality=52) 
    with suppress_stdout_stderr():
        lgt.fit(df=train_df)

    # outcomes data frame
    predicted_df_LGT = lgt.predict(df=test_df)

    # plot_predicted_data(
    # training_actual_df=train_df, predicted_df=predicted_df_LGT,
    # date_col=lgt.date_col, actual_col=lgt.response_col,
    # test_actual_df=test_df)

    last_pred_value_LGT= predicted_df_LGT.prediction.values[-1]



    ####### DLT 
    ### 10 procentov
    if last_pred_value_DLT >= 0:
        last_10_prct_upp_DLT = last_pred_value_DLT * 1.1
        last_10_prct_low_DLT = last_pred_value_DLT * 0.9
    else:
        last_10_prct_upp_DLT = last_pred_value_DLT * 0.9
        last_10_prct_low_DLT = last_pred_value_DLT * 1.1
    if last_10_prct_low_DLT <= last_all_value and last_10_prct_upp_DLT >= last_all_value:
        ten_prct_DLT = 1
    else:
        ten_prct_DLT = 0
    ## 20 procentov
    if last_pred_value_DLT >= 0:
        last_20_prct_upp_DLT = last_pred_value_DLT * 1.2
        last_20_prct_low_DLT = last_pred_value_DLT * 0.8
    else:
        last_20_prct_upp_DLT = last_pred_value_DLT * 0.8
        last_20_prct_low_DLT = last_pred_value_DLT * 1.2
    if last_20_prct_low_DLT <= last_all_value and last_20_prct_upp_DLT >= last_all_value:
        twenty_prct_DLT = 1
    else:
        twenty_prct_DLT = 0

    ## 30 procentov
    if last_pred_value_DLT >= 0:
        last_30_prct_upp_DLT = last_pred_value_DLT * 1.3
        last_30_prct_low_DLT = last_pred_value_DLT * 0.7
    else:
        last_30_prct_upp_DLT = last_pred_value_DLT * 0.7
        last_30_prct_low_DLT = last_pred_value_DLT * 1.3
    if last_30_prct_low_DLT <= last_all_value and last_30_prct_upp_DLT >= last_all_value:
        thirty_prct_DLT = 1
    else:
        thirty_prct_DLT = 0

    ## 50 procentov
    if last_pred_value_DLT >= 0:
        last_50_prct_upp_DLT = last_pred_value_DLT * 1.5
        last_50_prct_low_DLT = last_pred_value_DLT * 0.5
    else:
        last_50_prct_upp_DLT = last_pred_value_DLT * 0.5
        last_50_prct_low_DLT = last_pred_value_DLT * 1.5
    if last_50_prct_low_DLT <= last_all_value and last_50_prct_upp_DLT >= last_all_value:
        fifty_prct_DLT = 1
    else:
        fifty_prct_DLT = 0

    ## 100 procentov
    if last_pred_value_DLT >= 0:
        last_100_prct_upp_DLT = last_pred_value_DLT * 2
        last_100_prct_low_DLT = last_pred_value_DLT * 0
    else:
        last_100_prct_upp_DLT = last_pred_value_DLT * 0
        last_100_prct_low_DLT = last_pred_value_DLT * 2
    if last_100_prct_low_DLT <= last_all_value and last_100_prct_upp_DLT >= last_all_value:
        hundred_prct_DLT = 1
    else:
        hundred_prct_DLT = 0


################################################################################################################################################################################
    ####### ETS
    ### 10 procentov
    if last_pred_value_ETS >= 0:
        last_10_prct_upp_ETS = last_pred_value_ETS * 1.1
        last_10_prct_low_ETS = last_pred_value_ETS * 0.9
    else:
        last_10_prct_upp_ETS = last_pred_value_ETS * 0.9
        last_10_prct_low_ETS = last_pred_value_ETS * 1.1
    if last_10_prct_low_ETS <= last_all_value and last_10_prct_upp_ETS >= last_all_value:
        ten_prct_ETS = 1
    else:
        ten_prct_ETS = 0
    ## 20 procentov
    if last_pred_value_ETS >= 0:
        last_20_prct_upp_ETS = last_pred_value_ETS * 1.2
        last_20_prct_low_ETS = last_pred_value_ETS * 0.8
    else:
        last_20_prct_upp_ETS = last_pred_value_ETS * 0.8
        last_20_prct_low_ETS = last_pred_value_ETS * 1.2
    if last_20_prct_low_ETS <= last_all_value and last_20_prct_upp_ETS >= last_all_value:
        twenty_prct_ETS = 1
    else:
        twenty_prct_ETS = 0

    ## 30 procentov
    if last_pred_value_ETS >= 0:
        last_30_prct_upp_ETS = last_pred_value_ETS * 1.3
        last_30_prct_low_ETS = last_pred_value_ETS * 0.7
    else:
        last_30_prct_upp_ETS = last_pred_value_ETS * 0.7
        last_30_prct_low_ETS = last_pred_value_ETS * 1.3
    if last_30_prct_low_ETS <= last_all_value and last_30_prct_upp_ETS >= last_all_value:
        thirty_prct_ETS = 1
    else:
        thirty_prct_ETS = 0

    ## 50 procentov
    if last_pred_value_ETS >= 0:
        last_50_prct_upp_ETS = last_pred_value_ETS * 1.5
        last_50_prct_low_ETS = last_pred_value_ETS * 0.5
    else:
        last_50_prct_upp_ETS = last_pred_value_ETS * 0.5
        last_50_prct_low_ETS = last_pred_value_ETS * 1.5
    if last_50_prct_low_ETS <= last_all_value and last_50_prct_upp_ETS >= last_all_value:
        fifty_prct_ETS = 1
    else:
        fifty_prct_ETS = 0

    ## 100 procentov
    if last_pred_value_ETS >= 0:
        last_100_prct_upp_ETS = last_pred_value_ETS * 2
        last_100_prct_low_ETS = last_pred_value_ETS * 0
    else:
        last_100_prct_upp_ETS = last_pred_value_ETS * 0
        last_100_prct_low_ETS = last_pred_value_ETS * 2
    if last_100_prct_low_ETS <= last_all_value and last_100_prct_upp_ETS >= last_all_value:
        hundred_prct_ETS = 1
    else:
        hundred_prct_ETS = 0



################################################################################################################################################################################
    ####### LGT
    ### 10 procentov
    if last_pred_value_LGT >= 0:
        last_10_prct_upp_LGT = last_pred_value_LGT * 1.1
        last_10_prct_low_LGT = last_pred_value_LGT * 0.9
    else:
        last_10_prct_upp_LGT = last_pred_value_LGT * 0.9
        last_10_prct_low_LGT = last_pred_value_LGT * 1.1
    if last_10_prct_low_LGT <= last_all_value and last_10_prct_upp_LGT >= last_all_value:
        ten_prct_LGT = 1
    else:
        ten_prct_LGT = 0
    ## 20 procentov
    if last_pred_value_LGT >= 0:
        last_20_prct_upp_LGT = last_pred_value_LGT * 1.2
        last_20_prct_low_LGT = last_pred_value_LGT * 0.8
    else:
        last_20_prct_upp_LGT = last_pred_value_LGT * 0.8
        last_20_prct_low_LGT = last_pred_value_LGT * 1.2
    if last_20_prct_low_LGT <= last_all_value and last_20_prct_upp_LGT >= last_all_value:
        twenty_prct_LGT = 1
    else:
        twenty_prct_LGT = 0

    ## 30 procentov
    if last_pred_value_LGT >= 0:
        last_30_prct_upp_LGT = last_pred_value_LGT * 1.3
        last_30_prct_low_LGT = last_pred_value_LGT * 0.7
    else:
        last_30_prct_upp_LGT = last_pred_value_LGT * 0.7
        last_30_prct_low_LGT = last_pred_value_LGT * 1.3
    if last_30_prct_low_LGT <= last_all_value and last_30_prct_upp_LGT >= last_all_value:
        thirty_prct_LGT = 1
    else:
        thirty_prct_LGT = 0

    ## 50 procentov
    if last_pred_value_LGT >= 0:
        last_50_prct_upp_LGT = last_pred_value_LGT * 1.5
        last_50_prct_low_LGT = last_pred_value_LGT * 0.5
    else:
        last_50_prct_upp_LGT = last_pred_value_LGT * 0.5
        last_50_prct_low_LGT = last_pred_value_LGT * 1.5
    if last_50_prct_low_LGT <= last_all_value and last_50_prct_upp_LGT >= last_all_value:
        fifty_prct_LGT = 1
    else:
        fifty_prct_LGT = 0

    ## 100 procentov
    if last_pred_value_LGT >= 0:
        last_100_prct_upp_LGT = last_pred_value_LGT * 2
        last_100_prct_low_LGT = last_pred_value_LGT * 0
    else:
        last_100_prct_upp_LGT = last_pred_value_LGT * 0
        last_100_prct_low_LGT = last_pred_value_LGT * 2
    if last_100_prct_low_LGT <= last_all_value and last_100_prct_upp_LGT >= last_all_value:
        hundred_prct_LGT = 1
    else:
        hundred_prct_LGT = 0


    Eror_DLT = last_all_value - last_pred_value_DLT
    Eror_ETS = last_all_value - last_pred_value_ETS
    Eror_LGT = last_all_value - last_pred_value_DLT

    sez_DLT.append((Eror_DLT, ten_prct_DLT, twenty_prct_DLT, thirty_prct_DLT, fifty_prct_DLT, hundred_prct_DLT))
    sez_ETS.append((Eror_ETS, ten_prct_ETS, twenty_prct_ETS, thirty_prct_ETS, fifty_prct_ETS, hundred_prct_ETS))
    sez_LGT.append((Eror_LGT, ten_prct_LGT, twenty_prct_LGT, thirty_prct_LGT, fifty_prct_LGT, hundred_prct_LGT))
    return sez_DLT, sez_ETS, sez_LGT


# Gets data for one day 
    
def Orbit_prediction_one_day(date="2022-02-10", X=30, y=2):
    ##########DLT
    napaka_DLT = 0
    stevilo_nanov_DLT = 0
    ten_prct_DLT = 0
    twenty_prct_DLT = 0
    thirty_prct_DLT = 0
    fifty_prct_DLT = 0
    hundred_prct_DLT = 0

    ##########ETS
    napaka_ETS = 0
    stevilo_nanov_ETS = 0
    ten_prct_ETS = 0
    twenty_prct_ETS = 0
    thirty_prct_ETS = 0
    fifty_prct_ETS = 0
    hundred_prct_ETS = 0

    ##########LGT
    napaka_LGT = 0
    stevilo_nanov_LGT = 0
    ten_prct_LGT = 0
    twenty_prct_LGT = 0
    thirty_prct_LGT = 0
    fifty_prct_LGT = 0
    hundred_prct_LGT = 0

    for i in range(24):
        if i == 0:
            string = "{} 0{}:00:00".format(date, i)
            sez_DLT, sez_ETS, sez_LGT = Orbit_model(string, X=30, y=2, sez_DLT = [], sez_ETS = [], sez_LGT = [])
        else:
            string = "{} {}:00:00".format(date, i)
            sez_DLT, sez_ETS, sez_LGT = Orbit_model(string, X=30, y=2, sez_DLT = [], sez_ETS = [], sez_LGT = [])
    for i in range(len(sez_LGT)):
        # print("Tule je sez[i][0] = " ,sez[i][0])
        if sez_DLT[i][0] == "N":
            stevilo_nanov_DLT += 1
        else:
            napaka_DLT += abs(sez_DLT[i][0])
            ten_prct_DLT += sez_DLT[i][1]
            twenty_prct_DLT += sez_DLT[i][2]
            thirty_prct_DLT += sez_DLT[i][3]
            fifty_prct_DLT += sez_DLT[i][4]
            hundred_prct_DLT += sez_DLT[i][5]
        
        if sez_ETS[i][0] == "N":
            stevilo_nanov_ETS += 1
        else:
            napaka_ETS += abs(sez_ETS[i][0])
            ten_prct_ETS += sez_ETS[i][1]
            twenty_prct_ETS += sez_ETS[i][2]
            thirty_prct_ETS += sez_ETS[i][3]
            fifty_prct_ETS += sez_ETS[i][4]
            hundred_prct_ETS += sez_ETS[i][5]

        if sez_LGT[i][0] == "N":
            stevilo_nanov_LGT += 1
        else:
            napaka_LGT += abs(sez_LGT[i][0])
            ten_prct_LGT += sez_LGT[i][1]
            twenty_prct_LGT += sez_LGT[i][2]
            thirty_prct_LGT += sez_LGT[i][3]
            fifty_prct_LGT += sez_LGT[i][4]
            hundred_prct_LGT += sez_LGT[i][5]
        
    napaka_delez_DLT = (napaka_DLT / len(sez_DLT))
    napaka_delez_ETS = (napaka_ETS / len(sez_ETS))
    napaka_delez_LGT = (napaka_LGT / len(sez_LGT))

    napaka_procenti_DLT = (napaka_delez_DLT ,ten_prct_DLT, twenty_prct_DLT, thirty_prct_DLT, fifty_prct_DLT, hundred_prct_DLT)
    napaka_procenti_ETS = (napaka_delez_ETS ,ten_prct_ETS, twenty_prct_ETS, thirty_prct_ETS, fifty_prct_ETS, hundred_prct_ETS)
    napaka_procenti_LGT = (napaka_delez_LGT ,ten_prct_LGT, twenty_prct_LGT, thirty_prct_LGT, fifty_prct_LGT, hundred_prct_LGT)
    # print(sez,"v procentih kok jih zadane ta interval ", delez, "absolutna napaka napovedi za ta dan ", napaka_delez)
    return (napaka_procenti_DLT, napaka_procenti_ETS, napaka_procenti_LGT)


## Only for ETS model (which is fastest)
def plottt_only_one_model(date="2021-11-11", X=30, y=2):
    ##########ETS
    napaka_ETS = 0
    stevilo_ETS = 0
    stevilo_nanov_ETS = 0
    ten_prct_ETS = 0
    twenty_prct_ETS = 0
    thirty_prct_ETS = 0
    fifty_prct_ETS = 0
    hundred_prct_ETS = 0

    for i in range(24):
        if i == 0:
            string = "{} 0{}:00:00".format(date, i)
            sez_ETS = Orbit_model_ETS([], string, X, y)
        else:
            sez_ETS = sez_ETS
            string = "{} {}:00:00".format(date, i)
            sez_ETS = Orbit_model_ETS(sez_ETS, string, X, y)
    for i in range(len(sez_ETS)):
        # print("Tule je sez[i][0] = " ,sez[i][0])

        if sez_ETS[i][0] == "N":
            stevilo_nanov_ETS += 1
        else:
            napaka_ETS += abs(sez_ETS[i][0])
            ten_prct_ETS += sez_ETS[i][1]
            twenty_prct_ETS += sez_ETS[i][2]
            thirty_prct_ETS += sez_ETS[i][3]
            fifty_prct_ETS += sez_ETS[i][4]
            hundred_prct_ETS += sez_ETS[i][5]        
    
    napaka_procenti_ETS = (napaka_ETS ,ten_prct_ETS, twenty_prct_ETS, thirty_prct_ETS, fifty_prct_ETS, hundred_prct_ETS, stevilo_nanov_ETS)
    return napaka_procenti_ETS





# Orbit prediction for number of days in question
def Orbit_prediction_days(date="2021-11-11", number_of_days=2, X=30, y=2):
    dates = pd.date_range(start=date, periods=number_of_days)
    sez2 = []

    napaka_ETS = 0
    stevilo_ETS = 0
    stevilo_nanov_ETS = 0
    ten_prct_ETS = 0
    twenty_prct_ETS = 0
    thirty_prct_ETS = 0
    fifty_prct_ETS = 0
    hundred_prct_ETS = 0
    stevec_bad = 0
    stevec_pred_good = 0

    for i in range(number_of_days):
        i = str(dates[i])
        # print(i.split(" ")[0])
        procenti_stevila = plottt_only_one_model(i.split(" ")[0], X, y)
        
        napaka_ETS += procenti_stevila[0]
        ten_prct_ETS += procenti_stevila[1]
        twenty_prct_ETS += procenti_stevila[2]
        thirty_prct_ETS += procenti_stevila[3]
        fifty_prct_ETS += procenti_stevila[4]
        hundred_prct_ETS += procenti_stevila[5]

        stevilo_nanov_ETS += procenti_stevila[6]
        

    dolzina = 24 * number_of_days - stevilo_nanov_ETS
    napaka_abs_all = napaka_ETS / dolzina

    in_ten_prct = ten_prct_ETS / dolzina
    in_twenty_prct = twenty_prct_ETS / dolzina
    in_thirty_prct = thirty_prct_ETS / dolzina
    in_fifty_prct = fifty_prct_ETS / dolzina
    in_hundred_prct = hundred_prct_ETS / dolzina

    print("Napake za število dni", number_of_days, "\n",
          "\n Kolikšna je povprečna napaka ", napaka_abs_all, "postopoma v intervalih: \n",
          "10 procentni absolutni interval", in_ten_prct, " \n",
          "20 procentov", in_twenty_prct, " \n", "30 procentov", in_thirty_prct, " \n", "50 procentov", in_fifty_prct,
          " \n", "100 procentov", in_hundred_prct)

    return (napaka_abs_all, in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct)

 

def best_interval_X(date="2021-11-11", number_of_days=50, freq=7, start_time=3, end_time=9):
    ten_prct = []
    twenty_prct = []
    thirty_prct = []
    fifty_prct = []
    hundred_prct = []
    in_prophet = []
    napaka_abs_all_sez = []
    stevec_bad = 0
    stevec_pred_good = 0

    for i in np.linspace(start_time, end_time, freq):
        napaka_abs_all, in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct = Orbit_prediction_days(
                                                                                        date, number_of_days, X=i, y=2)
        f = open("src/data_analysis/Generated_data/Information_about_orbit_februar_50.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, X = {i}, in za koliko dni je narejena analiza {number_of_days} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)

    f = open("src/data_analysis/Generated_data/Information_about_orbit_februar_50.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()


def best_interval_Y(date="2021-11-10", number_of_days=50, freq=8, start_time=0.5, end_time=4, X= 30):
    ten_prct = []
    twenty_prct = []
    thirty_prct = []
    fifty_prct = []
    hundred_prct = []
    in_prophet = []
    napaka_abs_all_sez = []
    stevec_bad = 0
    stevec_pred_good = 0

    for y in np.linspace(start_time, end_time, freq):
        napaka_abs_all, in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct  = Orbit_prediction_days(date, number_of_days, X , y)
        f = open("src/data_analysis/Generated_data/Information_about_orbit_Y_50.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, Y = {y}, in za koliko dni je narejena analiza {number_of_days} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

        # in_prophet.append(delez)
        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)

    f = open("src/data_analysis/Generated_data/Information_about_orbit_Y_50.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, orbit, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()

sez_feb_x = [
     [40.559493255013926, 40.575588803434506,  40.26557984708852, 39.97952822127603, 40.04594602860183, 40.22904402166987, 40.123904402166987, 38.123904402166987],
     [0.35672020287404904, 0.35383319292333615, 0.35576114381833474, 0.3546218487394958, 0.3518052057094878, 0.35318791946308725, 0.35918791946308725, 0.38918791946308725],
     [0.5950972104818258, 0.5922493681550126, 0.5962994112699748, 0.5991596638655462, 0.5961376994122586, 0.5973154362416108, 0.579154362416108, 0.569154362416108],
     [0.738799661876585, 0.7430497051390059, 0.7401177460050462, 0.7394957983193278, 0.7413937867338372, 0.7441275167785235, 0.781275167785235,   0.771275167785235],
     [0.8698224852071006, 0.8685762426284751,  0.8696383515559294, 0.8638655462184874,  0.8664987405541562, 0.8699664429530202,0.8719664429530202, 0.8819664429530202],
     [0.959425190194421, 0.9612468407750632,  0.95878889823381, 0.957983193277311,  0.9580184718723762, 0.9588926174496645,  0.9608926174496645, 0.9708926174496645]
                ]

sez_nov_x = [
     [34.88863303836274,  34.97447367613267, 34.67068700407235,  34.322446215925915, 34.412373915572395,  34.82566297044708, 34.43633524187389, 37.4101977692744], 
     [0.4, 0.3902777777777778, 0.3958333333333333, 0.4013888888888889,  0.4041666666666667,  0.39166666666666666,  0.4027777777777778, 0.3958333333333333],
     [0.6541666666666667, 0.6527777777777778, 0.6555555555555556, 0.6569444444444444,  0.6555555555555556, 0.6583333333333333, 0.6597222222222222,  0.6516666666666666],
     [0.8055555555555556, 0.8041666666666667,  0.8041666666666667,  0.8041666666666667, 0.8041666666666667, 0.8027777777777778, 0.7866666666666666, 0.8052139080111],
     [0.9263888888888889, 0.9277777777777778, 0.9277777777777778, 0.9263888888888889, 0.9263888888888889, 0.9277777777777778, 0.9263888888888889, 0.9225],
     [0.9888888888888889, 0.9875, 0.9888888888888889, 0.9833333333333333, 0.9847222222222223, 0.9861111111111112, 0.9833333333333333, 0.9858333333333333]
            ]

def plot_februar(sez_feb_x, sez_nov_x):
        # Seznam: Absolutna napaka, Prophet, 10%, 20%, 30%, 50%, 100%
    x = [3, 4, 5, 6, 7, 8, 9, 30]
    Labels = ["Absolutna napaka", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]
    stevec = 1
    for i in sez_feb_x[1:]:
        plt.plot(x, i, label = f"{Labels[stevec]}")
        stevec += 1
    
    plt.suptitle(f'Orbit napoved za 50 dni (februar)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()

    stevec = 1
    for i in sez_nov_x[1:]:
        plt.plot(x, i, label = f"{Labels[stevec]}")
        stevec += 1
    plt.suptitle(f'Orbit napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()


    plt.plot(x, sez_nov_x[0], label = "Absolutna napaka v novembru")
    plt.plot(x, sez_feb_x[0], label = "Absolutna napaka v februarju")
    plt.suptitle(f"Orbit napoved za 50 dni")
    plt.ylabel("Absolutna napaka")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend()
    plt.show()


sez_nov_Y = [
     [27.95483503067043, 34.75485888672886, 36.402097668759986, 37.79444130750017, 38.658625561382934, 40.278884306495186, 41.05733068123821, 41.706025492411],
     [0.5108333333333334, 0.42333333333333334, 0.4116666666666667, 0.3975, 0.37083333333333335, 0.37583333333333335, 0.36666666666666664, 0.3675],
     [0.7616666666666667, 0.6808333333333333, 0.6591666666666667, 0.6575, 0.6433333333333333, 0.6266666666666667, 0.6266666666666667, 0.62],
     [0.8741666666666666, 0.8216666666666667, 0.8125, 0.7941666666666667, 0.79, 0.7816666666666666, 0.7775, 0.7783333333333333],
     [0.9558333333333333, 0.9316666666666666, 0.9291666666666667, 0.93, 0.9258333333333333, 0.9225, 0.9183333333333333, 0.9158333333333334],
     [0.9875, 0.9841666666666666, 0.985, 0.9875, 0.9891666666666666, 0.9858333333333333, 0.9891666666666666, 0.9875]
            ]

def plot_y(sez_nov_Y):
    x_nov = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    Labels = ["Absolutna napaka", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]

    stevec = 1
    for i in sez_nov_Y[1:]:
        plt.plot(x_nov, i, label = f"{Labels[stevec]}")
        stevec += 1

    plt.suptitle(f'Orbit napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur napovedi pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()



if __name__ == "__main__":
    # Orbit_model_DLT(time_1="2021-11-11 00:00:00", X=30, y=2)
    # Orbit_model_ETS(sez_ETS = [], time_1="2022-02-27 12:00:00", X=30, y=2)
    # Orbit_model_LGT(time_1="2021-11-11 00:00:00", X=30, y=2)
    # Orbit_model(date="2021-11-11 00:00:00", X=30, y=2, sez_DLT = [], sez_ETS = [], sez_LGT = [])
    # Orbit_prediction_one_day(date="2022-02-10", X=30, y=2) # takes really long
    # plottt_only_one_model(date="2021-11-11", X=30, y=2)
    # best_interval_X(date="2022-02-10", number_of_days=50, freq=1, start_time=9, end_time=9)
    # best_interval_Y(date="2021-11-10", number_of_days=50, freq=8, start_time=0.5, end_time=4, X= 30)

    # plot_februar(sez_feb_x, sez_nov_x)
    # plot_y(sez_nov_Y)


    pass