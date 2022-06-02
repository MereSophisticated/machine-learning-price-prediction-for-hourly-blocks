from cProfile import label
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from inspect import trace
# from msilib.schema import Error
from re import X
from tracemalloc import start
from turtle import color

import pandas as pd
from pyparsing import col

pd.options.mode.chained_assignment = None

import numpy as np
# from yaml import serializew

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import sys
sys.path.insert(0, "src/data_analysis")

from data_retrieval import get_intra_day_data_for_region
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pmdarima.arima import ADFTest
from datetime import datetime, timedelta

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from pmdarima import auto_arima, AutoARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from spectrum import Periodogram, data_cosine

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import signal
import os
import scipy, pylab

from Ignore_warnings import suppress_stdout_stderr

# All dates
# start_date = "2021-11-08"
# end_date = "2022-05-01"

#time_1 is a random specific date
time_1 = "2021-11-23 12:00:00"
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

    
    plt.plot(prediction, label="Napoved")
    plt.plot(result[-24:], label="Realni podatki")
    plt.legend()
    plt.show()

    plt.plot(result, label="Realni podatki")
    plt.plot(novi, label="Napoved")
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

# Hourly Blocks prediction

def arima_model(time_1, sez=[], X=5, y=2):
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
    # print(stepwise_fit)

    stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
    order = (int(stepwise_fit[0]), stevec, int(stepwise_fit[2][0]))
    
    # # mod = ARIMA(train_data["diff"], order=order)
    # with suppress_stdout_stderr:
    #     res = mod.fit()

    # all_data["fit"] = res.fittedvalues

    # end = len(all_data)
    # start = len(train_data)
    # pred = res.predict(start=start, end=end, typ='levels', color="red")
    
    # nans = (len(train_data)) * [np.nan]
    # valuess = pred.values
    # predictionn = np.concatenate((nans, valuess))
    # all_data["prediction"] = pd.Series(predictionn)
    
    ### Plot

    # ax = all_data.plot(x="trd_execution_time", y="prediction", color="red", label="Prediciton")
    # all_data.plot(x="trd_execution_time", y="trd_price", color="blue", label="data", ax=ax)
    # plt.legend()
    # plt.show()


    on = "trd_execution_time"
    df = all_data
    interval = "30S"
    number_of_predictions = int(y * 120) # da je y urna napoved
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))

    df.reset_index(inplace=True)
    try:
        test_size = number_of_predictions
        train_df = df[:-test_size]
        test_df = df[-test_size:]
    except TypeError:
        sez.append("N")
        print("N")
        return sez
    # stepwise_fit = auto_arima(train_df["trd_price_mean"], trace=False, suppress_warnings=True, seasonal=False)

    # print(stepwise_fit)
    # stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
    # order = (int(stepwise_fit[0]), stevec, int(stepwise_fit[2][0]))
    # print(order)
    print(time_1)
    if time_1 == "2021-11-11 7:00:00" or "2021-11-15 9:00:00":
        sez.append("N")
        return sez
    mod1 = ARIMA(train_df["trd_price_mean"], order=order)
    try:
        res1 = mod1.fit()
    except ValueError or np.linalg.LinAlgError:
        sez.append("N")
        return sez


    end = len(train_df) + len(test_df)
    start = len(train_df)
    # pred1 = res.predict(start=start, end=end, typ='levels', color="red")
    pred2 = res1.predict(start=start, end=end, typ='levels', color="red")

    nans = (len(train_df)) * [np.nan]
    # valuess1 = pred1.values
    valuess2 = pred2.values
    # predictionn1 = np.concatenate((nans, valuess1))
    predictionn2 = np.concatenate((nans, valuess2))
    # df["prediction1"] = pd.Series(predictionn1)
    df["prediction2"] = pd.Series(predictionn2)

    #PLOTS
    
    # plt.plot(df["trd_execution_time"], df["prediction2"], color= "red", label= "Napoved na spremenjenih podatkih")
    # plt.plot(all_data["trd_execution_time"], all_data["trd_price"], color= "blue", label= "Realni podatki")
    # plt.suptitle("ARIMA napoved za blok 11.11.2021 ob 12:00")
    # plt.xlabel("Čas")
    # plt.ylabel("Cena")
    # plt.legend()
    # plt.show()


    # ax1 = df.plot(x="trd_execution_time", y="prediction2", color="red", label="Prediciton on resampled data")
    # df.plot(x="trd_execution_time", y="prediction1", color="orange", label="Prediciton on data", ax=ax1)
    # df.plot(x="trd_execution_time", y="trd_price_mean", color="blue", label="data", ax=ax1)

    # df.plot(x="trd_execution_time", y="trd_price_mean", color="blue", label="data", ax=ax1)

    # plt.legend()
    # plt.show()

    
    # ax = pylab.subplot(111)
    # # ax.plot(df["trd_execution_time"], df["prediction2"], color= "red", label="Prediciton on resampled data")
    # ax.plot(df["trd_execution_time"], df["prediction1"], color= "orange", label="Napovedani podatki")
    # ax.plot(all_data["trd_execution_time"], all_data["trd_price"], label = "Resnični podatki")
    # ax.figure.legend(loc='lower left')
    # ax.figure.show()
    last_pred = pred2.values[-1]
    last_all = all_data["trd_price"].values[-1]

        ### 10 procentov
    if last_pred >= 0:
        last_10_prct_upp = last_pred * 1.1
        last_10_prct_low = last_pred * 0.9
    else:
        last_10_prct_upp = last_pred * 0.9
        last_10_prct_low = last_pred * 1.1
    if last_10_prct_low <= last_all and last_10_prct_upp >= last_all:
        ten_prct = 1
    else:
        ten_prct = 0
    ## 20 procentov
    if last_pred >= 0:
        last_20_prct_upp = last_pred * 1.2
        last_20_prct_low = last_pred * 0.8
    else:
        last_20_prct_upp = last_pred * 0.8
        last_20_prct_low = last_pred * 1.2
    if last_20_prct_low <= last_all and last_20_prct_upp >= last_all:
        twenty_prct = 1
    else:
        twenty_prct = 0

    ## 30 procentov
    if last_pred >= 0:
        last_30_prct_upp = last_pred * 1.3
        last_30_prct_low = last_pred * 0.7
    else:
        last_30_prct_upp = last_pred * 0.7
        last_30_prct_low = last_pred * 1.3
    if last_30_prct_low <= last_all and last_30_prct_upp >= last_all:
        thirty_prct = 1
    else:
        thirty_prct = 0

    ## 50 procentov
    if last_pred >= 0:
        last_50_prct_upp = last_pred * 1.5
        last_50_prct_low = last_pred * 0.5
    else:
        last_50_prct_upp = last_pred * 0.5
        last_50_prct_low = last_pred * 1.5
    if last_50_prct_low <= last_all and last_50_prct_upp >= last_all:
        fifty_prct = 1
    else:
        fifty_prct = 0

    ## 100 procentov
    if last_pred >= 0:
        last_100_prct_upp = last_pred * 2
        last_100_prct_low = last_pred * 0
    else:
        last_100_prct_upp = last_pred * 0
        last_100_prct_low = last_pred * 2
    if last_100_prct_low <= last_all and last_100_prct_upp >= last_all:
        hundred_prct = 1
    else:
        hundred_prct = 0


    # print(last_all, last_pred)
    Eror = last_pred - last_all
    sez.append((Eror, ten_prct, twenty_prct, thirty_prct, fifty_prct, hundred_prct))


    return sez



# plots one day and calculates the margins
def Arima_prediction_one_day(date="2021-11-10", X=5, y=2):
    napaka = 0
    stevilo2 = 0
    stevilo_nanov = 0
    ten_prct = 0
    twenty_prct = 0
    thirty_prct = 0
    fifty_prct = 0
    hundred_prct = 0

    for i in range(24):
        if i == 0:
            string = "{} 0{}:00:00".format(date, i)
            sez = arima_model(string, [], X, y)
        else:
            string = "{} {}:00:00".format(date, i)
            sez = arima_model(string, sez, X, y)
    for i in range(len(sez)):
        if sez[i][0] == "N":
            stevilo_nanov += 1
        else:
            napaka += abs(sez[i][0])
            ten_prct += sez[i][1]
            twenty_prct += sez[i][2]
            thirty_prct += sez[i][3]
            fifty_prct += sez[i][4]
            hundred_prct += sez[i][5]

    procenti_stevila = (ten_prct, twenty_prct, thirty_prct, fifty_prct, hundred_prct)
    # print(sez,"v procentih kok jih zadane ta interval ", delez, "absolutna napaka napovedi za ta dan ", napaka_delez)
    return sez, napaka, stevilo_nanov, procenti_stevila



# Makes prediction for the number of days in question
def Arima_prediction_days(date="2021-11-10", number_of_days=5, X=5, y=2):
    dates = pd.date_range(start=date, periods=number_of_days)
    sez2 = []
    napaka_2 = 0
    dolzina = 0

    ten_prct = 0
    twenty_prct = 0
    thirty_prct = 0
    fifty_prct = 0
    hundred_prct = 0

    for i in range(number_of_days):
        i = str(dates[i])
        # print(i.split(" ")[0])
        sez, napaka, stevilo_nanov, procenti_stevila = Arima_prediction_one_day(i.split(" ")[0], X, y)
        dolzina += 24 - stevilo_nanov
    
        napaka_2 += napaka

        ten_prct += procenti_stevila[0]
        twenty_prct += procenti_stevila[1]
        thirty_prct += procenti_stevila[2]
        fifty_prct += procenti_stevila[3]
        hundred_prct += procenti_stevila[4]

        sez2.append(sez)
    
    napaka_abs_all = napaka_2 / dolzina

    in_ten_prct = ten_prct / dolzina
    in_twenty_prct = twenty_prct / dolzina
    in_thirty_prct = thirty_prct / dolzina
    in_fifty_prct = fifty_prct / dolzina
    in_hundred_prct = hundred_prct / dolzina

    print("Napake za število dni", number_of_days, "\n",
          "\n Kolikšna je povprečna napaka ", napaka_abs_all, "postopoma v intervalih: \n",
          "10 procentni absolutni interval", in_ten_prct, " \n",
          "20 procentov", in_twenty_prct, " \n", "30 procentov", in_thirty_prct, " \n", "50 procentov", in_fifty_prct,
          " \n", "100 procentov", in_hundred_prct)

    return (in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, napaka_abs_all)


# Checks for the best X
def best_interval_X(date="2021-11-10", number_of_days=20, freq=7, start_time=3, end_time=9):
    ten_prct = []
    twenty_prct = []
    thirty_prct = []
    fifty_prct = []
    hundred_prct = []
    napaka_abs_all_sez = []
    

    for i in np.linspace(start_time, end_time, freq):
        in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, napaka_abs_all = Arima_prediction_days(
            date, number_of_days, i, y=2)
        f = open("src/data_analysis/Generated_data/Information_about_Arima_november.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, X = {i}, in za koliko dni je narejena analiza {number_of_days} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)



    f = open("src/data_analysis/Generated_data/Information_about_Arima_november.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()



# Checks for best Y
def best_interval_Y(date="2021-11-10", number_of_days=20, freq=8, start_time=0.5, end_time=4, X= 30):
    ten_prct = []
    twenty_prct = []
    thirty_prct = []
    fifty_prct = []
    hundred_prct = []
    napaka_abs_all_sez = []
    

    for y in np.linspace(start_time, end_time, freq):
        in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, napaka_abs_all = Arima_prediction_days(date, number_of_days, X , y)
        f = open("src/data_analysis/Generated_data/Information_about_Arima_Y_50.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, Y = {y}, in za koliko dni je narejena analiza {number_of_days} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

    
        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)
        

    f = open("src/data_analysis/Generated_data/Information_about_Arima_Y_50.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()


sez_november_x =  [
     [34.25841656112742, 34.832085982795014, 37.90252218242501, 35.24702901463004, 35.24319805758758, 35.39637202876754, 35.34861528193223, 38.394535663813805],
     [0.4, 0.40208333333333335, 0.38125, 0.38333333333333336, 0.3770833333333333, 0.3770833333333333, 0.3770833333333333, 0.3862212943632568],
     [0.6604166666666667, 0.6583333333333333, 0.65625, 0.6541666666666667, 0.6541666666666667, 0.65625, 0.6541666666666667, 0.6555323590814196],
     [0.83125, 0.825, 0.8208333333333333, 0.8208333333333333, 0.8208333333333333, 0.8229166666666666, 0.8229166666666666, 0.8329853862212944],
     [0.9541666666666667, 0.95, 0.9416666666666667, 0.9458333333333333, 0.94375, 0.9416666666666667, 0.94375, 0.9394572025052192],
     [0.9958333333333333, 0.9958333333333333, 0.9958333333333333, 0.9958333333333333, 0.9958333333333333, 0.9958333333333333, 0.9958333333333333, 0.9937369519832986]
    ]






def plot_november(sez_november_x):
    sez_nov = sez_november_x
        # Seznam: Absolutna napaka, Prophet, 10%, 20%, 30%, 50%, 100%
    x_nov = [3, 4, 5, 6, 7, 8, 9, 30]
    Labels = ["Absolutna napaka", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]
    stevec = 1
    for i in sez_nov[1:]:
        plt.plot(x_nov, i, label = f"{Labels[stevec]}")
        stevec += 1
    plt.suptitle(f'Arima napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(x_nov, sez_nov[0], label = "Absolutna napaka v novembru")
    plt.suptitle(f"Arima napoved za 50 dni")
    plt.ylabel("Absolutna napaka")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend()
    plt.show()






def plot_y(sez_nov_Y):

    x_nov = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    Labels = ["Absolutna napaka", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]

    stevec = 1
    for i in sez_nov_Y[1:]:
        plt.plot(x_nov, i, label = f"{Labels[stevec]}")
        stevec += 1

    plt.suptitle(f'Arima napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur napovedi pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()













################################################################################################################################################################################
################################################################################################################################################################################

## Export data for R

def data_pandas_all(start_date='2021-11-08', end_date='2022-05-01'):
    drugi = get_intra_day(start_date, end_date)[["trd_price", "trd_delivery_time_start"]]
    drugi.to_csv("All_data.csv")


################################################################################################################################################################################
################################################################################################################################################################################


if __name__ == "__main__":
    # One_block_X_h(time_1, X=5, y=2)
    # get_intra_day(df_intra_day, start_date='2022-02-02', end_date='2022-03-24')
    # difference(start_date='2022-02-02', end_date='2022-03-24')
    # difference_predicition(start_date='2022-02-02', end_date='2022-03-24')
    # arima_model('2022-02-27 12:00:00' , X=5, y=2)

    # Arima_prediction_one_day(date="2021-11-10", X=5, y=2)
    # Arima_prediction_days(date="2021-11-10", number_of_days=5, X=5, y=2)
    # plot_november(sez_februar_x, sez_november_x)

    # best_interval_X(date="2021-11-10", number_of_days=20, freq=1, start_time=30, end_time=30)
    # best_interval_X(date="2022-02-10", number_of_days=20, freq=7, start_time=3, end_time=9)
    # best_interval_Y(date="2021-11-10", number_of_days=20, freq=6, start_time=1.5, end_time=4, X= 30)
    pass