from cProfile import label
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from inspect import trace
# from msilib.schema import Error
from re import X
from turtle import color

import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np
from yaml import serialize

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import sys
sys.path.insert(0, "src/data_analysis")

from data_retrieval import get_intra_day_data_for_region
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

from prophet import Prophet
from Ignore_warnings import suppress_stdout_stderr


time_1 = "2022-02-27 12:00:00"
pd.set_option('display.expand_frame_repr', False)
df_intra_day = get_intra_day_data_for_region("GERMANY")

df_intra_day = df_intra_day.sort_values(by=['trd_execution_time',
                                            'trd_buy_delivery_area',
                                            'trd_sell_delivery_area'])


def Prophet_hour_block(time_1, sez=[], X=5, y=2, stevec=0, stevec_pred=0):
    all_data, train_data, last_data = One_block_X_h(time_1, X, y)
    train_data = train_data[["trd_execution_time", "trd_price"]]
    train_data.rename(columns={'trd_execution_time': 'ds', 'trd_price': 'y'}, inplace=True)
    all_data.rename(columns={'trd_execution_time': 'ds', 'trd_price': 'y'}, inplace=True)
    
    if time_1 == "2022-02-15 01:00:00":
        sez.append("N")
        stevec += 1
        return sez, stevec, stevec_pred

    # If data is not good enough, dont include it in the statistics
    try:
        m = Prophet()
        with suppress_stdout_stderr():
            m.fit(train_data)
            stevec_pred += 1
    except ValueError:
        sez.append("N")
        stevec += 1
        return sez, stevec, stevec_pred
    pred_time = int(y * 60)
    future = m.make_future_dataframe(periods=pred_time, freq="1min")
    forecast = m.predict(future)


    last_date = forecast.ds.iloc[-1]
    last_real_data_date = all_data["ds"].iloc[-1]

    # If data is not good enough, dont include it in the statistics
    if last_real_data_date - last_date >= pd.Timedelta(1, unit='H'):
        sez.append("N")
        stevec += 1
        return sez, stevec, stevec_pred
    
    last_pred = forecast.yhat.values[-1]
    last_upper = forecast.yhat_upper.values[-1]
    last_lower = forecast.yhat_lower.values[-1]
    last_all = all_data["y"].values[-1]

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

    ### In the interval of the prophet
    if last_upper >= last_all and last_lower <= last_all:
        in_interval = 1
    else:
        in_interval = 0

    # print(last_all, last_pred)
    Eror = last_pred - last_all
    sez.append((Eror, in_interval, ten_prct, twenty_prct, thirty_prct, fifty_prct, hundred_prct))

    ### ERROR but have to fix the lenghts
    # rmse=sqrt(mean_squared_error(pred, dataa))
    # print(rmse)

    ############# PLOT
    # m.plot(forecast)
    # plt.show()

    # ax = forecast.plot(x="ds", y="yhat", color="red", label="Napoved")
    # all_data.plot(x="ds", y="y", color="blue", label="Realni podatki", ax=ax)
    # forecast.plot(x="ds", y="yhat_lower", color="grey", label="Interval zaupanja", ax=ax)
    # forecast.plot(x="ds", y="yhat_upper", color="grey", label='_nolegend_', ax=ax)
    # plt.fill_between(forecast.ds, forecast["yhat_lower"], forecast["yhat_upper"] , interpolate=True, color='grey', alpha=0.5)
    # plt.suptitle(f'Prophet napoved za {time_1}')
    # plt.ylabel("Vrednosti")
    # plt.xlabel("Čas")
    # plt.legend()
    # plt.show()

    return sez, stevec, stevec_pred

# plots one day and calculates the margins
def Prohpet_prediction_one_day(date="2022-02-10", X=5, y=2):
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
            sez, stevec, stevec_pred = Prophet_hour_block(string, [], X, y, stevec=0, stevec_pred=0)
        else:
            string = "{} {}:00:00".format(date, i)
            sez, stevec, stevec_pred = Prophet_hour_block(string, sez, X, y, stevec, stevec_pred)
    for i in range(len(sez)):
        # print("Tule je sez[i][0] = " ,sez[i][0])
        if sez[i][0] == "N":
            stevilo_nanov += 1
        else:
            napaka += abs(sez[i][0])
            stevilo2 += sez[i][1]
            ten_prct += sez[i][2]
            twenty_prct += sez[i][3]
            thirty_prct += sez[i][4]
            fifty_prct += sez[i][5]
            hundred_prct += sez[i][6]

    procenti_stevila = (ten_prct, twenty_prct, thirty_prct, fifty_prct, hundred_prct)
    delez = stevilo2 / (len(sez) - stevilo_nanov) * 100
    napaka_delez = napaka / (len(sez) - stevilo_nanov)
    # print(sez,"v procentih kok jih zadane ta interval ", delez, "absolutna napaka napovedi za ta dan ", napaka_delez)
    return sez, stevilo2, napaka, stevilo_nanov, procenti_stevila, stevec, stevec_pred

# Makes prediction for the number of days in question
def Prohpet_prediction_days(date="2022-02-10", number_of_days=5, X=5, y=2):
    dates = pd.date_range(start=date, periods=number_of_days)
    sez2 = []
    stevilo2_2 = 0
    napaka_2 = 0
    dolzina = 0

    ten_prct = 0
    twenty_prct = 0
    thirty_prct = 0
    fifty_prct = 0
    hundred_prct = 0

    stevec_bad = 0
    stevec_pred_good = 0

    for i in range(number_of_days):
        i = str(dates[i])
        # print(i.split(" ")[0])
        sez, stevilo2, napaka, stevilo_nanov, procenti_stevila, stevec, stevec_pred = Prohpet_prediction_one_day(i.split(" ")[0], X, y)
        dolzina += 24 - stevilo_nanov
        stevilo2_2 += stevilo2
        napaka_2 += napaka

        ten_prct += procenti_stevila[0]
        twenty_prct += procenti_stevila[1]
        thirty_prct += procenti_stevila[2]
        fifty_prct += procenti_stevila[3]
        hundred_prct += procenti_stevila[4]

        stevec_bad += stevec
        stevec_pred_good += stevec_pred

        sez2.append(sez)
    delez = stevilo2_2 / dolzina
    napaka_abs_all = napaka_2 / dolzina

    in_ten_prct = ten_prct / dolzina
    in_twenty_prct = twenty_prct / dolzina
    in_thirty_prct = thirty_prct / dolzina
    in_fifty_prct = fifty_prct / dolzina
    in_hundred_prct = hundred_prct / dolzina

    print("Napake za število dni", number_of_days, "\n", "kolikokrat zadanemo interval Propheta ", delez,
          "\n Kolikšna je povprečna napaka ", napaka_abs_all, "postopoma v intervalih: \n",
          "10 procentni absolutni interval", in_ten_prct, " \n",
          "20 procentov", in_twenty_prct, " \n", "30 procentov", in_thirty_prct, " \n", "50 procentov", in_fifty_prct,
          " \n", "100 procentov", in_hundred_prct)

    return (in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, delez, napaka_abs_all, stevec_bad, stevec_pred_good)

# Checks for the best X
def best_interval_X(date="2022-02-10", number_of_days=5, freq=9, start_time=4, end_time=6):
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
        in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, delez, napaka_abs_all, stevec, stevec_pred = Prohpet_prediction_days(
            date, number_of_days, i, y=2)
        f = open("src/data_analysis/Generated_data/Information_about_prohpet_november.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, X = {i}, in za koliko dni je narejena analiza {number_of_days} \n kolikokrat zadanemo interval Propheta {delez} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

        in_prophet.append(delez)
        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)



    f = open("src/data_analysis/Generated_data/Information_about_prohpet_november.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, fb prophet, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{in_prophet},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()
    

# Checks for best Y
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
        in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, delez, napaka_abs_all, stevec, stevec_pred = Prohpet_prediction_days(date, number_of_days, X , y)
        f = open("src/data_analysis/Generated_data/Information_about_prohpet_Y_50.txt", "a+")
        f.write(
            f"\n \n Začetni datum {date}, Y = {y}, in za koliko dni je narejena analiza {number_of_days} \n kolikokrat zadanemo interval Propheta {delez} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()

        in_prophet.append(delez)
        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)

        stevec_bad += stevec
        stevec_pred_good += stevec_pred

        

    print( "stevec bad = ", stevec_bad, "  Stevec prediction good ", stevec_pred_good)

    f = open("src/data_analysis/Generated_data/Information_about_prohpet_Y_50.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, fb prophet, 10%, 20%, 30%, 50%, 100%")
    f.write(
        f"\n Seznam = [{napaka_abs_all_sez},{in_prophet},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()



if __name__ == "__main__":
    Prophet_hour_block(time_1, sez=[], X=5, y=2, stevec=0, stevec_pred=0)
    # Prohpet_prediction_one_day(date="2022-02-10", X=5, y=2)
    # Prohpet_prediction_days(date="2022-02-10", number_of_days=5, X=5, y=2)
    # best_interval_X(date="2022-02-10", number_of_days=5, freq=9, start_time=4, end_time=6)
    # best_interval_Y(date="2021-11-10", number_of_days=50, freq=8, start_time=0.5, end_time=4, X= 30)

    pass