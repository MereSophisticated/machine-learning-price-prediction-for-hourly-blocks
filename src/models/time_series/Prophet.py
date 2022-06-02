from cProfile import label
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from inspect import trace
# from msilib.schema import Error
from re import X
from tkinter.messagebox import NO
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


sez_februar_x =[
        [76.286712749165, 71.45194355224702, 67.14760124962491, 64.80029793116448, 62.62776705674282, 61.71739551922807, 60.67275988785725,59.79778966648058, 59.066521355225184, 58.2753958597662, 57.72380408956688, 57.23304662438029, 56.562229400898346, 50.866864732762274],
        [0.8594276094276094, 0.8168067226890756,  0.7605042016806722, 0.7237615449202351, 0.6767422334172963, 0.6208053691275168, 0.572986577181208, 0.5385906040268457, 0.4979044425817267,  0.4752724224643755, 0.42797319932998323, 0.4045226130653266, 0.3777219430485762, 0.16847372810675562], 
        [0.1978114478114478, 0.20252100840336135,  0.21596638655462186, 0.2174643157010915, 0.22670025188916876, 0.22902684563758388, 0.2332214765100671,0.23825503355704697, 0.24643755238893544, 0.25230511316010057, 0.2562814070351759, 0.25711892797319935, 0.25963149078726966, 0.2727272727272727], 
        [0.3653198653198653, 0.3865546218487395, 0.4,  0.4139378673383711, 0.4139378673383711, 0.41191275167785235, 0.41946308724832215, 0.4312080536912752, 0.43671416596814755, 0.44341994970662196, 0.45226130653266333, 0.4581239530988275, 0.4564489112227806, 0.49958298582151794], 
        [0.5084175084175084, 0.5243697478991597,  0.5327731092436975, 0.5541561712846348, 0.5709487825356843, 0.5738255033557047, 0.5788590604026845, 0.5855704697986577, 0.5926236378876781, 0.5968147527242247, 0.592964824120603, 0.5971524288107203, 0.5988274706867671, 0.6355296080066722], 
        [0.6742424242424242, 0.6941176470588235, 0.7151260504201681, 0.7237615449202351, 0.7355163727959698, 0.7323825503355704, 0.7332214765100671, 0.735738255033557, 0.7426655490360435, 0.7460184409052808, 0.7470686767169179,0.7495812395309883, 0.7512562814070352, 0.7856547122602169], 
        [0.8333333333333334, 0.838655462184874, 0.8571428571428571, 0.8656591099916037, 0.874895046179681, 0.8791946308724832, 0.8842281879194631, 0.886744966442953, 0.8851634534786254, 0.8885163453478625, 0.8919597989949749, 0.890284757118928, 0.8927973199329984, 0.9090909090909091]
                ]

sez_november_x = [
        [72.18046359787297, 63.21393956950056, 58.87468022096166, 56.24590853406177, 54.66479264673738, 53.80000860364345, 53.15215232655278, 48.032061612431065],
        [0.9148580968280468, 0.8607172643869891, 0.78, 0.7025, 0.6, 0.5141666666666667, 0.4525, 0.19416666666666665],
        [0.2287145242070117, 0.24437030859049208, 0.26416666666666666, 0.2733333333333333, 0.2866666666666667, 0.28583333333333333, 0.2841666666666667,0.30333333333333334],
        [0.4073455759599332, 0.45454545454545453, 0.4766666666666667, 0.4866666666666667, 0.495, 0.5091666666666667, 0.5125, 0.5541666666666667],
        [0.5692821368948247, 0.5938281901584654, 0.635, 0.6533333333333333, 0.6591666666666667, 0.6641666666666667, 0.6716666666666666, 0.7141666666666666],
        [0.7495826377295493, 0.8006672226855713, 0.825, 0.8325, 0.8333333333333334, 0.8466666666666667, 0.8541666666666666, 0.865],
        [0.8914858096828047, 0.9257714762301918, 0.9308333333333333, 0.9358333333333333, 0.9383333333333334, 0.9475, 0.9508333333333333, 0.9583333333333334]
                ]
 




def plot_februar(sez_februar_x, sez_november_x):
    sez_feb = sez_februar_x
    sez_nov = sez_november_x
        # Seznam: Absolutna napaka, Prophet, 10%, 20%, 30%, 50%, 100%
    x_feb = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 30]
    x_nov = [3, 4, 5, 6, 7, 8, 9, 30]
    Labels = ["Absolutna napaka", "Prophetov interval zaupanja", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]
    stevec = 1
    for i in sez_feb[1:]:
        plt.plot(x_feb, i, label = f"{Labels[stevec]}")
        stevec += 1
    
    plt.suptitle(f'Prophet napoved za 50 dni (februar)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()
    stevec = 1
    for i in sez_nov[1:]:
        plt.plot(x_nov, i, label = f"{Labels[stevec]}")
        stevec += 1
    plt.suptitle(f'Prophet napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()


    plt.plot(x_feb, sez_feb[0], label = "Absolutna napaka v februarju")
    plt.plot(x_nov, sez_nov[0], label = "Absolutna napaka v novembru")
    plt.suptitle(f"Prophet napoved za 50 dni")
    plt.ylabel("Absolutna napaka")
    plt.xlabel("Število ur podatkov pred zaprtjem")
    plt.legend()
    plt.show()

sez_nov_Y = [
    [43.284090335359075, 40.07719783131048, 49.437600118176874, 49.88199082553852, 57.72689287120632, 52.68041596310269, 60.130628336367096, 57.039846807092715],
    [0.1325, 0.15833333333333333, 0.21083333333333334, 0.24416666666666667, 0.23916666666666667, 0.2825, 0.2775, 0.31666666666666665],
    [0.38416666666666666, 0.37416666666666665, 0.33, 0.2925, 0.26666666666666666, 0.25416666666666665, 0.2633333333333333, 0.275],
    [0.6158333333333333, 0.6191666666666666, 0.5475, 0.55, 0.49, 0.5141666666666667, 0.4691666666666667, 0.49583333333333335],
    [0.7525, 0.7575, 0.6991666666666667, 0.6991666666666667, 0.6516666666666666, 0.6741666666666667, 0.6333333333333333, 0.6625],
    [0.8858333333333334, 0.9008333333333334, 0.8566666666666667, 0.8616666666666667, 0.8075, 0.8391666666666666, 0.815, 0.8316666666666667],
    [0.9508333333333333, 0.9733333333333334, 0.9533333333333334, 0.95, 0.9358333333333333, 0.9416666666666667, 0.9275, 0.9366666666666666]
        ]

def plot_y(sez_nov_Y):
    x_nov = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    Labels = ["Absolutna napaka", "Prophetov interval zaupanja", "10% absolutni interval", "20% absolutni interval", "30% absolutni interval", "50% absolutni interval", "100% absolutni interval"]

    stevec = 1
    for i in sez_nov_Y[1:]:
        plt.plot(x_nov, i, label = f"{Labels[stevec]}")
        stevec += 1

    plt.suptitle(f'Prophet napoved za 50 dni (november)')
    plt.ylabel("Kolikokrat zadanemo interval (delež)")
    plt.xlabel("Število ur napovedi pred zaprtjem")
    plt.legend(loc="upper right")
    plt.show()




if __name__ == "__main__":
    # Prophet_hour_block(time_1, sez=[], X=5, y=2, stevec=0, stevec_pred=0)
    # Prohpet_prediction_one_day(date="2022-02-10", X=5, y=2)
    # Prohpet_prediction_days(date="2022-02-10", number_of_days=5, X=5, y=2)
    # best_interval_X(date="2022-02-10", number_of_days=5, freq=9, start_time=4, end_time=6)
    # best_interval_Y(date="2021-11-10", number_of_days=50, freq=8, start_time=0.5, end_time=4, X= 30)
    
    # plot_februar(sez_februar_x, sez_november_x)
    # plot_y(sez_nov_Y)
    pass