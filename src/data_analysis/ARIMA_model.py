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

from data_retrieval import get_intra_day_data_for_region, get_day_ahead_data, get_sun_data, get_wind_data, \
    get_wind_forecast, get_data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pmdarima.arima import ADFTest
from datetime import datetime

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima, AutoARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from spectrum import Periodogram, data_cosine


from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import signal
import os
import sys


from prophet import Prophet



class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)



# import warnings
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)




#Partial data
time_1 = "2022-02-03 22:00:00"
pd.set_option('display.expand_frame_repr', False)
df_intra_day_germany = get_intra_day_data_for_region("GERMANY")


##All data

df_intra_day_germany_ALL = get_data("GERMANY")

df_intra_day_germany_ALL.drop(['trd_deleted', 'trd_quantity', 'trd_lot'], axis=1, inplace=True)
df_intra_day_germany_ALL = df_intra_day_germany_ALL.drop_duplicates(
    subset=['trd_execution_time', 'trd_buy_delivery_area', 'trd_sell_delivery_area'], keep='first').reset_index(drop=True)

df_intra_day_germany_ALL = df_intra_day_germany_ALL.sort_values(by=['trd_execution_time', 'trd_buy_delivery_area', 'trd_sell_delivery_area'])






# str(df_intra_day_germany_ALL["trd_delivery_time_end"][0]).split(" ")[0]

# df_intra_day_germany_ALL["trd_delivery_time_end"].values

def One_block_X_h(time_1, X=5):
    df_single_block = df_intra_day_germany_ALL[df_intra_day_germany_ALL['trd_delivery_time_start'] == time_1]

    df_single_block['diff'] = (df_single_block['trd_delivery_time_start'] - df_single_block['trd_execution_time'])


    all_data = df_single_block[df_single_block['diff'] <= pd.Timedelta(X, unit='H')]
    train_data = all_data[all_data['diff'] >= pd.Timedelta(2, unit='H')]
    last_data = all_data[all_data['diff'] <= pd.Timedelta(2, unit='H')]                        
    
    # print(df_single_block[['trd_delivery_time_start', 'trd_delivery_time_end', 'trd_execution_time']])

    # stepwise_fit = auto_arima(train_data["trd_price"].values, trace=False, suppress_warnings=True)

    # all_data.iloc[::1, :].plot(x='trd_execution_time', y='trd_price',
    #                                   title= str(time_1),
    #                                   xlabel="Time", ylabel="Price", legend=False)
    # plt.show()
    return all_data, train_data, last_data

# "2022-02-03 22:00:00"
def check1():
    x = []
    for i in range(100):
        x.append(i)
    stepwise_fit = auto_arima(x, trace=True, suppress_warnings=True, D=1, seasonal = False) 
    
    stepwise2 = auto_arima(x, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, trace=True, stepwise = True)
    print(stepwise_fit, stepwise2)

    dates = pd.date_range(end = datetime.today(), periods = 100)
    tuplee = list(zip(x, dates))
    data = pd.DataFrame(tuplee, columns =["a", 'b'])
    print(data)
    

    adf_test = ADFTest(alpha=0.05)
    print(adf_test.should_diff(data["a"].values))

    mod = ARIMA(x, order=(1,1,1))
    res = mod.fit()
    # res.plot_predict(dynamic=False)
    print(res.summary())

    start = len(x)
    end = len(x) + 10
    pred = res.predict(start=start,end=end,typ='levels')
    plt.plot(pred)
    plt.show()



# Arima_hour_block("2022-02-10 12:00:00")
def Arima_hour_block(time_1):
    all_data, train_data, last_data = One_block_X_h(time_1)
    # print(all_data.shape, "train data: ", train_data.shape)

    
    stepwise_fit = auto_arima(train_data["trd_price"], trace=False, suppress_warnings=False)
    print(stepwise_fit)
    stepwise_fit = str(stepwise_fit).split("(")[1].split(",")
    order = (int(stepwise_fit[0]), int(stepwise_fit[1]), int(stepwise_fit[2][0]))
    

    # train_data['trd_execution_time'] = pd.to_datetime(train_data['trd_execution_time'])
    # adf_test = ADFTest(alpha=0.05)
    # print(adf_test.should_diff(np.diff(train_data["trd_price"])))

    mod = ARIMA(train_data['trd_price'], order=(3,0,3))
    res = mod.fit()
    
    # print(res.summary())

    start = len(all_data["trd_price"].values) - len(train_data["trd_price"].values)
    end = len(all_data["trd_price"].values) - 1
    pred = res.predict(start=start,end=end,typ='levels', color="red")
    print("DOLŽINA PRED JE ", len(pred), all_data["trd_price"].values)
    plt.plot(pred, label= "Prediction")
    plt.plot(all_data["trd_price"].values, label="Data", color="blue")
    plt.legend()
    plt.show()
    # return all_data

#[18.969573847491006, 29.414899655977308, -26.389818556303993, -80.32598957321147, -27.6785858713873, 76.20111720392791, -85.89416996156817, 91.44057315901676, 34.502638038584905, -13.294545527303825, -48.554566071377266, -16.534389246287304, 41.37197214409929, 42.03197633617344, 13.383906298298456, 11.231920778978093, 32.816675760957395, -26.459818024096762, 23.626952905464947, -7.040889512061767, 27.204945386526333, 13.053460902922382, -6.26366336612395, -57.26182815476264]
# "2022-02-14 23:00:00"
# Prophet_hour_block("2022-02-14 23:00:00", [])
# best_interval(date="2022-02-10", number_of_days=50, freq=2, start_time = 8.5, end_time=9)
def Prophet_hour_block(time_1, sez, X=5):
    stevec = 0
    
    all_data, train_data, last_data = One_block_X_h(time_1, X)

    stevec += 1
    
    train_data = train_data[["trd_execution_time", "trd_price"]]
    train_data.rename(columns = {'trd_execution_time':'ds', 'trd_price':'y'}, inplace = True)
    all_data.rename(columns = {'trd_execution_time':'ds', 'trd_price':'y'}, inplace = True)
    print(time_1)
    if time_1 == "2022-02-15 1:00:00":
        sez.append("N")
        return sez
    try:
        m = Prophet()
        with suppress_stdout_stderr():
            m.fit(train_data)
    except ValueError:
        sez.append("N")
        return sez

    future = m.make_future_dataframe(periods=120, freq="1min")
    forecast = m.predict(future)

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

    Eror = last_pred - last_all
    sez.append((Eror, in_interval, ten_prct, twenty_prct, thirty_prct, fifty_prct, hundred_prct))
    

    pred = future.values[-120:]
    dataa = last_data.values





    ### Buy or sell strategy 30%


    ### ERROR but have to fix the lenghts
    # rmse=sqrt(mean_squared_error(pred, dataa))
    # print(rmse)

    ############# PLOT
    # m.plot(forecast)
    # plt.show()

    # ax = forecast.plot(x="ds", y="yhat", color="red", label="Prediciton")
    # all_data.plot(x="ds", y="y", color="blue", label="data", ax=ax)
    # forecast.plot(x="ds", y="yhat_lower", color="green", label="lower_bound", ax=ax)
    # forecast.plot(x="ds", y="yhat_upper", color="green", label="upper_bound", ax=ax)
    # plt.legend()
    # plt.show()

    return sez

    

# basic plot
def plott(date = "2022-02-10"):
    for i in range(24):
        if i == 0:
            string = "{} 0{}:00:00".format(date, i)
            Arima_hour_block(string)    
        string = "{} {}:00:00".format(date, i)
        Arima_hour_block(string)

# plots all
def plot_plot(date="2022-02-10", number_of_days=5, X=5):
    dates = pd.date_range(start = date, periods = number_of_days)
    sez2= []
    stevilo2_2 = 0
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
        sez, stevilo2, napaka, stevilo_nanov, procenti_stevila = plottt(i.split(" ")[0], X)
        dolzina += 24 - stevilo_nanov
        stevilo2_2 += stevilo2
        napaka_2 += napaka

        ten_prct += procenti_stevila[0]
        twenty_prct += procenti_stevila[1]
        thirty_prct += procenti_stevila[2]
        fifty_prct += procenti_stevila[3]
        hundred_prct += procenti_stevila[4]

        sez2.append(sez)
    delez = stevilo2_2/dolzina
    napaka_abs_all = napaka_2/dolzina

    in_ten_prct = ten_prct/dolzina
    in_twenty_prct = twenty_prct/dolzina
    in_thirty_prct = thirty_prct/dolzina
    in_fifty_prct = fifty_prct/dolzina
    in_hundred_prct = hundred_prct/dolzina

    

    print("Napake za število dni", number_of_days, "\n", "kolikokrat zadanemo interval Propheta ", delez, "\n Kolikšna je povprečna napaka ", napaka_abs_all, "postopoma v intervalih: \n", "10 procentni absolutni interval", in_ten_prct, " \n",
    "20 procentov", in_twenty_prct, " \n", "30 procentov", in_thirty_prct, " \n", "50 procentov", in_fifty_prct, " \n", "100 procentov", in_hundred_prct)

    return (in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, delez, napaka_abs_all)

def best_interval(date="2022-02-10", number_of_days=5, freq=9, start_time = 4, end_time=6):
    ten_prct = []
    twenty_prct = []
    thirty_prct = []
    fifty_prct = []
    hundred_prct = []
    in_prophet = []
    napaka_abs_all_sez = []

    for i in np.linspace(start_time, end_time, freq):
        in_ten_prct, in_twenty_prct, in_thirty_prct, in_fifty_prct, in_hundred_prct, delez, napaka_abs_all = plot_plot(date, number_of_days, i)
        f= open("Information_about_prohpet_november.txt", "a+")
        f.write(f"\n \n Začetni datum {date}, X = {i}, in za koliko dni je narejena analiza {number_of_days} \n kolikokrat zadanemo interval Propheta {delez} \n Kolikšna je povprečna napaka {napaka_abs_all} \n Postopoma v intervalih: \n 10 procentni absolutni interval {in_ten_prct}  \n 20 procentni absolutni interval {in_twenty_prct} \n 30 procentni absolutni interval {in_thirty_prct}  \n 50 procentni interval {in_fifty_prct}  \n 100 procentni interval {in_hundred_prct}")
        f.close()
        
        in_prophet.append(delez)
        ten_prct.append(in_ten_prct)
        twenty_prct.append(in_twenty_prct)
        thirty_prct.append(in_thirty_prct)
        fifty_prct.append(in_fifty_prct)
        hundred_prct.append(in_hundred_prct)
        napaka_abs_all_sez.append(napaka_abs_all)

    f= open("Information_about_prohpet_november.txt", "a+")
    f.write("\n \n \n Seznam: Absolutna napaka, fb prophet, 10%, 20%, 30%, 50%, 100%")
    f.write(f"\n Seznam = [{napaka_abs_all_sez},{in_prophet},{ten_prct},{twenty_prct},{thirty_prct},{fifty_prct},{hundred_prct}]")
    f.close()
    
# plottt(date = "2022-02-15", X=8.5)
# plots one day and calculates the margin
def plottt(date = "2022-02-10", X=5):
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
            sez = Prophet_hour_block(string, [], X)    
        else:
            string = "{} {}:00:00".format(date, i)
            sez = Prophet_hour_block(string, sez, X)
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
    delez = stevilo2/(len(sez) - stevilo_nanov)* 100
    napaka_delez = napaka/(len(sez) - stevilo_nanov)
    # print(sez,"v procentih kok jih zadane ta interval ", delez, "absolutna napaka napovedi za ta dan ", napaka_delez)
    return sez, stevilo2, napaka, stevilo_nanov, procenti_stevila
 
####################################################################################################################################################################################################################################################################

#All dates
# start_date = "2021-11-08"
# end_date = "2022-05-01"

def get_intra_day(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = df_intra_day_germany

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


# dftest = adfuller(get_intra_day(start_date='2022-02-02', end_date='2022-03-24')["trd_price"], autolag = 'AIC')
# print("P value for unchanged graph is ", dftest[1])



def difference(start_date='2022-02-02', end_date='2022-03-24', arima= None):
    drugi = np.diff(get_intra_day(start_date, end_date)["trd_price"].values)
    drugi = np.insert(drugi, 0, 0)

    df = get_intra_day(start_date, end_date)
    df["difference"] = drugi

    dftest = adfuller(df["difference"], autolag = 'AIC')
    print("P-Value test is = ", dftest[1])
    if arima != None:
        stepwise_fit = auto_arima(df["difference"], trace=False, suppress_warnings=True)
        print(stepwise_fit)
    #Plot difference 
    df.plot(x='trd_delivery_time_start', y='difference')
    plt.show()
    
    train=df.iloc[:-24]
    test=df.iloc[-24:]
    return (train, test, drugi)



def difference_predicition(start_date='2022-02-02', end_date='2022-03-24'):
    train, test, drugi = difference(start_date, end_date)

    mod = sm.tsa.arima.ARIMA(train['difference'].values, order=(2, 0, 5))
    res = mod.fit()
    # print(res.summary())

    start = len(train)
    end = len(train)+len(test) - 1
    pred = res.predict(start=start,end=end,typ='levels')
    plt.plot(pred, label='ARIMA prediction')
    plt.plot(test['difference'].values, label='Data')
    plt.xlabel("Time (Hour Blocks)")
    plt.ylabel("Price")
    plt.legend()    
    plt.show()
    
    drugi[0] = (get_intra_day(start_date, end_date)["trd_price"].values[0])
    result = (drugi.cumsum())

    pred[0] = result[-24]
    prediction = pred.cumsum()
    
    novi = [None] * len(result)
    novi[-24:] = prediction

    plt.plot(prediction, label="prediction")
    plt.plot(result[-24:], label="data")
    plt.legend()
    plt.show()

    plt.plot(result, label="Data")
    plt.plot(novi, label="prediciton")
    plt.show()

    ### Test how good the prediction is, very big rmse :/
    # print(test['difference'].values.mean())
    # rmse = sqrt(mean_squared_error(pred, test['difference'].values))
    # print(rmse)

################################################################################################################################################################################

def get_intra_day2(start_date='2021-11-08', end_date='2022-05-01'):
    df_intra_day = df_intra_day_germany_ALL

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



def difference2(start_date='2021-11-08', end_date='2022-05-01', arima= None):
    drugi = np.diff(get_intra_day2(start_date, end_date)["trd_price"].values)
    drugi = np.insert(drugi, 0, 0)

    
    df = get_intra_day2(start_date, end_date)
    df["difference"] = drugi

    dftest = adfuller(df["difference"], autolag = 'AIC')
    print("P-Value test is = ", dftest[1])
    if arima != None:
        stepwise_fit = auto_arima(df["difference"], trace=False, suppress_warnings=True)
        print(stepwise_fit)
    #Plot difference 
    # df.plot(x='trd_delivery_time_start', y='difference')
    # plt.show()
    
    train=df.iloc[:-24]
    test=df.iloc[-24:]
    return (train, test, drugi)


def difference_predicition2(start_date='2021-11-08', end_date='2022-05-01'):
    train, test, drugi = difference2(start_date, end_date)

    mod = sm.tsa.arima.ARIMA(train['difference'].values, order=(2, 0, 2))
    res = mod.fit()
    # print(res.summary())

    start = len(train)
    end = len(train) + len(test) - 1
    pred = res.predict(start=start,end=end,typ='levels')
    plt.plot(pred, label='ARIMA prediction')
    plt.plot(test['difference'].values, label='Data')
    plt.xlabel("Time (Hour Blocks)")
    plt.ylabel("Price")
    plt.legend()    
    plt.show()
    
    drugi[0] = (get_intra_day2(start_date, end_date)["trd_price"].values[0])
    result = (drugi.cumsum())

    pred[0] = result[-24]
    prediction = pred.cumsum()
    
    novi = [None] * len(result)
    novi[-24:] = prediction

    plt.plot(prediction, label="prediction")
    plt.plot(result[-24:], label="data")
    plt.legend()
    plt.show()

    plt.plot(result, label="Data")
    plt.plot(novi, label="prediciton")
    plt.show()


################################################################################################################################################################################
################################################################################################################################################################################





def data_pandas(start_date='2022-02-02', end_date='2022-03-24'):
    drugi = get_intra_day(start_date, end_date)[["trd_price", "trd_delivery_time_start"]]
    drugi.to_csv("filename.csv")

def data_pandas_all(start_date='2021-11-08', end_date='2022-05-01'):
    drugi = get_intra_day2(start_date, end_date)[["trd_price", "trd_delivery_time_start"]]
    drugi.to_csv("All_data.csv")

# data_pandas_all(start_date='2021-11-08', end_date='2022-05-01')
#Can't use logaritmic bc of negative values
def logaritmic(start_date='2022-02-02', end_date='2022-03-24'):
    pass



################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################









if __name__ == "__main__":
    # get_intra_day(start_date='2022-02-02', end_date='2022-03-24').plot(x='trd_delivery_time_start', y='trd_price')
    # plt.show()
    # difference(start_date='2022-02-02', end_date='2022-03-24', arima= None)
    # difference_predicition(start_date='2022-02-02', end_date='2022-03-24')


    # ### All data
    # difference_predicition2(start_date='2021-11-08', end_date='2022-05-01')
    # difference2(start_date='2021-11-08', end_date='2022-05-01')


    ### Prophet 
    # Prophet_hour_block("2021-11-09 00:00:00", [])
    # plot_plot(date="2021-11-10", number_of_days=1, X=5)
    # best_interval(date="2021-11-10", number_of_days=50, freq=7, start_time = 3, end_time=9)
    
    pass