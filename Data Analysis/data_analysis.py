import pandas
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import tree
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from data_parsing import get_intra_day_data_for_region, get_day_ahead_data, get_sun_data, get_wind_data, \
    get_wind_forecast, get_data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import dtreeviz
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta

pd.set_option('display.expand_frame_repr', False)
df_intra_day_germany = get_intra_day_data_for_region("GERMANY")



def single_block():
    df_single_block = df_intra_day_germany[df_intra_day_germany['trd_delivery_time_start'] == "2022-02-01 10:00:00"]
    print(df_single_block.trd_buy_delivery_area.unique())
    print(df_single_block.trd_sell_delivery_area.unique())
    # print(df_single_block[['trd_delivery_time_start', 'trd_delivery_time_end', 'trd_execution_time']])

    df_single_block.iloc[::1, :].plot(x='trd_execution_time', y='trd_price',
                                      title="2022-02-01 10:00:00 - 2022-02-01 11:00:00",
                                      xlabel="Time", ylabel="Price", legend=False)
    plt.show()
    return df_single_block                             
    


def transform_day_ahead(df_day_ahead):
    df_day_ahead = df_day_ahead.groupby('DateCET').apply(lambda row: pd.DataFrame(
        {'trd_delivery_time_start': pd.date_range(row['DateCET'].iloc[0],
                                                  periods=len(df_day_ahead.columns[1:]), freq='1H'),
         'trd_price': np.repeat(np.array([row[col].iloc[0] for col in df_day_ahead.columns[1:]]), 1)}))
    df_day_ahead.reset_index(inplace=True, drop=True)
    return df_day_ahead


def get_intra_day_min_max_mean(df_intra_day=df_intra_day_germany, interval='15min', on='trd_execution_time'):
    df_intra_day = df_intra_day.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                                              trd_price_min=('trd_price', np.min), 
                                                              trd_price_max=('trd_price', np.max))
    df_intra_day.reset_index(inplace=True)
    return df_intra_day


def average_price_at_given_time(start_date='2022-02-03', end_date='2022-02-03'):
    df_intra_day = get_intra_day_min_max_mean()
    df_intra_day = df_intra_day[(df_intra_day['trd_execution_time'] >= f'{start_date} 00:00:00') &
                                (df_intra_day['trd_execution_time'] <= f'{end_date} 23:00:00')]

    ax = df_intra_day.plot(x='trd_execution_time', y='trd_price_mean')
    plt.fill_between(df_intra_day['trd_execution_time'].dt.to_pydatetime(),
                     df_intra_day.trd_price_min,
                     df_intra_day.trd_price_max,
                     facecolor='lightblue', alpha=0.4, interpolate=True)

    df_day_ahead = get_day_ahead_data()
    df_day_ahead = df_day_ahead[(df_day_ahead['DateCET'] >= start_date) & (df_day_ahead['DateCET'] <= end_date)]
    df_day_ahead = transform_day_ahead(df_day_ahead)
    print(df_day_ahead)
    df_day_ahead.plot(x='trd_delivery_time_start', y='trd_price', use_index=True, ax=ax)

    ax.legend(["Intra-day average", "Min/Max Intra-day", "Day-ahead"])
    ax.set_title("Average price at given time")
    ax.set_xlabel("Price")
    ax.set_ylabel("Time")
    plt.tight_layout()
    plt.figure(figsize=(200, 150))
    plt.show()


def average_price_of_product(start_date='2022-02-03', end_date='2022-02-03', hours_before_closing=None):
    df_intra_day = df_intra_day_germany
    # Filter out to only trades that happened at most n-hours before closing
    if hours_before_closing:
        df_intra_day['diff'] = (df_intra_day['trd_delivery_time_start']
                                - df_intra_day['trd_execution_time'])
        df_intra_day = df_intra_day[df_intra_day['diff'] <= pd.Timedelta(hours_before_closing, unit='H')]
    df_intra_day = get_intra_day_min_max_mean(df_intra_day, interval='1H', on='trd_delivery_time_start')

    df_intra_day = df_intra_day[(df_intra_day['trd_delivery_time_start'] >= f'{start_date} 00:00:00') &
                                (df_intra_day['trd_delivery_time_start'] <= f'{end_date} 23:00:00')]

    ax = df_intra_day.plot(x='trd_delivery_time_start', y='trd_price_mean')
    plt.fill_between(df_intra_day['trd_delivery_time_start'].dt.to_pydatetime(),
                     df_intra_day.trd_price_min,
                     df_intra_day.trd_price_max,
                     facecolor='lightblue', alpha=0.4, interpolate=True)

    df_day_ahead = get_day_ahead_data()
    df_day_ahead = df_day_ahead[(df_day_ahead['DateCET'] >= start_date) & (df_day_ahead['DateCET'] <= end_date)]
    df_day_ahead = transform_day_ahead(df_day_ahead)
    df_day_ahead.plot(x='trd_delivery_time_start', y='trd_price', use_index=True, ax=ax)

    ax.legend(["Intra-day average", "Min/Max Intra-day", "Day-ahead"])
    if hours_before_closing:
        ax.set_title(f"Average price of product {hours_before_closing} hour(s) before closing")
    else:
        ax.set_title("Average price of product")
    ax.set_xlabel("Price")
    ax.set_ylabel("Product start time")
    plt.tight_layout()
    plt.figure(figsize=(200, 150))
    plt.show()


def get_pct_change_dataframe(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = get_intra_day(start_date, end_date)

    df_day_ahead = get_day_ahead_data()
    df_day_ahead = df_day_ahead[df_day_ahead['DateCET'] >= start_date]

    df_day_ahead = transform_day_ahead(df_day_ahead)

    df_pct_change = df_intra_day

    df_pct_change.rename(columns={'trd_price': 'trd_price_intra_day'}, inplace=True)

    df_pct_change['trd_price_day_ahead'] = df_day_ahead['trd_price']

    # Calculate percentage change (V_2 - V_1) / V_1 (remove abs if you want to differentiate between increase / decrease
    df_pct_change['percentage_change'] = ((df_pct_change['trd_price_intra_day'] - df_day_ahead['trd_price']) /
                                          df_day_ahead['trd_price']) * 100

    df_pct_change = df_pct_change[['trd_delivery_time_start',
                                   'trd_price_day_ahead',
                                   'trd_price_intra_day',
                                   'percentage_change']]
    print("DF_PCT_CHANGE:", df_pct_change)

    # Drop any inf values that were created due to day_ahead price of 0.0 (corner case)
    return df_pct_change.replace([np.inf, -np.inf], np.nan).dropna()


def day_ahead_as_intra_day_prediction_accuracy(start_date='2022-02-02', end_date='2022-03-24'):
    df_pct_change = get_pct_change_dataframe(start_date, end_date)

    print(df_pct_change.nlargest(n=5, columns=['percentage_change']))

    # Quantile box plot
    plt.boxplot(df_pct_change['percentage_change'])
    plt.title("Before removing outliers")
    plt.show()

    # Before removing outliers
    print("Before removing outliers")
    print("MIN: ", df_pct_change['percentage_change'].min())
    print("MAX: ", df_pct_change['percentage_change'].max())
    print("MEAN: ", df_pct_change['percentage_change'].mean())

    """
    z_scores = zscore(df_pct_change[['trd_price_intra_day', 'percentage_change', 'trd_price_dayahead']])
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < 3).all(axis=1)

    df_pct_change = df_pct_change[filtered_entries]
    """

    # Calculate limits for extreme outliers (3 instead of 1.5) with interquartile range
    # Will be more accurate when we get more data
    q1 = df_pct_change['percentage_change'].quantile(0.25)
    q3 = df_pct_change['percentage_change'].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 3 * iqr
    upper_limit = q3 + 3 * iqr
    # print(lower_limit, upper_limit)
    print("Under lower limit", df_pct_change[df_pct_change['percentage_change'] < lower_limit].shape[0])
    print("Over upper limit", df_pct_change[df_pct_change['percentage_change'] > upper_limit].shape[0])

    # Remove extreme outliers
    df_pct_change = df_pct_change[(df_pct_change['percentage_change'] >= lower_limit)
                                  & (df_pct_change['percentage_change'] <= upper_limit)]

    # After removing outliers
    print("After removing outliers")
    print("MIN: ", df_pct_change['percentage_change'].min())
    print("MAX: ", df_pct_change['percentage_change'].max())
    print("MEAN: ", df_pct_change['percentage_change'].mean())

    plt.boxplot(df_pct_change['percentage_change'])
    plt.title("After removing outliers")
    plt.show()

    # Plot percentage change
    ax = df_pct_change.plot(x='trd_delivery_time_start', y='percentage_change', kind='barh', legend=None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_xlabel("Percentage Change")
    ax.set_ylabel("Product start time")
    plt.tight_layout()
    plt.show()

    # check if a certain time of day has highest variance
    times = pd.to_datetime(df_pct_change['trd_delivery_time_start'])
    print("HERE")
    # print(df_pct_change.groupby(times.dt.hour)['percentage_change'].transform('var').head(48))
    print(df_pct_change.groupby(times.dt.hour)['percentage_change'].agg(['var', 'std']))
    # print(df)


def increase_decrease_analysis(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = df_intra_day_germany

    # Time to start of block from purchase time (execution time)
    df_intra_day['diff'] = (df_intra_day['trd_delivery_time_start']
                            - df_intra_day['trd_execution_time'])
    df_intra_day = df_intra_day[(df_intra_day['trd_delivery_time_start']
                                 >= start_date)
                                & (df_intra_day['trd_delivery_time_end']
                                   <= f'{end_date} 00:00:00')]

    df_intra_day = df_intra_day[df_intra_day['diff'] <= pd.Timedelta(1, unit='H')]

    df_intra_day = df_intra_day.resample('H', on='trd_delivery_time_start').mean().reset_index()

    df_day_ahead = get_day_ahead_data()
    df_day_ahead = df_day_ahead[(df_day_ahead['DateCET'] >= start_date) & (df_day_ahead['DateCET'] <= end_date)]

    df_day_ahead = transform_day_ahead(df_day_ahead)

    # Some products might not be traded in the hour before closing, so drop those
    print(df_intra_day.shape[0])
    print(sum([True for idx, row in df_intra_day.iterrows() if any(row.isnull())]))
    df_intra_day = df_intra_day
    df_intra_day.dropna(inplace=True)

    df_increase_decrease = df_day_ahead
    df_increase_decrease['increase_decrease'] = np.sign(df_day_ahead['trd_price'] - df_intra_day['trd_price'])

    # Drop any 100% accurate predictions
    df_increase_decrease.drop(df_increase_decrease.loc[df_increase_decrease['trd_price'] == 0].index, inplace=True)
    print(df_increase_decrease['increase_decrease'].value_counts())


def weather_analysis(start_date='2022-02-02', end_date='2022-03-24'):
    df_wind_historic = get_wind_data()
    df_wind_historic['Date'] = pd.to_datetime(df_wind_historic['Date'], format='%Y%m%d')
    # Fix 24th hour of day to actually be the 0th hour of the next day
    df_wind_historic.insert(2, '0', df_wind_historic['24'].shift(12))
    df_wind_historic.dropna(inplace=True)
    df_wind_historic.drop(columns='24', inplace=True)
    print("DF WIND HISTORIC\n", df_wind_historic)

    # Creates a dataframe for each region in appropriate shape
    wind_historic_dfs = []
    for region in df_wind_historic['Region'].unique():
        df = df_wind_historic[df_wind_historic['Region'] == region]
        df = df.groupby('Date').apply(lambda row: pd.DataFrame(
            {'timestamp': pd.date_range(row['Date'].iloc[0],
                                        periods=len(df.columns[2:]), freq='1H'),
             region: np.repeat(np.array([row[col].iloc[0] for col in df.columns[2:]]), 1)}))
        df.reset_index(inplace=True, drop=True)
        # print(f'{region}:\n', df)
        wind_historic_dfs.append(df)

    # Concatenates dataframes for each region into a single dataframe
    df_wind = pd.concat(wind_historic_dfs, axis=1)
    df_wind = df_wind.loc[:, ~df_wind.columns.duplicated()]
    df_wind['wind_mean'] = df_wind.iloc[:, 1:12].mean(axis=1)
    df_wind.rename(columns={"timestamp": "trd_delivery_time_start"}, inplace=True)

    print("DF_WIND\n", df_wind)

    df = pd.merge(df_wind, get_pct_change_dataframe(start_date, end_date), on='trd_delivery_time_start')

    print("MERGED DF\n", df)
    print(df[df.columns[1:14]].apply(lambda x: x.corr(df['percentage_change'], method='pearson')))

    # df['trd_delivery_time_start'] = df['trd_delivery_time_start'].apply(lambda x: x.toordinal())
    # regr = DecisionTreeRegressor(random_state=1234, max_depth=5)
    # model = regr.fit(X=df[df.columns[0:-3]], y=df['trd_price_intra_day'])

    """
    print(df.iloc[:1][df.columns[0:-3]])
    print(df.columns[0:-3])
    viz = dtreeviz(regr,
                   df[df.columns[0:-3]],
                   df['trd_price_intra_day'],
                   target_name='trd_price_intra_day',
                   orientation='LR',  # left-right orientation
                   feature_names=df.columns[0:-3],
                   X=df.iloc[:1][df.columns[0:-3]])  # need to give single observation for prediction

    viz.view()
    """
    # plt.figure(figsize=(20, 20))
    # tree.plot_tree(regr, fontsize=10)
    # plt.savefig('tree_high_dpi', dpi=100)

    """ Ignore sun data for now
    
    df_sun_historic = get_sun_data()
    df_sun_historic['Date'] = pd.to_datetime(df_sun_historic['Date'], format='%Y%m%d')
    sun_historic_dfs = []
    for region in df_sun_historic['Region'].unique():
        df = df_sun_historic[df_sun_historic['Region'] == region]
        df = df.groupby('Date').apply(lambda row: pd.DataFrame(
            {'timestamp': pd.date_range(row['Date'].iloc[0],
                                        periods=len(df.columns[2:]), freq='1H'),
             region: np.repeat(np.array([row[col].iloc[0] for col in df.columns[2:]]), 1)}))
        df.reset_index(inplace=True, drop=True)
        sun_historic_dfs.append(df)

    # Concatenates dataframes for each region into a single dataframe
    df_sun = pd.concat(sun_historic_dfs, axis=1)
    df_sun = df_sun.loc[:, ~df_sun.columns.duplicated()]
    print("Sun\n:", df_sun)
    print(df_sun.iloc[12])
    """


def forecast_weather_analysis(start_date='2022-02-02', end_date='2022-03-24'):
    df_wind_forecast = get_wind_forecast()
    df_wind_forecast = df_wind_forecast[df_wind_forecast['Time'] == 12]
    df_wind_forecast['Date'] = pd.to_datetime(df_wind_forecast['Date'], format='%Y%m%d')
    # Fix 24th hour of day to actually be the 0th hour of the next day
    df_wind_forecast.drop(columns=['Time', '18', '30'], inplace=True)
    df_wind_forecast.insert(2, '0', df_wind_forecast['24'].shift(12))
    df_wind_forecast.dropna(inplace=True)
    print("DF WIND FORECAST\n", df_wind_forecast)

    # Creates a dataframe for each region in appropriate shape
    wind_historic_dfs = []
    for region in df_wind_forecast['Region'].unique():
        df = df_wind_forecast[df_wind_forecast['Region'] == region]
        df = df.groupby('Date').apply(lambda row: pd.DataFrame(
            {'timestamp': pd.date_range(row['Date'].iloc[0],
                                        periods=len(df.columns[2:]), freq='12H'),
             region: np.repeat(np.array([row[col].iloc[0] for col in df.columns[2:]]), 1)}))
        df.reset_index(inplace=True, drop=True)
        # print(f'{region}:\n', df)
        wind_historic_dfs.append(df)

    # Concatenates dataframes for each region into a single dataframe
    df_wind = pd.concat(wind_historic_dfs, axis=1)
    df_wind = df_wind.loc[:, ~df_wind.columns.duplicated()]
    df_wind['wind_mean'] = df_wind.iloc[:, 1:12].mean(axis=1)
    df_wind.rename(columns={"timestamp": "trd_delivery_time_start"}, inplace=True)

    print("DF_WIND\n", df_wind)

    df = pd.merge(df_wind, get_pct_change_dataframe(start_date, end_date), on='trd_delivery_time_start')

    print("MERGED DF\n", df)
    print(df[df.columns[1:14]].apply(lambda x: x.corr(df['percentage_change'], method='pearson')))


def get_intra_day(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = df_intra_day_germany

    # Time to start of block from purchase time (execution time)
    df_intra_day['diff'] = (df_intra_day['trd_delivery_time_start']
                            - df_intra_day['trd_execution_time'])
    

    # print(df_intra_day['diff'].head())
    
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


def granger_causality(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = get_intra_day(start_date, end_date)

    by_hours = []

    for hour in range(24):
        stationary = False
        difference_order = 0
        df_hour = df_intra_day[df_intra_day['trd_delivery_time_start'].dt.hour == hour]['trd_price'].rename(str(hour))
        hour = str(hour)
        while not stationary:
            print(df_hour)
            # check if the time series is stationary or not
            print(100 * '-')
            print(f'HOUR: {hour}')
            result = adfuller(df_hour)
            print(result)
            print(f'Augmented Dickey-Fuller Test Statistics: {result[0]}')
            print(f'P-value: {result[1]}')
            print(f'Truncation lag parameter: {result[2]}')
            print(f'Critical_values: {result[4]}')

            if result[1] > 0.05:
                stationary = False
                print("Series is not stationary according to ADF\n\n")
            else:
                stationary = True
                print("Series is stationary according to ADF\n\n")

            result = kpss(df_hour, regression='ct')

            print(f'Kwiatkowski-Phillips-Schmidt-Shin Test Statistics: {result[0]}')
            print(f'P-value: {result[1]}')
            print(f'Truncation lag parameter: {result[2]}')
            print(f'Critical_values: {result[3]}')

            if result[1] > 0.05:
                # stationary = False
                print("Series is not stationary according to KPSS\n\n")
            else:
                # stationary = True
                print("Series is stationary according to KPSS\n\n")

            if not stationary:
                df_hour = df_hour.diff().dropna()
                difference_order += 1

        by_hours.append(df_hour.reset_index()[hour])
        print(f'Difference order for {hour}: {difference_order}')
    print(by_hours[1])
    df_by_hours = pandas.concat(by_hours, axis=1).dropna()

    print("Dataframe in question\n", df_by_hours)

    grangers_causation_matrix = get_grangers_causation_matrix(df_by_hours, variables=df_by_hours.columns)

    print(grangers_causation_matrix)


# Taken from
# https://stackoverflow.com/questions/58005681/is-it-possible-to-run-a-vector-autoregression-analysis-on-a-large-gdp-data-with
# Might make sense to change maxlag to 24 when more data will be available
def get_grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=13):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def stats_plots():
    df_intraday = get_intra_day()
    df_intraday['trd_delivery_time_start'] = pd.to_datetime(df_intraday['trd_delivery_time_start'])
    df_intraday = df_intraday.set_index('trd_delivery_time_start')
    result = seasonal_decompose(df_intraday, period=1)
    result.plot()
    plt.show()
    result = seasonal_decompose(df_intraday, period=24)
    result.plot()
    plt.show()
    result = seasonal_decompose(df_intraday, period=168)
    result.plot()
    plt.show()
    """
    result = seasonal_decompose(df_intraday, period=730)
    result.plot()
    plt.show()
    """
    plot_acf(df_intraday)
    plt.show()
    plot_pacf(df_intraday)
    plt.show()


if __name__ == "__main__":
    # average_price_at_given_time()
    # average_price_of_product()
    # average_price_at_given_time(start_date='2022-02-01', end_date='2022-03-24')
    # average_price_of_product(start_date='2022-02-01', end_date='2022-03-24')
    # average_price_of_product(start_date='2022-02-01', end_date='2022-03-24', hours_before_closing=1)
    # average_price_of_product(start_date='2022-02-03', end_date='2022-02-03', hours_before_closing=1)
    # day_ahead_as_intra_day_prediction_accuracy()
    # increase_decrease_analysis()
    # forecast_weather_analysis()
    # granger_causality()
    # stats_plots()
    pass

""" 
TODO:
    Information gain bolj glede na to a podatki dejansko pomagajo
    
    Ali je v določenih delih dneva večje odstopanje kot v drugem? (ne po grafu)
    

DONE: 
    Ali obstaja kakšna korelacija (kavzalnost) med gibanjem urnih produktov? 
    Če pride do premika v nekem urnem produktu, ali se tudi ostali za njim?
    Grangerjev test (napovedljivost časovnih vrst)
    
    Box plot po kvanitlih za odstopanja
    
Day-ahead average
Intra-day average (day-ahead ceno in kako se to primerja z average produktov v zadnje pol ure,
kakšna je distribucija razlik (ne procentov, histogram)
Poglej še absolutno razliko za var / std

Veliko večja šansa je, da če bomo vedno short, rabimo sam pravi money managment poračunat
(Če veš v koliko procentih primerov dobiš, recimo 60%, se to že da izkoristt)

Vremenski podatki - naključna točka (0 korelacije), uteženost glede na oddaljenost

- novi podatki (dobiva v tem tednu)
Za naprej:
  - analiza day-ahead, realizacija (kako blizu smo), distribucija (mogoče normalna?)
  - baseline model (napovemo ceno iz day-aheada za vse ure v dnevu (
    napoved zadnjo uro / zadnje pol ure, preveri koliko tradeov je takrtat),
    napaka je razlika)
  - modeliranje (autoregresija - ARIMA itd., FB, Amazon itd., poda se krivuljo (časovno vrsto), 
    pustimo si še par ur trejdov do izteka napovedanega produkta)
    prvih 5 ur morda ne trejdamo (vprašanje koliko podatkov se do takrat pridobi za druge produkte)
    Kako združevat podatke v časovno vrsto? (Povprečenje / model specifično za določen produkt)
    Druga opcija: feature based, bolj klasični modeli (input je lahko day-ahead cena, premik / volatilnost  v tistem času,
    kaj se je dogajalo zadnje pol ure, linearna interpolacija (kako strmo pada cena), statistične vrednosti (kakšna je povprečna vrednost cene za obdobje))
    Obdobje cen -> preslikano v featurje (random forest, xgboost) (feature importance - shap)
    Lahko tudi klasifikacija (nizka, visoka cena, več cenovnih classov (5 do 10), za sprobat kaj je smislno)
    Dnevne / nočne ure, dan v tednu (weekend)
    "WISHLIST" - NN, osnovne vremenske napovedi (dobiva podatke, če pridem do tega), izberemo le del produktov za trejdat (recimo na podlagi statistik, model glede na to v katerih situacijah se dobro dela)
    Če se odločimo za klasifikacijo: - ostali pristopi morajo biti tudi ocenjeni po tem (baseline)
    Knjižnica - avtomatsko generiranje časovnih vrst / featurjev
"""



# def get_intra_day22(start_date='2022-02-02', end_date='2022-03-24', date='2022-02-02'):
#     df_intra_day = df_intra_day_germany

#     start_date = datetime.strptime(str(start_date), "%Y-%m-%d")
#     end_date = datetime.strptime(str(end_date), "%Y-%m-%d")

#     # Time to start of block from purchase time (execution time)
#     df_intra_day['diff'] = (df_intra_day['trd_delivery_time_start']
#                             - df_intra_day['trd_execution_time'])

                                
    
    
#     yesterday = start_date + timedelta(days = -1)
#     print(yesterday)
#     # print(df_intra_day['diff'].head())
    
#     df_intra_day = df_intra_day[(df_intra_day['trd_delivery_time_start']
#                                  >= yesterday)
#                                 & (df_intra_day['trd_delivery_time_end']
#                                    <= end_date)]
#     # print("to iscem", df_intra_day)                                   
#     df_intra_day = df_intra_day[df_intra_day['diff'] <= pd.Timedelta(5, unit='H')]

#     print("TU JE ZAJ VAŽNO KAJ PIŠE", df_intra_day)
#     # df_intra_day = df_intra_day.resample('H', on='trd_delivery_time_start').mean()
#     # df_intra_day.reset_index(inplace=True)

#     # # Some products might not be traded in the 5 hours before closing, so drop those
#     # df_intra_day.dropna(inplace=True)

#     df_single_block = df_intra_day_germany[df_intra_day_germany['trd_delivery_time_start'] == time_1]


#     return df_intra_day



    