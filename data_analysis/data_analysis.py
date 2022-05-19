import numpy as np
import pandas as pd
import pmdarima as pmd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests

from data_retrieval import get_wind_forecast, \
    get_transformed_day_ahead, get_intra_day_min_max_mean, get_diff, get_pct_change_dataframe, get_intra_day_by_hours

plot_path = "../data_analysis/plots"
csv_path = "../data_analysis/csv"


def plot_price_at_given_time(start_date='2021-11-09',
                             end_date='2022-03-23'):
    """
    Plots intra-day average, min/max area and day-ahead price
    :param start_date: start date of plot (x-axis)
    :param end_date: end date of plot (x-axis)
    """
    df_intra_day = get_intra_day_min_max_mean(start_date=start_date, end_date=end_date)

    ax = df_intra_day.plot(x='trd_execution_time', y='trd_price_mean')
    plt.fill_between(df_intra_day['trd_execution_time'].dt.to_pydatetime(),
                     df_intra_day.trd_price_min,
                     df_intra_day.trd_price_max,
                     facecolor='lightblue', alpha=0.6, interpolate=True)

    df_day_ahead = get_transformed_day_ahead(start_date=start_date, end_date=end_date)
    df_day_ahead.plot(x='trd_delivery_time_start', y='trd_price', use_index=True, ax=ax)

    ax.legend(["Intra-day average", "Min/Max Intra-day", "Day-ahead"])
    ax.set_title("Price at given time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.figure(figsize=(200, 150))
    plt.savefig(f'{plot_path}/price_at_given_time_{start_date}-{end_date}.png')


def plot_average_price_of_product(max_time_before_closing=None,
                                  min_time_before_closing=None,
                                  unit=None,
                                  start_date='2021-11-09',
                                  end_date='2022-03-23'):
    """
    Plots the average price of product
    :param max_time_before_closing: only trades after maximum till product closes
    :param min_time_before_closing: only trades before minimum time till product closes
    :param unit: pandas unit for time before closing
        Possible values:
        ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
        ‘days’ or ‘day’
        ‘hours’, ‘hour’, ‘hr’, or ‘h’
        ‘minutes’, ‘minute’, ‘min’, or ‘m’
        ‘seconds’, ‘second’, or ‘sec’
        ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
        ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
        ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: None
    """
    df_intra_day = get_intra_day_min_max_mean(interval='1H',
                                              on='trd_delivery_time_start',
                                              max_time_before_closing=max_time_before_closing,
                                              min_time_before_closing=min_time_before_closing,
                                              unit=unit,
                                              start_date=start_date,
                                              end_date=end_date)

    ax = df_intra_day.plot(x='trd_delivery_time_start', y='trd_price_mean')
    plt.fill_between(df_intra_day['trd_delivery_time_start'].dt.to_pydatetime(),
                     df_intra_day.trd_price_min,
                     df_intra_day.trd_price_max,
                     facecolor='lightblue', alpha=0.4, interpolate=True)

    df_day_ahead = get_transformed_day_ahead(start_date=start_date, end_date=end_date)
    df_day_ahead.plot(x='trd_delivery_time_start', y='trd_price', use_index=True, ax=ax)

    ax.legend(["Intra-day average", "Min/Max Intra-day", "Day-ahead"])
    if max_time_before_closing:
        ax.set_title(f"Average price of product {max_time_before_closing} {unit} before closing")
    else:
        ax.set_title("Average price of product")
    ax.set_xlabel("Product start time")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.figure(figsize=(200, 150))
    plt.savefig(f'{plot_path}/average_price_{start_date}-{end_date}'
                f'-m{min_time_before_closing}-M{min_time_before_closing}-{unit}.png')


def get_day_ahead_as_intra_day_prediction_accuracy(box_plot=False,
                                                   percentage=False,
                                                   remove_outliers=False,
                                                   interval='H',
                                                   max_time_before_closing=None,
                                                   min_time_before_closing=None,
                                                   unit=None,
                                                   start_date='2021-11-09',
                                                   end_date='2022-03-23'):
    """
    Calculates standard deviation for intra-day prices by day
    TODO: add option to input max/min time before closing to get_pct_change, maybe even add param to this function to group by something other than hour / not group
    :param box_plot: to plot the box plot or not
    :param percentage: to use percentage or price difference
    :param remove_outliers: if outliers should be removed before calculating accuracy (according to interquartile range)
    :param interval: time interval on which to group intra-day prices
    :param max_time_before_closing: only trades after maximum till product closes
    :param min_time_before_closing: only trades before minimum time till product closes
    :param unit: pandas unit for time before closing
        Possible values:
        ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
        ‘days’ or ‘day’
        ‘hours’, ‘hour’, ‘hr’, or ‘h’
        ‘minutes’, ‘minute’, ‘min’, or ‘m’
        ‘seconds’, ‘second’, or ‘sec’
        ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
        ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
        ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return:
    """
    if percentage:
        name = 'percentage'
        column = 'percentage_change'
        df = get_pct_change_dataframe(start_date, end_date)
    else:
        name = 'difference'
        column = 'price_diff'
        df = get_diff(absolute=False,
                      interval=interval,
                      max_time_before_closing=max_time_before_closing,
                      min_time_before_closing=min_time_before_closing,
                      unit=unit,
                      start_date=start_date,
                      end_date=end_date)

    if remove_outliers:
        if box_plot:
            # Quantile box plot
            plt.boxplot(df['percentage_change'])
            plt.title("Before removing outliers")
            plt.savefig(f'{plot_path}/{name}-box_plot_before-{start_date}-{end_date}.png')

        # Calculate limits for  outliers with interquartile range
        # Will be more accurate when we get more data
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # Remove extreme outliers
        df = df[(df[column] >= lower_limit)
                & (df[column] <= upper_limit)]

        if box_plot:
            # Quantile box plot
            plt.boxplot(df[column])
            plt.title("After removing outliers")
            plt.savefig(f'{plot_path}/{name}-box_plot_after-{start_date}-{end_date}.png')

    df.plot(kind='hist', bins=100, color='blue', edgecolor='black')
    plt.savefig(f'{plot_path}/{name}-_change_hist.png')

    sns.distplot(df[column], hist=True, kde=True,
                 bins=100, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2})
    plt.savefig(f'{plot_path}/{name}_dist.png')

    # var and std by hour of day
    if percentage:
        times = pd.to_datetime(df['trd_delivery_time_start'])
        return df.groupby(times.dt.hour)[column].agg(['var', 'std']).rename(index={'trd_delivery_time_start': 'hour'})
    return df.groupby(df.index.hour)[column].agg(['var', 'std']).rename(index={'trd_delivery_time_start': 'hour'})


def get_increase_decrease(start_date='2021-11-09',
                          end_date='2022-03-23'):
    """
    Calculates times when price decreased, increased or remained the same
    TODO: create a separate function for value counts, this one should just return labeled data
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: pandas dataframe with value counts for positive, negative and neutral changes
    """
    df_intra_day = get_intra_day_min_max_mean(interval='H', on='trd_delivery_time_start', start_date=start_date,
                                              end_date=end_date, max_time_before_closing=1, unit='H')

    df = get_transformed_day_ahead(start_date=start_date, end_date=end_date)

    conditions = [
        (df['trd_price'] - df_intra_day['trd_price_mean']) > 0,
        (df['trd_price'] - df_intra_day['trd_price_mean']) == 0,
        (df['trd_price'] - df_intra_day['trd_price_mean']) < 0,
    ]

    values = [
        'Positive',
        'Neutral',
        'Negative'
    ]
    df['label'] = np.select(conditions, values, default='Undefined')
    return df['label'].value_counts()


def get_wind_correlation(time,
                         start_date='2021-11-09',
                         end_date='2022-03-23'):
    """
    Calculates correlation between wind forecast for each region and change in price
    :param time: time at which forecast happened
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe of co
    """
    df_wind = get_wind_forecast()
    df_wind = df_wind[df_wind['Time'] == time]
    df_wind['Date'] = pd.to_datetime(df_wind['Date'], format='%Y%m%d')
    # Fix 24th hour of day to actually be the 0th hour of the next day
    df_wind.insert(2, '0', df_wind['24'].shift(12))
    df_wind.drop(columns=['Time', '18', '24', '30'], inplace=True)
    df_wind.dropna(inplace=True)

    # Creates a dataframe for each region in appropriate shape
    wind_dfs = []
    for region in df_wind['Region'].unique():
        df = df_wind[df_wind['Region'] == region]
        df = df.groupby('Date').apply(lambda row: pd.DataFrame(
            {'timestamp': pd.date_range(row['Date'].iloc[0],
                                        periods=len(df.columns[2:]), freq='12H'),
             region: np.repeat(np.array([row[col].iloc[0] for col in df.columns[2:]]), 1)}))
        df.reset_index(inplace=True, drop=True)
        wind_dfs.append(df)

    # Concatenates dataframes for each region into a single dataframe
    df_wind = pd.concat(wind_dfs, axis=1)
    df_wind = df_wind.loc[:, ~df_wind.columns.duplicated()]
    df_wind['wind_mean'] = df_wind.iloc[:, 1:12].mean(axis=1)
    df_wind.rename(columns={"timestamp": "trd_delivery_time_start"}, inplace=True)

    # TODO: Might make sense to group by interval of 12 hours,
    #  currently you're looking at forecast at 0 for 12 and price at 0, get pct change gonna need more params
    # df = df_wind.merge(get_pct_change_dataframe(start_date, end_date), on='trd_delivery_time_start')
    df = df_wind.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date), on='trd_delivery_time_start')
    df.dropna(inplace=True)
    pd.set_option('display.expand_frame_repr', False)

    # print(f"Correlation between forecast wind data at hour {time} for 12 hours ahead and price change:")
    return df[df.columns[1:14]].apply(lambda x: x.corr(df['price_diff'], method='pearson'))


def plot_seasonality():
    """
    Plots observed, trend, seasonal and residual time series
    """
    df_intra_day = get_intra_day_min_max_mean(on='trd_delivery_time_start', interval='H')
    df_intra_day['trd_delivery_time_start'] = pd.to_datetime(df_intra_day['trd_delivery_time_start'])
    df_intra_day = df_intra_day.set_index('trd_delivery_time_start')

    def plot(result, title):
        fig = result.plot()
        ax = fig.gca()
        ax.xaxis_date()
        fig.autofmt_xdate()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{plot_path}/seasonality-{(title.lower())}.png')

    plot(seasonal_decompose(df_intra_day['trd_price_mean'], period=1), 'Hourly')
    plot(seasonal_decompose(df_intra_day['trd_price_mean'], period=24), 'Daily')
    plot(seasonal_decompose(df_intra_day['trd_price_mean'], period=168), 'Weekly')
    plot(seasonal_decompose(df_intra_day['trd_price_mean'], period=730), 'Monthly')


def plot_auto_correlation():
    """
    Plots auto-correlation and partial auto-correlation
    """
    # Auto-correlation plot
    df_intra_day = get_intra_day_min_max_mean(on='trd_delivery_time_start', interval='H')
    df_intra_day['trd_delivery_time_start'] = pd.to_datetime(df_intra_day['trd_delivery_time_start'])
    df_intra_day = df_intra_day.set_index('trd_delivery_time_start')
    plot_acf(df_intra_day['trd_price_mean'])
    plt.savefig(f'{plot_path}/auto_correlation.png')

    # Partial auto-correlation plot
    plot_pacf(df_intra_day['trd_price_mean'], method='ywm')
    plt.savefig(f'{plot_path}/partial_auto_correlation.png')


def plot_diff(start_date='2021-11-09',
              end_date='2022-03-23'):
    """
    Plots density plots and histograms for price differences between intra-day and day-ahead data
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    """

    def plot_df(df):
        df.plot(kind='hist', bins=100, color='blue', edgecolor='black')
        plt.show()

        # Density Plot and Histogram of all differences
        sns.distplot(df['price_diff'], hist=True, kde=True,
                     bins=100, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2})
        plt.show()

    # plot_df(get_diff(absolute=False, start_date=start_date, end_date=end_date, time_before_closing=30, unit='minutes'))
    plot_df(get_diff(absolute=False, start_date=start_date, end_date=end_date, interval='H'))


def plot_std_of_diff_by_day():
    """
    Plots standard deviation between intra-day and day-ahead prices
    TODO: currently it's for all products in a day, add option to pass max/min time before closing
    """

    def plot_df(df):
        df = df.groupby(df.index.day_name())['price_diff'].agg(['var', 'std'])
        df.rename(index={'trd_delivery_time_start': 'hour'})
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['Day'] = pd.Categorical(df.index, categories=cats, ordered=True)
        df = df.sort_values('Day')
        df = df.set_index('Day').reindex(cats)
        df['std'].plot(kind="bar")
        plt.tight_layout()
        plt.savefig(f'{plot_path}/std_of_diff_by_day.png')

    plot_df(get_diff())


def granger_causality(start_date='2021-11-09',
                      end_date='2022-03-23'):
    """
    Checks if intra-day price the day before gra
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: pandas dataframe with p-values for granger causality for all hour combinations
    """
    df = get_intra_day_by_hours(start_date=start_date, end_date=end_date)
    stationary = False
    max_diff = 0
    # Make time series stationary
    while not stationary:
        stationary = True
        for hour in range(24):
            # Augmented Dickey-Fuller test
            diff = pmd.arima.ndiffs(df[hour], test="adf")
            if diff > 0:
                stationary = False
            if diff > max_diff:
                max_diff = diff
            # Kwiatkowski–Phillips–Schmidt–Shin test
            diff = pmd.arima.ndiffs(df[hour], test="kpss")
            if diff > 0:
                stationary = False
            if diff > max_diff:
                max_diff = diff
        while max_diff > 0:
            df = df.diff().dropna()
            max_diff -= 1

    return get_grangers_causation_matrix(df, variables=df.columns)


# Taken from
# https://stackoverflow.com/questions/58005681/is-it-possible-to-run-a-vector-autoregression-analysis-on-a-large-gdp-data-with
# Max lag is 1 as it was only necessary to difference once
def get_grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, max_lag=1):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    print(data)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=max_lag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(max_lag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df


if __name__ == "__main__":
    plot_price_at_given_time()
    plot_average_price_of_product()
    plot_average_price_of_product(max_time_before_closing=30, unit='minutes')
    plot_average_price_of_product(max_time_before_closing=1, unit='hours')
    df_acc = get_day_ahead_as_intra_day_prediction_accuracy()
    df_acc.to_csv(f'{csv_path}/accuracy.csv')
    df_inc_dec = get_increase_decrease()
    df_inc_dec.to_csv(f'{csv_path}/increase_decrease.csv')
    df_wind_corr_0 = get_wind_correlation(time=0)
    df_wind_corr_0.to_csv(f'{csv_path}/wind_correlation_0.csv')
    df_wind_corr_12 = get_wind_correlation(time=12)
    df_wind_corr_12.to_csv(f'{csv_path}/wind_correlation_12.csv')
    plot_seasonality()
    plot_diff()
    plot_std_of_diff_by_day()
    df_granger_causation_matrix = granger_causality()
    df_granger_causation_matrix.to_csv(f'{csv_path}/granger_causality.csv')
