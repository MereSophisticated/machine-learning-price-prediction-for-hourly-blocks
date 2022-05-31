import numpy as np
import pandas as pd
import pmdarima as pmd
import seaborn as sns
import datetime as dt
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests

from data_retrieval import get_wind_forecast, \
    get_transformed_day_ahead, get_intra_day_min_max_mean, get_diff, get_pct_change_dataframe, get_intra_day_by_hours, \
    get_wind, get_solar, get_residual_load, get_wind_diff, get_solar_diff, get_residual_load_diff, \
    get_wind_deltas_previous_24h, get_solar_deltas_previous_24h, get_residual_deltas_previous_24h

plot_path = "plots"
csv_path = "csv"


def plot_price_at_given_time(start_date='2021-11-09',
                             end_date='2022-03-23'):
    """
    Plots intra-day average, min/max area and day-ahead price
    :param start_date: start date of plot (x-axis)
    :param end_date: end date of plot (x-axis)
    """
    df_intra_day = get_intra_day_min_max_mean(start_date=start_date, end_date=end_date, interval='H')
    print(df_intra_day)

    ax = df_intra_day.plot(x='trd_execution_time', y='trd_price_mean')
    plt.fill_between(df_intra_day['trd_execution_time'].dt.to_pydatetime(),
                     df_intra_day.trd_price_min,
                     df_intra_day.trd_price_max,
                     facecolor='lightblue', alpha=0.6, interpolate=True)

    df_day_ahead = get_transformed_day_ahead(start_date=start_date, end_date=end_date)
    df_day_ahead.plot(x='trd_delivery_time_start', y='trd_price', use_index=True, ax=ax)

    ax.legend(["Povprečje znotraj dnevnega trga", "Min/max znotraj dnevnega trga", "Dnevni trg"])
    ax.set_xlabel("Čas")
    ax.set_ylabel("Cena")
    plt.tight_layout()
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

    ax.legend(["Povprečje znotraj dnevnega trga", "Min/max znotraj dnevnega trga", "Dnevni trg"])
    ax.set_xlabel("Čas začetka produkta")
    ax.set_ylabel("Cena")
    plt.tight_layout()
    plt.savefig(f'{plot_path}/average_price_{start_date}-{end_date}'
                f'-m{min_time_before_closing}-M{max_time_before_closing}-{unit}.png')


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
        df = get_pct_change_dataframe(interval=interval,
                                      max_time_before_closing=max_time_before_closing,
                                      min_time_before_closing=min_time_before_closing,
                                      unit=unit,
                                      start_date=start_date,
                                      end_date=end_date)
    else:
        name = 'difference'
        column = 'price_diff'
        df = get_diff(absolute=False,
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
    plt.savefig(f'{plot_path}/{name}_change_hist.png')

    ax = sns.displot(data=df, x=column, kde=True)
    ax.set(xlabel='Razlika med ceno na dnevnem in znotraj dnevnem trgu', ylabel='Število')
    plt.savefig(f'{plot_path}/{name}_dist.png')

    # var and std by hour of day
    if percentage:
        times = pd.to_datetime(df['trd_delivery_time_start'])
        df = df.groupby(times.dt.hour)[column].agg(['var', 'std']).rename(index={'trd_delivery_time_start': 'hour'})
    else:
        df = df.groupby(df.index.hour)[column].agg(['var', 'std']).rename(index={'trd_delivery_time_start': 'hour'})
    ax = df.plot(kind='bar', y='std')
    ax.set(xlabel="Ura", ylabel='Razlika med ceno na dnevnem in znotraj dnevnem trgu')
    plt.savefig(f'{plot_path}/hour_dist.png')
    return df


def get_increase_decrease(start_date='2021-11-09',
                          end_date='2022-03-23'):
    """
    Calculates times when price decreased, increased or remained the same
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: pandas dataframe with labels for positive, negative and neutral changes
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
    return df


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

    df = df_wind.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date), on='trd_delivery_time_start')
    df.dropna(inplace=True)
    pd.set_option('display.expand_frame_repr', False)

    return df[df.columns[1:14]].apply(lambda x: x.corr(df['price_diff'], method='pearson'))


def get_wind_diff_correlation(start_date='2021-11-09',
                              end_date='2022-03-23'):
    """
    Calculates correlation between the difference of forecast and actual wind power
    and the difference in price between intra-day and day-ahead
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe of correlations
    """
    df = get_wind_diff()
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)

    return df[['ec00_delta_wnd', 'gfs00_delta_wnd', 'icon00_delta_wnd']]. \
        apply(lambda x: x.corr(df['price_diff'], method='pearson'))


def get_wind_deltas_correlation(start_date='2021-11-09',
                                end_date='2022-03-23'):
    df = get_wind_deltas_previous_24h(start_date=start_date, end_date=end_date)
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)
    return df.corr(method='pearson')


def get_solar_diff_correlation(start_date='2021-11-09',
                               end_date='2022-03-23'):
    """
    Calculates correlation between the transformed values for difference of forecast and actual solar power
    and the difference in price between intra-day and day-ahead
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe of correlations
    """
    df = get_solar_diff(start_date=start_date, end_date=end_date)
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)

    return df[['ec00_delta_spv', 'gfs00_delta_spv', 'icon00_delta_spv']]. \
        apply(lambda x: x.corr(df['price_diff'], method='pearson'))


def get_solar_deltas_correlation(start_date='2021-11-09',
                                 end_date='2022-03-23'):
    df = get_solar_deltas_previous_24h(start_date=start_date, end_date=end_date)
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)
    return df.corr(method='pearson')


def get_residual_load_diff_correlation(start_date='2021-11-09',
                                       end_date='2022-03-23'):
    """
    Calculates correlation between the transformed values for difference of forecast and actual solar power
    and the difference in price between intra-day and day-ahead
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe of correlations
    """
    df = get_residual_load_diff(start_date=start_date, end_date=end_date)
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)

    return df[['ec00_delta_rdl', 'gfs00_delta_rdl']].apply(lambda x: x.corr(df['price_diff'], method='pearson'))


def get_residual_load_deltas_correlation(start_date='2021-11-09',
                                         end_date='2022-03-23'):
    df = get_residual_deltas_previous_24h(start_date=start_date, end_date=end_date)
    df = df.merge(get_diff(absolute=False, start_date=start_date, end_date=end_date),
                  left_index=True, right_index=True)

    return df.corr(method='pearson')


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
        plt.savefig(f'{plot_path}/diff_hist.png')

        # Density Plot and Histogram of all differences
        sns.distplot(df['price_diff'], hist=True, kde=True,
                     bins=100, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2})
        plt.savefig(f'{plot_path}/diff_dens.png')

    plot_df(get_diff(absolute=False, start_date=start_date, end_date=end_date))


def plot_std_of_diff_by_day(max_time_before_closing=None,
                            min_time_before_closing=None,
                            unit=None):
    """
    Plots standard deviation between intra-day and day-ahead prices
    :param max_time_before_closing: only trades after maximum till product closes
    :param min_time_before_closing: only trades before minimum time till product closes
    """

    def plot_df(df):
        df = df.groupby(df.index.day_name())['price_diff'].agg(['var', 'std'])
        df.rename(index={'trd_delivery_time_start': 'hour'})
        print(df)
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['Day'] = pd.Categorical(df.index, categories=cats, ordered=True)
        df = df.sort_values('Day')
        df = df.set_index('Day').reindex(cats)
        print(df)
        df['std'].plot(kind="bar")
        bars = ['Ponedeljek', 'Torek', 'Sreda', 'Četrtek', 'Petek', 'Sobota', 'Nedelja']
        y_pos = np.arange(len(bars))
        plt.xticks(y_pos, bars)
        plt.xlabel(None)
        plt.ylabel('Razlika med ceno na dnevnem in znotraj dnevnem trgu')
        plt.tight_layout()
        plt.savefig(f'{plot_path}/std_of_diff_by_day.png')

    plot_df(get_diff(max_time_before_closing=max_time_before_closing,
                     min_time_before_closing=min_time_before_closing,
                     unit=unit))


def granger_causality(start_date='2021-11-09',
                      end_date='2022-03-23'):
    """
    Checks if a product of an hour granger causes subsequent hours
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
    df_acc.to_csv(f'{csv_path}/accuracy_diff.csv')
    df_acc = get_day_ahead_as_intra_day_prediction_accuracy(percentage=True, remove_outliers=True)
    df_acc.to_csv(f'{csv_path}/accuracy_percentage.csv')
    df_inc_dec = get_increase_decrease()
    # saved to a different directory, so it doesn't get committed
    df_inc_dec.to_csv('../../data/increase_decrease.csv')
    df_wind_corr_0 = get_wind_correlation(time=0)
    df_wind_corr_0.to_csv(f'{csv_path}/wind_correlation_0.csv')
    df_wind_corr_12 = get_wind_correlation(time=12)
    df_wind_corr_12.to_csv(f'{csv_path}/wind_correlation_12.csv')
    plot_seasonality()
    plot_diff()
    plot_std_of_diff_by_day()
    df_granger_causation_matrix = granger_causality()
    df_granger_causation_matrix.to_csv(f'{csv_path}/granger_causality.csv')
    df_wind_diff_corr = get_wind_diff_correlation()
    df_wind_diff_corr.to_csv(f'{csv_path}/wind_diff_corr.csv')
    df_solar_diff_corr = get_solar_diff_correlation()
    df_solar_diff_corr.to_csv(f'{csv_path}/solar_diff_corr.csv')
    df_residual_diff_corr = get_residual_load_diff_correlation()
    df_residual_diff_corr.to_csv(f'{csv_path}/residual_diff_corr.csv')
    df_corr_wind_deltas = get_wind_deltas_correlation()
    plt.figure(figsize=(5, 20))
    ax = sns.heatmap(
        df_corr_wind_deltas[['price_diff']].drop('price_diff').sort_values(by=['price_diff'],
                                                                           ascending=False, key=abs),
        annot=True, cmap=sns.diverging_palette(240, 10, as_cmap=True), fmt='.2g', center=0)
    plt.title("Wind")
    plt.tight_layout()
    plt.save(f'{plot_path}/residual_deltas.png')

    df_corr_solar_deltas = get_solar_deltas_correlation()
    plt.figure(figsize=(5, 20))
    ax = sns.heatmap(
        df_corr_solar_deltas[['price_diff']].drop('price_diff').sort_values(by=['price_diff'],
                                                                            ascending=False, key=abs),
        annot=True, cmap=sns.diverging_palette(240, 10, as_cmap=True), fmt='.2g', center=0)
    plt.title("Solar")
    plt.tight_layout()
    plt.save(f'{plot_path}/solar_deltas.png')

    df_corr_residual_deltas = get_residual_load_deltas_correlation()
    plt.figure(figsize=(5, 20))
    ax = sns.heatmap(
        df_corr_residual_deltas[['price_diff']].drop('price_diff').sort_values(by=['price_diff'],
                                                                               ascending=False, key=abs),
        annot=True, cmap=sns.diverging_palette(240, 10, as_cmap=True), fmt='.2g', center=0)
    plt.title("Residual")
    plt.tight_layout()
    plt.save(f'{plot_path}/residual_deltas.png')
