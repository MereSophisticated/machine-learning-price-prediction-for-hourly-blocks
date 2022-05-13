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


# Might make sense to do an analysis how price variance changes depending on how much time is left till closing time
# TODO: save plots in files / labeli s slovenščini


def plot_price_at_given_time(start_date='2022-02-03', end_date='2022-03-03'):
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
    plt.show()


def plot_average_price_of_product(start_date='2022-02-03', end_date='2022-02-03', time_before_closing=None, unit=None):
    df_intra_day = get_intra_day_min_max_mean(interval='1H',
                                              on='trd_delivery_time_start',
                                              time_before_closing=time_before_closing,
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
    if time_before_closing:
        ax.set_title(f"Average price of product {time_before_closing} {unit} before closing")
    else:
        ax.set_title("Average price of product")
    ax.set_xlabel("Product start time")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.figure(figsize=(200, 150))
    plt.show()


def day_ahead_as_intra_day_prediction_accuracy(start_date='2022-02-02', end_date='2022-03-23', box_plot=False):
    df = get_pct_change_dataframe(start_date, end_date)

    if box_plot:
        # Quantile box plot
        plt.boxplot(df['percentage_change'])
        plt.title("Before removing outliers")
        plt.show()

    # Calculate limits for  outliers with interquartile range
    # Will be more accurate when we get more data
    q1 = df['percentage_change'].quantile(0.25)
    q3 = df['percentage_change'].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    # Remove extreme outliers
    df = df[(df['percentage_change'] >= lower_limit)
            & (df['percentage_change'] <= upper_limit)]

    if box_plot:
        # Quantile box plot
        plt.boxplot(df['percentage_change'])
        plt.title("After removing outliers")
        plt.show()

    df.plot(kind='hist', bins=100, color='blue', edgecolor='black')
    plt.show()

    sns.distplot(df['percentage_change'], hist=True, kde=True,
                 bins=100, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2})
    plt.show()

    # var and std by hour of day
    times = pd.to_datetime(df['trd_delivery_time_start'])
    print(df.groupby(times.dt.hour)['percentage_change'].agg(['var', 'std']))


def increase_decrease_analysis(start_date='2022-02-02', end_date='2022-03-23'):
    df_intra_day = get_intra_day_min_max_mean(interval='H', on='trd_delivery_time_start', start_date=start_date,
                                              end_date=end_date, time_before_closing=1, unit='H')

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
    print(df['label'].value_counts())


def wind_analysis(time, start_date='2022-02-02', end_date='2022-03-23'):
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
    df = df_wind.merge(get_pct_change_dataframe(start_date, end_date), on='trd_delivery_time_start')
    df.dropna(inplace=True)
    pd.set_option('display.expand_frame_repr', False)

    print(f"Correlation between forecast wind data at hour {time} for 12 hours ahead and price change:")
    print(df[df.columns[1:14]].apply(lambda x: x.corr(df['percentage_change'], method='pearson')), end=3 * '\n')


def stats_plots():
    df_intra_day = get_intra_day_min_max_mean(on='trd_delivery_time_start', interval='H')
    df_intra_day['trd_delivery_time_start'] = pd.to_datetime(df_intra_day['trd_delivery_time_start'])
    df_intra_day = df_intra_day.set_index('trd_delivery_time_start')

    # Hourly trend, seasonality and residuals
    result = seasonal_decompose(df_intra_day['trd_price_mean'], period=1)
    fig = result.plot()
    ax = fig.gca()
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.show()

    # Daily trend, seasonality and residuals
    result = seasonal_decompose(df_intra_day['trd_price_mean'], period=24)
    fig = result.plot()
    ax = fig.gca()
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.show()

    # Weekly trend, seasonality and residuals
    result = seasonal_decompose(df_intra_day['trd_price_mean'], period=168)
    fig = result.plot()
    ax = fig.gca()
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.show()

    # TODO: # Monthly trend, seasonality and residuals

    # Autocorrelation plot
    ax = plot_acf(df_intra_day['trd_price_mean'])
    plt.show()

    # Partial autocorrelation plot
    plot_pacf(df_intra_day['trd_price_mean'], method='ywm')
    plt.show()


def plot_diff_and_abs_diff(start_date='2022-02-02', end_date='2022-03-24'):
    def plot_df(df):
        df.plot(kind='hist', bins=100, color='blue', edgecolor='black')
        plt.show()

        # Density Plot and Histogram of all differences
        sns.distplot(df['price_diff'], hist=True, kde=True,
                     bins=100, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2})
        plt.show()

    plot_df(get_diff(start_date=start_date, end_date=end_date))
    plot_df(get_diff(absolute=False, start_date=start_date, end_date=end_date))


def var_and_std_of_diff_by_day():
    def plot_df(df):
        df = df.groupby(df.index.day_name())['price_diff'].agg(['var', 'std'])
        df.rename(index={'trd_delivery_time_start': 'hour'})
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['Day'] = pd.Categorical(df.index, categories=cats, ordered=True)
        df = df.sort_values('Day')
        df = df.set_index('Day').reindex(cats)
        df['std'].plot(kind="bar")
        plt.tight_layout()
        plt.show()

    plot_df(get_diff())
    plot_df(get_diff(absolute=False))


def granger_causality(start_date='2022-02-02', end_date='2022-03-24'):
    df = get_intra_day_by_hours(start_date=start_date, end_date=end_date)
    stationary = False
    max_diff = 0
    while not stationary:
        stationary = True
        for hour in range(24):
            diff = pmd.arima.ndiffs(df[hour], test="adf")
            if diff > 0:
                stationary = False
            if diff > max_diff:
                max_diff = diff
            diff = pmd.arima.ndiffs(df[hour], test="kpss")
            if diff > 0:
                stationary = False
            if diff > max_diff:
                max_diff = diff
        while max_diff > 0:
            df = df.diff().dropna()
            max_diff -= 1

    return get_grangers_causation_matrix(df, variables=df.columns)


def adf_pmd(x):
    adf_test = pmd.arima.stationarity.ADFTest(alpha=0.05)
    res = adf_test.should_diff(x)
    conclusion = "non-stationary" if res[0] > 0.05 else "stationary"
    resdict = {"should we difference? ": res[1], "p-value ": res[0], "conclusion": conclusion}
    n_adf = pmd.arima.ndiffs(x, test="adf")
    return resdict, n_adf


def kpss_pmd(x):
    kpss_test = pmd.arima.stationarity.KPSSTest(alpha=0.05)
    res = kpss_test.should_diff(x)
    conclusion = "not stationary" if res[0] <= 0.05 else "stationary"
    resdict = {"should we difference? ":res[1], "p-value ":res[0], "conclusion":conclusion}
    n_kpss = pmd.arima.ndiffs(x, test="kpss")
    return resdict, n_kpss


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
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df


if __name__ == "__main__":
    plot_price_at_given_time()
    plot_average_price_of_product()
    plot_average_price_of_product(time_before_closing=30, unit='minutes')
    plot_average_price_of_product(time_before_closing=1, unit='hours')
    day_ahead_as_intra_day_prediction_accuracy()
    increase_decrease_analysis()
    wind_analysis(time=0)
    wind_analysis(time=12)
    stats_plots()
    plot_diff_and_abs_diff()
    var_and_std_of_diff_by_day()
    granger_causation_matrix = granger_causality()
    print(granger_causation_matrix)
