import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Global intra day and day ahead dataframes for functions to use
files = ['trds_1.parquet',
         'trds_2.parquet',
         'trds_3.parquet',
         'trds_4.parquet',
         'trds_5.parquet',
         'trds_6.parquet',
         ]
tables = []
for file in files:
    tables.append(pq.read_table(Path(__file__).parent / f'../data/{file}',
                                columns=['trd_trade_id', 'trd_execution_time', 'trd_venue',
                                         'trd_buy_delivery_area', 'trd_sell_delivery_area',
                                         'trd_tariff',
                                         'trd_sequence_name', 'trd_product', 'trd_price',
                                         'trd_delivery_time_name', 'trd_delivery_time_start',
                                         'trd_delivery_time_end']))
table = pa.concat_tables(tables)
df_intra_day = table.to_pandas()
df_intra_day = df_intra_day.sort_values(by=['trd_execution_time', 'trd_buy_delivery_area', 'trd_sell_delivery_area'])

df_day_ahead = pd.read_csv(Path(__file__).parent / '../data/DE_DA_prices.csv',
                           converters={'timestamp': lambda t: pd.Timestamp(t).timestamp()})

# Global wind dataframes for functions to use
df_wind_historic = pd.read_csv('../data/wind_data_average.csv')
df_wind_forecast = pd.read_csv('../data/forecast_wind_data_average.csv')


def get_intra_day_data_for_region(region: str):
    """
    :param region: (sub)string that is contained in the region name
    :return: intra-day dataframe for provided region
    """
    region = region.upper()
    return df_intra_day[df_intra_day["trd_buy_delivery_area"].str.contains(region) &
                        df_intra_day["trd_sell_delivery_area"].str.contains(region)].copy()


def get_intra_day(max_time_before_closing=None,
                  min_time_before_closing=None,
                  unit=None,
                  region="GERMANY",
                  start_date=None,
                  end_date=None):
    """
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
    :param region: (sub)string that is contained in the region name
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: pandas dataframe for intra-day prices
    """
    df = get_intra_day_data_for_region(region)
    # Filter to trades that happened between start_date and end_date
    if start_date and end_date:
        df = df[(df['trd_delivery_time_start'] >= f'{start_date} 00:00:00') &
                (df['trd_delivery_time_start'] <= f'{end_date} 23:00:00')]
    # Filter to only trades that happened at most time_before_closing-units before closing
    if max_time_before_closing and min_time_before_closing and unit:
        df['diff'] = (df['trd_delivery_time_start']
                      - df['trd_execution_time'])
        df = df[(df['diff'] <= pd.Timedelta(max_time_before_closing, unit=unit)) &
                (df['diff'] >= pd.Timedelta(min_time_before_closing, unit=unit))]
    if max_time_before_closing and unit:
        df['diff'] = (df['trd_delivery_time_start']
                      - df['trd_execution_time'])
        df = df[df['diff'] <= pd.Timedelta(max_time_before_closing, unit=unit)]
    elif min_time_before_closing and unit:
        df['diff'] = (df['trd_delivery_time_start']
                      - df['trd_execution_time'])
        df = df[df['diff'] >= pd.Timedelta(min_time_before_closing, unit=unit)]
    return df


def get_day_ahead_data():
    """
    :return: pandas dataframe of day-ahead data
    """
    return df_day_ahead.copy()


def get_wind_historic():
    """
    :return: pandas dataframe of historic (actual) wind data
    """
    return df_wind_historic.copy()


def get_wind_forecast():
    """
    :return: pandas dataframe of forecast wind data
    """
    return df_wind_forecast.copy()


def get_intra_day_min_max_mean(interval='15min',
                               on='trd_execution_time',
                               max_time_before_closing=None,
                               min_time_before_closing=None,
                               unit=None,
                               start_date=None,
                               end_date=None):
    """
    :param interval: interval to resample on
    :param on: column to resample on
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
    :return: day-ahead dataframe with tdr_price_mean, trd_price_min and trd_price_max
    """
    df = get_intra_day(start_date=start_date, end_date=end_date,
                       max_time_before_closing=max_time_before_closing,
                       min_time_before_closing=min_time_before_closing, unit=unit)
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))
    df.reset_index(inplace=True)
    return df


def get_transformed_day_ahead(start_date=None,
                              end_date=None):
    """
    Transforms columnar DateCET and hour data into a single timestamp column and the corresponding price
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: transformed day ahead dataframe
    """
    df = get_day_ahead_data()
    df = df.groupby('DateCET').apply(lambda row: pd.DataFrame(
        {'trd_delivery_time_start': pd.date_range(row['DateCET'].iloc[0],
                                                  periods=len(df.columns[1:]), freq='1H'),
         'trd_price': np.repeat(np.array([row[col].iloc[0] for col in df.columns[1:]]), 1)}))
    # Filter to trades that happened between start_date and end_date
    if start_date and end_date:
        df = df[(df['trd_delivery_time_start'] >= f'{start_date} 00:00:00') &
                (df['trd_delivery_time_start'] <= f'{end_date} 23:00:00')]
    df.reset_index(inplace=True, drop=True)
    return df


def get_diff(absolute=True,
             interval='H',
             max_time_before_closing=None,
             min_time_before_closing=None,
             unit=None,
             start_date='2021-11-09',
             end_date='2022-03-23'):
    """
    :param absolute: to calculate absolute diff or not
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
    :return: dataframe with price differences between day-ahead and intra-day
    """
    df_id = get_intra_day_min_max_mean(interval=interval, on='trd_delivery_time_start', start_date=start_date,
                                       end_date=end_date,
                                       max_time_before_closing=max_time_before_closing,
                                       min_time_before_closing=min_time_before_closing, unit=unit) \
        .drop(columns=['trd_price_min', 'trd_price_max']).rename(columns={"trd_price_mean": "trd_price"})
    df_da = get_transformed_day_ahead(start_date=start_date, end_date=end_date)

    df_id.set_index('trd_delivery_time_start', inplace=True)
    df_da.set_index('trd_delivery_time_start', inplace=True)

    if absolute:
        df = (df_id - df_da).abs()
    else:
        df = df_id - df_da
    df.dropna(inplace=True)
    df.rename(columns={"trd_price": "price_diff"}, inplace=True)

    return df


def get_pct_change_dataframe(start_date='2021-11-09',
                             end_date='2022-03-23'):
    """
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe with percentage change between day-ahead and intra-day
    """
    df = get_transformed_day_ahead(start_date=start_date, end_date=end_date)
    df['percentage_change'] = ((get_intra_day_min_max_mean(interval='H',
                                                           on='trd_delivery_time_start',
                                                           max_time_before_closing=60,
                                                           unit='minutes',
                                                           start_date=start_date,
                                                           end_date=end_date)['trd_price_mean']
                                - df['trd_price'].values) /
                               df[
                                   'trd_price'].values) * 100
    df.drop(columns='trd_price', inplace=True)

    # Drop any inf values that were created due to day_ahead price of 0.0 (corner case)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def get_intra_day_by_hours(start_date='2021-11-09',
                           end_date='2022-03-23'):
    """
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :return: dataframe of intra-day prices split into columns by hours
    """
    df = get_intra_day_min_max_mean(on='trd_delivery_time_start', interval='H',
                                       start_date=start_date, end_date=end_date)
    by_hours = []

    for hour in range(24):
        df_hour = df[df['trd_delivery_time_start'].dt.hour == hour]['trd_price_mean'] \
            .rename(hour)
        by_hours.append(df_hour.reset_index()[hour])
    df = pd.concat(by_hours, axis=1).dropna()
    return df


def get_intra_day_mean(interval='H',
                       on='trd_delivery_time_start',
                       max_time_before_closing=30,
                       min_time_before_closing=None,
                       unit='minutes',
                       start_date=None,
                       end_date=None):
    """
    :param interval: time interval on which to group intra-day prices
    :param on: column to aggregate
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
    :return: dataframe of intra-day price mean (trd_price_mean)
    """
    df = get_intra_day(start_date=start_date, end_date=end_date,
                       max_time_before_closing=max_time_before_closing,
                       min_time_before_closing=min_time_before_closing, unit=unit)
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean))
    df.reset_index(inplace=True)
    return df


def get_std_by_day(absolute=False,
                   interval='H',
                   max_time_before_closing=None,
                   min_time_before_closing=None,
                   unit='hours',
                   start_date='2021-11-09',
                   end_date='2022-03-23'):
    """

    :param absolute: to calculate absolute diff or not
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
    :return: dataframe with standard deviation between day-ahead and intra-day by day
    """

    df = get_diff(absolute=absolute,
                  interval=interval,
                  max_time_before_closing=max_time_before_closing,
                  min_time_before_closing=min_time_before_closing,
                  unit=unit,
                  start_date=start_date, end_date=end_date)
    df['std'] = df.groupby(df.index.day_name())['price_diff'].transform('std')
    return df
