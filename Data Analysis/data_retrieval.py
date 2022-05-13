import numpy as np
import pandas as pd

# Global intra day and day ahead dataframes for functions to use
df_intra_day = pd.read_parquet('../data/trds.parquet', engine='pyarrow')

type_dict = {
    0: 'datetime'
}
df_day_ahead = pd.read_csv('../data/DE_DA_prices.csv',
                           converters={'timestamp': lambda t: pd.Timestamp(t).timestamp()})

# Drop useless attributes
df_intra_day.drop(['trd_deleted', 'trd_quantity', 'trd_lot'], axis=1, inplace=True)
# Diff between dropping duplicates by 'trd_product' as well is only 29 rows (271488 - 2714859 = 29)
# As it is such a small number, it might make sense to just ignore those (it's the same prices anyway)
df_intra_day = df_intra_day.drop_duplicates(
    subset=['trd_execution_time', 'trd_buy_delivery_area', 'trd_sell_delivery_area'],
    keep='first'
).reset_index(drop=True)

df_intra_day = df_intra_day.sort_values(by=['trd_execution_time', 'trd_buy_delivery_area', 'trd_sell_delivery_area'])


def get_intra_day_data_for_region(region: str):
    region = region.upper()
    return df_intra_day[df_intra_day["trd_buy_delivery_area"].str.contains(region) &
                        df_intra_day["trd_sell_delivery_area"].str.contains(region)].copy()


def get_intra_day(start_date=None, end_date=None, time_before_closing=None, unit=None):
    region = "GERMANY"
    df = df_intra_day[df_intra_day["trd_buy_delivery_area"].str.contains(region) &
                      df_intra_day["trd_sell_delivery_area"].str.contains(region)].copy()
    # Filter to trades that happened between start_date and end_date
    if start_date and end_date:
        df = df[(df['trd_delivery_time_start'] >= f'{start_date} 00:00:00') &
                (df['trd_delivery_time_start'] <= f'{end_date} 23:00:00')]
    # Filter to only trades that happened at most time_before_closing-units before closing
    if time_before_closing and unit:
        df['diff'] = (df['trd_delivery_time_start']
                      - df['trd_execution_time'])
        df = df[df['diff'] <= pd.Timedelta(time_before_closing, unit=unit)]
    return df


def get_day_ahead_data():
    return df_day_ahead.copy()


def get_wind_data():
    df_wind_historic = pd.read_csv('../data/wind_data_average.csv')
    return df_wind_historic


def get_wind_forecast():
    df_wind_forecast = pd.read_csv('../data/forecast_wind_data_average.csv')
    return df_wind_forecast


def get_intra_day_min_max_mean(interval='15min', on='trd_execution_time',
                               start_date=None, end_date=None, time_before_closing=None, unit=None):
    """
    :param interval: interval to resample on
    :param on: column to resample on
    :param start_date: filter trades to those that happened on or after start_date
    :param end_date: filter trades to those that happened on or before end_date
    :param time_before_closing: time in unit before closing
    :param unit: pandas unit for time before closing
    :return: day-ahead dataframe with tdr_price_mean, trd_price_min and trd_price_max
    """
    df = get_intra_day(start_date=start_date, end_date=end_date,
                       time_before_closing=time_before_closing, unit=unit)
    df = df.resample(interval, on=on).agg(trd_price_mean=('trd_price', np.mean),
                                          trd_price_min=('trd_price', np.min),
                                          trd_price_max=('trd_price', np.max))
    df.reset_index(inplace=True)
    return df


def get_transformed_day_ahead(start_date=None, end_date=None):
    """
    Transforms columnar DateCET and hour data into a single timestamp column and the corresponding price
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


def get_diff(absolute=True, start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = get_intra_day_min_max_mean(interval='H', on='trd_delivery_time_start', start_date=start_date,
                                              end_date=end_date, time_before_closing=30, unit='minutes')\
        .drop(columns=['trd_price_min', 'trd_price_max']).rename(columns={"trd_price_mean": "trd_price"})

    df_day_ahead = get_transformed_day_ahead(start_date=start_date, end_date=end_date)

    df_intra_day.set_index('trd_delivery_time_start', inplace=True)
    df_day_ahead.set_index('trd_delivery_time_start', inplace=True)

    if absolute:
        df = (df_intra_day - df_day_ahead).abs()
    else:
        df = df_intra_day - df_day_ahead
    df.dropna(inplace=True)
    df.rename(columns={"trd_price": "price_diff"}, inplace=True)

    return df


def get_pct_change_dataframe(start_date='2022-02-02', end_date='2022-03-23'):
    df = get_transformed_day_ahead(start_date=start_date, end_date=end_date)
    df['percentage_change'] = ((get_intra_day_min_max_mean(interval='H',
                                                           on='trd_delivery_time_start',
                                                           time_before_closing=60,
                                                           unit='minutes',
                                                           start_date=start_date,
                                                           end_date=end_date)['trd_price_mean']
                                - df['trd_price'].values) /
                               df[
                                   'trd_price'].values) * 100
    df.drop(columns='trd_price', inplace=True)

    # Drop any inf values that were created due to day_ahead price of 0.0 (corner case)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def get_intra_day_by_hours(start_date='2022-02-02', end_date='2022-03-24'):
    df_intra_day = get_intra_day_min_max_mean(on='trd_delivery_time_start', interval='H',
                                              start_date=start_date, end_date=end_date)
    by_hours = []

    for hour in range(24):
        df_hour = df_intra_day[df_intra_day['trd_delivery_time_start'].dt.hour == hour]['trd_price_mean']\
                    .rename(hour)
        by_hours.append(df_hour.reset_index()[hour])
    df = pd.concat(by_hours, axis=1).dropna()
    return df
