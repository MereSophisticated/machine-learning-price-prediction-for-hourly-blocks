import pandas as pd

# Global intra day and day ahead dataframes for functions to use
df_intra_day = pd.read_parquet('../data/trds.parquet', engine='pyarrow')

def get_data(region: str):
    region = region.upper()
    frames = []
    for i in range(1,7):
        frames.append(pd.read_parquet('../data/ID_DE_TRADES/trds_{}.parquet'.format(i), engine='pyarrow'))
    result = pd.concat(frames)
    return result[result["trd_buy_delivery_area"].str.contains(region) & result["trd_sell_delivery_area"].str.contains(region)]

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
                        df_intra_day["trd_sell_delivery_area"].str.contains(region)]


def get_day_ahead_data():
    return df_day_ahead


def get_wind_data():
    df_wind_historic = pd.read_csv('../data/wind_data_2022.csv')
    return df_wind_historic

def get_wind_forecast():
    df_wind_forecast = pd.read_csv('../data/forecast_wind_data_average.csv')
    return df_wind_forecast


def get_sun_data():
    df_sun_historic = pd.read_csv('../data/data_sun_2022.csv')
    return df_sun_historic

get_wind_data()
get_sun_data()