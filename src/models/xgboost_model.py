import shap
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import pandas as pd
import numpy as np
import math
# Get data
from src.data_analysis.data_retrieval import get_intra_day_min_max_mean, get_transformed_day_ahead, get_std_by_day

df_intra_day = get_intra_day_min_max_mean(interval='H', on='trd_delivery_time_start',
                                          start_date='2021-11-09', end_date='2022-03-22', max_time_before_closing=30,
                                          unit='minutes').set_index('trd_delivery_time_start')

df_day_ahead = get_transformed_day_ahead(start_date='2021-11-09', end_date='2022-03-22').set_index(
    'trd_delivery_time_start')

print("1 hours")
df_diff = get_std_by_day(min_time_before_closing=1, unit='hours')
print(df_diff)
df = pd.concat([df_intra_day, df_day_ahead, df_diff], axis=1).dropna()
print(df)

base_model = False



def day_time(x):
    if (x > 4) and (x <= 8):
        return 'early_morning'
    elif (x > 8) and (x <= 12):
        return 'morning'
    elif (x > 12) and (x <= 16):
        return 'noon'
    elif (x > 16) and (x <= 20):
        return 'eve'
    elif (x > 20) and (x <= 24):
        return 'night'
    elif (x <= 4):
        return 'late_night'


if base_model:
    X = df[['trd_price']]
    X['day'] = X.index.dayofweek
    X['hour'] = X.index.hour
    y = df['trd_price_mean']
else:
    X = df[['trd_price', 'price_diff', 'std']]
    X['day'] = X.index.dayofweek
    X['hour'] = X.index.hour
    X['weekend'] = X.index.dayofweek > 4
    X['day_night_time'] = X['day'].apply(day_time)
    X['day_night_time'] = pd.factorize(X['day_night_time'])[0]
    y = df['trd_price_mean']

#print("X", X)
#print(X['day_night_time'].unique())
#print("Y", y)

mask = df.index >= '2022-03-21 00:00:00'

X_train, X_test, y_train, y_test = X[~mask], X[mask], y[~mask], y[mask]
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0, n_estimators=1000, learning_rate=0.8)
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
errors = abs(predictions - y_test)

print(f'Average absolute error:{round(np.mean(errors), 2)}')

mape = np.mean(100 * (errors / y_test))

accuracy = 100 - mape
print(f'MAPE: {mape}')
print(f'MSE: {mean_squared_error(y_test, predictions)}')
print(f'RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}')



explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_train, show=False)
# Fix high/low bar display in shap (another option is to set color_bar=False in summary plot and call plt.colorbar()
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.tight_layout()
plt.show()