import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data_analysis.data_retrieval import get_prepared_data

X, y, X_train, X_test, y_train, y_test = get_prepared_data()


model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
errors = abs(predictions - y_test)

print(f'Average absolute error:{round(np.mean(errors), 2)}')

mape = np.mean(100 * (errors / y_test))

accuracy = 100 - mape
print(f'MAPE: {mape}')
print(f'MSE: {mean_squared_error(y_test, predictions)}')
print(f'RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}')

explainer = shap.TreeExplainer(model)
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

"""for column in X_train.columns:
    if column != 'trd_price':
        shap.dependence_plot(column, shap_values, X_train, interaction_index="trd_price")
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
#shap.dependence_plot('day', shap_values, X_train, interaction_index="trd_price")
#shap.dependence_plot('weekend', shap_values, X_train, interaction_index="trd_price")


predictions = model.predict(X_test)
errors = abs(predictions - y_test)

print(f'Average absolute error:{round(np.mean(errors), 2)}')

mape = np.mean(100 * (errors / y_test))

accuracy = 100 - mape
print(f'MAPE: {mape}')
print(f'MSE: {mean_squared_error(y_test, predictions)}')
print(f'RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}')

expected_value = explainer.expected_value

if isinstance(expected_value, list):
    expected_value = expected_value#[1]
print(f"Explainer expected value: {expected_value}")

select = range(24)
features = X_test.iloc[select]
features_display = X.loc[features.index]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(features)[1]
    shap_interaction_values = explainer.shap_interaction_values(features)
if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]


shap.decision_plot(explainer.expected_value, shap_values, X)
print(y_test[0:24])
print(predictions[0:24])"""


