import math

import numpy as np
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# Get data
from src.data_analysis.data_retrieval import get_prepared_data

X, y, X_train, X_test, y_train, y_test = get_prepared_data()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0, n_estimators=1000)
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
shap_values = explainer.shap_values(X_train, check_additivity=False)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_train, show=False)
# Fix high/low bar display in shap (another option is to set color_bar=False in summary plot and call plt.colorbar()
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.tight_layout()
plt.show()