import math
import time

import numpy as np
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from src.data_analysis.data_retrieval import get_train_test_split

plot_path = 'plots'
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0, n_estimators=100)


def train_and_test_model(X_train, X_test, y_train, y_test, plt_title):
    """
       Train and test the model on the input dataset, then plot feature importance with shap.
       :param X_train: train input dataset
       :param X_test: test input dataset
       :param y_train: train targets
       :param y_test: test targets
       :param plt_title: plot title
       :return: None
       """
    plt_title = "r_xgboost_" + plt_title
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)

    print(f'{plt_title}:')
    print(f'Average absolute error:{round(np.mean(errors), 2)}')
    mape = np.mean(100 * (errors / y_test))
    print(f'MAPE: {mape}')
    print(f'MSE: {mean_squared_error(y_test, predictions)}')
    print(f'RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train, check_additivity=False)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/{plt_title}_imp.png')
    plt.clf()

    shap.summary_plot(shap_values, X_train, show=False)
    # Fix high/low bar display in shap (another option is to set color_bar=False in summary plot and call plt.colorbar()
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/{plt_title}_avg_imp.png')
    plt.clf()
    print(50 * "-")


if __name__ == "__main__":
    # SINGLE DAY
    start = time.time()
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            base=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="single_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00')
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="single_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="single_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            next_day=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="single_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True,
                                                            next_day=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="single_day_next_exo")

    # 20 DAY
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            base=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="20_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00')
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="20_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="20_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            next_day=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="20_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True,
                                                            next_day=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="20_day_next_exo")

    end = time.time()
    print(f'Seconds: {end - start}')
