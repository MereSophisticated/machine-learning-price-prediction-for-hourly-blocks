import time

import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from src.data_analysis.data_retrieval import get_train_test_split

plot_path = 'plots'
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


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
    plt_title = "c_xgboost_" + plt_title
    print(plt_title)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, predictions))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=y_train.unique(), show=False)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/{plt_title}_imp.png')
    plt.clf()
    print(50 * "-")


if __name__ == "__main__":
    # SIMPLE
    # SINGLE DAY
    start = time.time()
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            base=True,
                                                            labeled=True,
                                                            simple_labels=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_single_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_single_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_single_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            next_day=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_single_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True,
                                                            next_day=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_single_day_next_exo")

    # 20 DAY
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            base=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_20_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_20_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_20_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            next_day=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_20_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True,
                                                            next_day=True,
                                                            labeled=True,
                                                            simple_labels=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="simple_20_day_next_exo")

    # COMPLEX
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            base=True,
                                                            labeled=True)
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_single_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_single_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_single_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            next_day=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_single_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-21 00:00:00',
                                                            exogenous=True,
                                                            next_day=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_single_day_next_exo")

    # 20 DAY
    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            base=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_20_day_baseline")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_20_day")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_20_day_exo")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            next_day=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_20_day_next")

    X_train, X_test, y_train, y_test = get_train_test_split(split_timestamp='2022-03-03 00:00:00',
                                                            exogenous=True,
                                                            next_day=True,
                                                            labeled=True
                                                            )
    train_and_test_model(X_train, X_test, y_train, y_test, plt_title="complex_20_day_next_exo")

    end = time.time()
    print(f'Seconds: {end - start}')
