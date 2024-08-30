"""GitHub Classroom autograding script."""

import os
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_estimator():
    """Load trained model from disk."""

    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def load_datasets():
    """Load train and test datasets."""

    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")

    x_train = train_dataset.drop("target", axis=1)
    y_train = train_dataset["target"]

    x_test = test_dataset.drop("target", axis=1)
    y_test = test_dataset["target"]

    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    """Evaluate model performance."""

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2


def compute_metrics():
    """Compute model metrics."""

    estimator = load_estimator()
    assert estimator is not None, "Model not found"

    x_train, x_test, y_true_train, y_true_test = load_datasets()

    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)

    mse_train, mae_train, r2_train = eval_metrics(y_true_train, y_pred_train)
    mse_test, mae_test, r2_test = eval_metrics(y_true_test, y_pred_test)

    return mse_train, mae_train, r2_train, mse_test, mae_test, r2_test


def run_grading():
    """Run grading script."""

    mse_train, mae_train, r2_train, mse_test, mae_test, r2_test = compute_metrics()

    assert mse_train < 2903, f"Train MSE: {mse_train:.2f}"
    assert mse_test < 2856, f"Test MSE: {mse_test:.2f}"
    assert mae_train < 43.8, f"Train MAE: {mae_train:.2f}"
    assert mae_test < 43.2, f"Test MAE: {mae_test:.2f}"
    assert r2_train > 0.48, f"Train R2: {r2_train:.2f}"
    assert r2_test > 0.56, f"Test R2: {r2_test:.2f}"



if __name__ == "__main__":
    run_grading()
