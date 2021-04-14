import argparse

import joblib
import pandas as pd
from xgboost import XGBClassifier

from ml.helpers.aml import logger


def train_model(
    X_train_path: str,
    y_train_path: str,
    output_model_path: str,
):
    """Train the classification model using X input variables and y target variable.
    Persist the trained model.

    Args:
        X_train_path (str): The path for X dataset as input for training the model.
        y_train_path (str): The path for y dataset as input for training the model.
        output_model_path (str): The path to persist the trained model.
    """
    # load dataset
    logger.info("Load the training datasets")
    X_train = pd.read_parquet(X_train_path)
    logger.info(f"X shape:\t{X_train.shape}")

    y_train = pd.read_parquet(y_train_path)
    logger.info(f"y shape:\t{y_train.shape}")

    # build the classification model
    model_clf = XGBClassifier(
        colsample_bytree=1,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1e-05,
        n_estimators=200,
        objective="binary:logistic",
        subsample=0.5,
    )

    # fit the classification model
    model_clf.fit(X_train, y_train)
    logger.info(f"XGBClassifier model:\t{model_clf}")

    # persist the trained model
    logger.info("Persist the model sklearn pipeline")
    joblib.dump(model_clf, output_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--X_train",
        type=str,
        default="./data/tmp/X_train",
    )
    parser.add_argument(
        "--y_train",
        type=str,
        default="./data/tmp/y_train",
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="./data/tmp/heart_disease_clf.pkl",
    )
    args = parser.parse_args()

    train_model(
        args.X_train,
        args.y_train,
        args.trained_model,
    )