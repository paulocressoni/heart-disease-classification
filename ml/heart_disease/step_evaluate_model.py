import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
)
from azureml.core import Run
from azureml.core.run import _OfflineRun

from ml.helpers.aml import logger
from ml.heart_disease.dataset_preprocessor import DatasetPreprocessor


def plot_confusion_matrix(cm, target_col_description, title="Heart Disease"):
    """Plots a confusion matrix.

    Args:
        cm ([type]): Confusion Matrix.
        target_col_description ([array]): List of target columns descriptions in class order.
        title (str, optional): Title of the plot. Defaults to "Heart Disease".

    Returns:
        [type]: Image of the Confusion Matrix
    """
    plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 100})
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        square=True,
        cmap="Blues_r",
        xticklabels=target_col_description,
        yticklabels=target_col_description,
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title(title, size=16)
    return plt


def evaluate_model(
    X_test_path: str,
    y_test_path: str,
    model_path: str,
    model_score_path: str,
):
    """Evaluate the model using metrics such as accuracy and f1-score and persists these metrics generated as an object.
    Also logs these metrics to AzureML.

    Args:
        X_test_path (str): Path to X input for model evaluation
        y_test_path (str): Path to y input for model evaluation
        model_path (str): Path to model to be evaluated
        model_score_path (str): Path to persist the evaluation metrics of the model
    """
    # load dataset
    logger.info("Load the test datasets")

    X_test = pd.read_parquet(X_test_path)
    logger.info(f"X_test shape:\t{X_test.shape}")

    y_test = pd.read_parquet(y_test_path)
    logger.info(f"y_test shape:\t{y_test.shape}")

    # load the model
    model = joblib.load(model_path)

    # make predictions on the test set
    logger.info("Make predictions")
    y_hat = model.predict(X_test)

    scores = {}
    # model accuracy
    logger.info("Calculates accuracy")
    acc_score = accuracy_score(y_test, y_hat)
    scores["accuracy"] = acc_score

    # F1-Score Macro
    logger.info("Calculates F1-Score Macro")
    f1_score_macro = f1_score(y_test, y_hat, average="macro")
    scores["f1_score_macro"] = f1_score_macro

    # Recall
    logger.info("Calculates Recall")
    recall = recall_score(y_test, y_hat)
    scores["recall"] = recall

    logger.info(f"scores:\t{scores}")

    # Confusion Matrix
    logger.info("Calculates Confusion Matrix")
    cm = confusion_matrix(y_test, y_hat)
    data_prep = DatasetPreprocessor()
    cm_json = {
        "schema_type": "confusion_matrix",
        #    "schema_version": "v1",
        "data": {
            "class_labels": data_prep.target_col_description,
            "matrix": [[int(y) for y in x] for x in cm],
        },
    }

    # plot the confusion matrix
    logger.info("Plot Confusion Matrix")
    cm_plot = plot_confusion_matrix(
        cm, data_prep.target_col_description, "Heart Disease"
    )

    # Log to AzureML
    run = Run.get_context()
    if not isinstance(run, _OfflineRun):
        parent_run = run.parent
        parent_run.log(name="Accuracy", value=acc_score)
        parent_run.log(name="F1-Score Macro", value=f1_score_macro)
        parent_run.log(name="Recall", value=recall)
        parent_run.log_confusion_matrix(name="Confusion Matrix", value=cm_json)
        parent_run.log_image(name="Confusion Matrix", plot=cm_plot)
    else:
        logger.info("Offline run")
        run.log(name="Accuracy", value=acc_score)
        run.log(name="F1-Score Macro", value=f1_score_macro)
        run.log(name="Recall", value=recall)
        run.log(name="Confusion Matrix", value=cm)

    # persist scores
    logger.info(f"persist scores")
    joblib.dump(scores, model_score_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--X_test",
        type=str,
        default="./data/tmp/X_test",
    )
    parser.add_argument(
        "--y_test",
        type=str,
        default="./data/tmp/y_test",
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="./data/tmp/heart_disease_clf.pkl",
    )
    parser.add_argument(
        "--model_score",
        type=str,
        default="./data/tmp/heart_disease_clf_evaluation",
    )
    args = parser.parse_args()

    evaluate_model(
        args.X_test,
        args.y_test,
        args.trained_model,
        args.model_score,
    )