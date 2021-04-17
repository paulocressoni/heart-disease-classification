import argparse
import joblib
from sklearn.metrics import recall_score
import pandas as pd
from azureml.core.model import Model
from azureml.exceptions import (
    AzureMLException,
    WebserviceException,
    ModelNotFoundException,
)

from ml.helpers.aml import AmlCustomHelper, logger
from ml.heart_disease.constants import MODEL_NAME


def compare_models(
    model_path: str,
    X_test_path: str,
    y_test_path: str,
):
    """Compares the recall of the current model (new model) with the already registered model (old model).
    If the final model's score is NOT greater or equals than the old model's, an Error is raised stopping the execution,
    thus preventing the new model from being registered.

    Args:
        model_path (str): Path to the new model object.
        X_test_path (str): Path to the variables test dataset.
        y_test_path (str): Path to the target test dataset.

    Raises:
        AzureMLException: Error due to the final model having a worst recall than the old model.
    """
    aml_helper = AmlCustomHelper()
    try:
        # download old model
        logger.info("Download old model")
        old_model_aml = Model(aml_helper.ws, MODEL_NAME)

        logger.info(f"aml_helper.ASSETS_DIR:\t{aml_helper.ASSETS_DIR}")
        old_model_aml.download(
            target_dir=f"{'/'.join(aml_helper.ASSETS_DIR.split('/')[0:-1])}",
            exist_ok=True,
        )

        # load old model
        logger.info("Load old model")
        old_model = joblib.load(f"{aml_helper.ASSETS_DIR}/{model_path.split('/')[-1]}")
        logger.info(f"Old Model:\t{old_model}")

        # load new model
        logger.info("Load new model")
        new_model = joblib.load(model_path)
        logger.info(f"New Model:\t{new_model}")

        # load test dataset
        logger.info("Load the test datasets")
        X_test = pd.read_parquet(X_test_path)
        logger.info(f"X_test shape:\t{X_test.shape}")

        y_test = pd.read_parquet(y_test_path)
        logger.info(f"y_test shape:\t{y_test.shape}")

        # make predictions on the test set
        logger.info("Make predictions - Old Model")
        y_hat_old = old_model.predict(X_test)

        logger.info("Make predictions - Old Model")
        y_hat_new = new_model.predict(X_test)

        # Recall
        logger.info("Calculates Recall")
        recall_old = recall_score(y_test, y_hat_old)
        recall_new = recall_score(y_test, y_hat_new)
        logger.info(f"Old model recall:\t{recall_old}")
        logger.info(f"New model recall:\t{recall_new}")

        if recall_old > recall_new:
            # force to fail the AzureML pipeline step
            raise AzureMLException(
                exception_message="The new model metric scored less than the old model. Thus, We will not proceed to the new model registration."
            )

    except (WebserviceException, ModelNotFoundException):
        logger.info(f"No previous registered model found with the name:\t{MODEL_NAME}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model",
        type=str,
        default="./data/tmp/heart_disease_clf.pkl",
    )
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
    args = parser.parse_args()

    compare_models(
        args.trained_model,
        args.X_test,
        args.y_test,
    )