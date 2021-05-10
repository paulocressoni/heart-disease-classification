import argparse
import joblib

import pandas as pd
from azureml.core.model import Model

from ml.heart_disease.constants import MODEL_NAME, PREPROCESSOR_NAME
from ml.helpers.aml import AmlCustomHelper, logger


def batch_inference(dataset: pd.DataFrame, model):
    """Make the predictions on the data passed using the model passed.

    Args:
        dataset (pd.DataFrame): Dataset to perform the prediction.
        model (str): The model to execute the prediction.
    """
    pred = model.predict(dataset)
    return pred


def make_predictions(
    transformed_data_path: str,
    inference_path: str,
):
    """Loads the datasets already transformed and make the predictions.

    Args:
        transformed_data_path (str): Path to the transformed dataset ready to be
        feed to the model for prediction.
        inference_path (str): Path to the predictions.
    """
    # helper
    aml_helper = AmlCustomHelper()

    # load dataset
    logger.info("Load the training datasets")
    X = pd.read_parquet(transformed_data_path)
    logger.info(f"X shape:\t{X.shape}")

    # download the model
    logger.info("Download Azureml model")
    az_model = Model(aml_helper.ws, MODEL_NAME)

    logger.info(f"aml_helper.ASSETS_DIR:\t{aml_helper.ASSETS_DIR}")
    az_model.download(
        target_dir=f"{'/'.join(aml_helper.ASSETS_DIR.split('/')[0:-1])}",
        exist_ok=True,
    )

    # load old model
    logger.info("Load Azureml model")
    model = joblib.load(f"{aml_helper.ASSETS_DIR}/{MODEL_NAME}")
    logger.info(f"Model:\t{model}")

    # batch inference - get predictions
    pred = batch_inference(X, model)
    logger.info(f"pred.shape:\t{pred.shape}")

    # transform the predictions to DataFrame
    pred = pd.DataFrame(pred, columns=["prediction"])
    logger.info(pred.head())

    # persist predictions
    pred.to_parquet(inference_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformed_data_path",
        type=str,
        default="./data/tmp/transformed_data",
    )
    parser.add_argument(
        "--inference_path",
        type=str,
        default="./data/tmp/pred",
    )
    args = parser.parse_args()

    make_predictions(
        args.transformed_data_path,
        args.inference_path,
    )
