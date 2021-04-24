import argparse

import joblib
import pandas as pd
from azureml.core import Dataset
from azureml.data import DataType

from ml.helpers.aml import AmlCustomHelper, logger
from azureml.core.model import Model
from azureml.exceptions import (
    AzureMLException,
    WebserviceException,
    ModelNotFoundException,
)
from ml.heart_disease.constants import MODEL_NAME, PREPROCESSOR_NAME


def transform_infer_data(
    dataset: pd.DataFrame, aml_model_name: str, preprocessor_name: str
):
    """Transform the dataset for inference. Downloads the registered Model on AzureML and load the already
    fit preprocessor to use for the data transformation.

    Args:
        dataset (pd.DataFrame): Dataset to transform.
        aml_model_name (str): The name of the Model registered on AzureML.
        preprocessor_name (str): The name of the preprocessor registered with the Model's assets on AzureML.
    """
    aml_helper = AmlCustomHelper()
    try:
        # download old model
        logger.info("Download aml model")
        aml_model = Model(aml_helper.ws, aml_model_name)

        logger.info(f"aml_helper.ASSETS_DIR:\t{aml_helper.ASSETS_DIR}")
        aml_model.download(
            target_dir=f"{'/'.join(aml_helper.ASSETS_DIR.split('/')[0:-1])}",
            exist_ok=True,
        )

        # load old model
        logger.info("Load aml model's preprocessor")
        preprocessor = joblib.load(f"{aml_helper.ASSETS_DIR}/{preprocessor_name}")

        logger.info(f"Transform the dataset")
        transformed_dataset = preprocessor.transform(dataset, is_inference=True)

        return transformed_dataset

    except (WebserviceException, ModelNotFoundException):
        logger.info(f"No previous registered model found with the name:\t{MODEL_NAME}")
        # force to fail the AzureML pipeline step
        raise AzureMLException(exception_message="Stopping the execution.")


def preprocess_input_data(
    input_file_path: str,
    transformed_data_path: str,
):
    """Preprocess the dataset making it ready for inference. The preprocessor is loaded
    from model assets and it's transformation is applied to the dataset so it could be
    ready to input the model.

    Args:
        input_file_path (str): Path to the input file to be loaded.
        transformed_data_path (str): Path to persist transformed data.
    """
    # get datastore
    aml_helper = AmlCustomHelper()
    datastore = aml_helper.ws.get_default_datastore()

    # load dataset
    logger.info(f"Loading data..")
    # set column data types
    data_types = {
        "age": DataType.to_long(),
        "sex": DataType.to_string(),
        "cp": DataType.to_string(),
        "trestbps": DataType.to_long(),
        "chol": DataType.to_long(),
        "fbs": DataType.to_string(),
        "restecg": DataType.to_string(),
        "thalach": DataType.to_long(),
        "exang": DataType.to_string(),
        "oldpeak": DataType.to_float(),
        "slope": DataType.to_string(),
        "ca": DataType.to_string(),
        "thal": DataType.to_string(),
    }

    # Create a TabularDataset to represent tabular data in parquet files
    df_data = Dataset.Tabular.from_parquet_files(
        path=[(datastore, input_file_path)], set_column_types=data_types
    ).to_pandas_dataframe()

    logger.info(f"Loaded data shape:\t{df_data.shape}")
    logger.info(f"Loaded data info:\t{df_data.info()}")
    logger.info(f"Loaded data first rows:\t{df_data.head(5)}")

    # transform dataset
    transformed_data = transform_infer_data(df_data, MODEL_NAME, PREPROCESSOR_NAME)

    # save transformed dataset to parquet
    logger.info("Persist the transformed dataset")
    transformed_data.to_parquet(transformed_data_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        default="infer_data/dataset_to_infer.parquet",
    )
    parser.add_argument(
        "--transformed_data_path",
        type=str,
        default="./data/tmp/transformed_data",
    )
    args = parser.parse_args()

    preprocess_input_data(
        args.input_file_path,
        args.transformed_data_path,
    )