import argparse

from azureml.core import Dataset
from azureml.data import DataType
from ml.helpers.aml import AmlCustomHelper, logger

from ml.heart_disease.dataset_preprocessor import DatasetPreprocessor


def validate_input_data(input_file_path: str):
    """Validates the expected input used for the training.

    Args:
        input_file_path (str): Path used to read the input data for training.
    """
    # get datastore according to env
    aml_helper = AmlCustomHelper()
    # datastore that points to a storage account
    datastore = aml_helper.ws.get_default_datastore()
    logger.info(f"Datastore:\t{datastore}")

    # load dataset
    logger.info(f"Loading data...")

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

    # Create a TabularDataset to represent tabular data in delimited files
    df_data = Dataset.Tabular.from_parquet_files(
        path=[(datastore, input_file_path)], set_column_types=data_types
    ).to_pandas_dataframe()

    logger.info(f"Loaded data shape:\t{df_data.shape}")
    logger.info(f"Loaded data info:\t{df_data.info()}")
    logger.info(f"Loaded data first 5 rows:\t{df_data.head(5)}")

    # Validates the input dataset
    logger.info("Validates the input dataset")
    data_prep = DatasetPreprocessor()
    data_prep.validate_data(df_data, is_inference=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        default="infer_data/dataset_to_infer.parquet",
    )
    args = parser.parse_args()

    validate_input_data(
        args.input_file_path,
    )