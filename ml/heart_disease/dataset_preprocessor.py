import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.construction import array
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetPreprocessorValidationException(Exception):
    def __init__(self, message, columns):
        self.columns = columns
        super().__init__(message)

    def get_columns(self):
        return self.columns


class DatasetPreprocessorValidationMissingColumnsException(
    DatasetPreprocessorValidationException
):
    def __init__(self, missing_columns: set):
        self.message = (
            f"Missing {len(missing_columns)} required columns: {missing_columns}"
        )
        super().__init__(self.message, missing_columns)


class DatasetPreprocessorValidationIncorrectStringDatatypeException(
    DatasetPreprocessorValidationException
):
    def __init__(self, columns: set):
        self.message = f"{len(columns)} string columns with incorrect type: {columns}"
        super().__init__(self.message, columns)


class DatasetPreprocessorValidationIncorrectNumericDatatypeException(
    DatasetPreprocessorValidationException
):
    def __init__(self, columns: set):
        self.message = f"{len(columns)} numeric columns with incorrect type: {columns}"
        super().__init__(self.message, columns)


class DatasetPreprocessorValidationIncorrectBoolDatatypeException(
    DatasetPreprocessorValidationException
):
    def __init__(self, columns: set):
        self.message = f"{len(columns)} bool columns with incorrect type: {columns}"
        super().__init__(self.message, columns)


class DatasetPreprocessorValidationIncorrectDatetimeDatatypeException(
    DatasetPreprocessorValidationException
):
    def __init__(self, columns: set):
        self.message = f"{len(columns)} datetime columns with incorrect type: {columns}"
        super().__init__(self.message, columns)


class DatasetPreprocessor:
    """Data preprocessor for training and inference data.

    The transformations are:
    * Onehot encoding on categorical variables

    """

    base_schema = {
        "age": int,
        "sex": str,
        "cp": str,
        "trestbps": int,
        "chol": int,
        "fbs": str,
        "restecg": str,
        "thalach": int,
        "exang": str,
        "oldpeak": np.float64,
        "slope": str,
        "ca": str,
        "thal": str,
    }

    inference_schema = {**base_schema}

    train_schema = {
        **base_schema,
        **{
            "target": int,
        },
    }

    def get_columns(self, dtypes: array, is_inference: bool):
        """Get the name of the columns that matches the given the types.

        Args:
            dtypes (array): The list with the data types of the columns to be returned.
            is_inference (bool): Set to True when using for inference, and False when using to train a model.

        Returns:
            list: List with the column names.
        """
        if is_inference:
            schema = self.inference_schema
        else:
            schema = self.train_schema
        return [key for key, value in schema.items() if value in dtypes]

    def __init__(self):

        self.ohe = {
            k: OneHotEncoder(sparse=False, handle_unknown="ignore")
            for k in [key for key, value in self.base_schema.items() if value == str]
        }

    def validate_missing_columns(self, df: pd.DataFrame, is_inference: bool):
        """Validates the data for missing columns that sould be provided.

        Args:
            df (pd.DataFrame): DataFrame with the dataset to be validated.
            is_inference (bool): Set to True when using for inference, and False when using to train a model.

        Raises:
            DatasetPreprocessorValidationMissingColumnsException: Returns a message with the columns that are missing.
        """
        logger.info("Validating missing columns")
        if is_inference:
            schema = self.inference_schema
        else:
            schema = self.train_schema

        missing_required_columns = set(schema.keys()) - set(df.columns)
        if len(missing_required_columns) != 0:
            raise DatasetPreprocessorValidationMissingColumnsException(
                missing_required_columns
            )
        logger.info("Validate missing columns - OK")

    def validate_datatype_numeric_columns(self, df: pd.DataFrame, is_inference: bool):
        """Validates the data for missing numeric columns that sould be provided.

        Args:
            df (pd.DataFrame): DataFrame with the dataset to be validated.
            is_inference (bool): Set to True when using for inference, and False when using to train a model.

        Raises:
            DatasetPreprocessorValidationIncorrectNumericDatatypeException: Returns a message with the numeric columns that are missing.
        """
        logger.info("Validating numeric columns")

        numeric_columns = self.get_columns([int, np.float64], is_inference)

        # fill none values with number -1 to check dtype (avoids error)
        df_filled = df[numeric_columns].fillna(value=-1)

        columns_incorrect_dtype = set(numeric_columns) - set(
            df_filled.select_dtypes(include=np.number).columns
        )

        if len(columns_incorrect_dtype) != 0:
            raise DatasetPreprocessorValidationIncorrectNumericDatatypeException(
                columns_incorrect_dtype
            )
        logger.info("Validate numeric columns - OK")

    def validate_datatype_string_columns(self, df: pd.DataFrame, is_inference):
        """Validates the data for missing string columns that sould be provided.

        Args:
            df (pd.DataFrame): DataFrame with the dataset to be validated.
            is_inference (bool): Set to True when using for inference, and False when using to train a model.

        Raises:
            DatasetPreprocessorValidationIncorrectStringDatatypeException: Returns a message with the string columns that are missing.
        """
        logger.info("Validating string columns")
        string_columns = self.get_columns([str], is_inference)

        # fiil none values with string 'None' to check dtype (avoids error)
        df_filled = df[string_columns].fillna(value="None")
        columns_incorrect_dtype = set(string_columns) - set(
            df_filled.select_dtypes(include=np.object).columns
        )

        if len(columns_incorrect_dtype) != 0:
            raise DatasetPreprocessorValidationIncorrectStringDatatypeException(
                columns_incorrect_dtype
            )
        logger.info("Validate string columns - OK")

    def validate_data(self, df: pd.DataFrame, is_inference: bool = False):
        """Validates the data and types needed for the dataset.

        Args:
            df (pd.DataFrame): Dataset to be validated
            is_inference (bool, optional): Set to True when using for inference, and False when using to train a model.
            Defaults to False.
        """
        logger.info("Validating data")
        self.validate_missing_columns(df, is_inference)
        self.validate_datatype_numeric_columns(df, is_inference)
        self.validate_datatype_string_columns(df, is_inference)