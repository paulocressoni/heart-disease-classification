import argparse

import joblib
from azureml.core import Dataset
from azureml.data import DataType
from sklearn.model_selection import StratifiedShuffleSplit

from ml.helpers.aml import AmlCustomHelper, logger
from ml.heart_disease.dataset_preprocessor import DatasetPreprocessor


def preprocess_input_data(
    input_file_path: str,
    preprocessor: str,
    labels: str,
    output_data_X_train: str,
    output_data_y_train: str,
    output_data_X_test: str,
    output_data_y_test: str,
):
    """Prepare the data for the model training and model evaluation. Fit a preprocessor (for later use when inferencing)
    and applies it to the dataset to transform it to the expected input for the model.

    Args:
        input_file_path (str): Path to the input file to be loaded.
        preprocessor (str): Path to persist the fit preprocessor.
        labels (str): Path to persist the class labels.
        output_data_X_train (str): Path to persist X_train.
        output_data_y_train (str): Path to persist y_train.
        output_data_X_test (str): Path to persist X_test.
        output_data_y_test (str): Path to persist y_test.
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
        "target": DataType.to_long(),
    }

    # Create a TabularDataset to represent tabular data in delimited files
    df_data = Dataset.Tabular.from_delimited_files(
        path=[(datastore, input_file_path)], set_column_types=data_types
    ).to_pandas_dataframe()

    logger.info(f"Loaded data shape:\t{df_data.shape}")
    logger.info(f"Loaded data info:\t{df_data.info()}")
    logger.info(f"Loaded data first rows:\t{df_data.head(5)}")

    logger.info("Fit the input data to the preprocessor")
    data_prep = DatasetPreprocessor()
    data_prep.fit(df_data)

    # apply the transformations on the input dataset
    logger.info("apply the transformations on the input dataset")
    logger.info(f"before transformations: {df_data.shape}")

    output_df = data_prep.transform(df_data, is_inference=False)

    logger.info(f"after transformations: {output_df.shape}")
    logger.info(f"after transformations: {output_df.info()}")

    # split training and target features
    logger.info("split training and target features")
    X = output_df.drop(columns=[data_prep.target_col])
    y = output_df[[data_prep.target_col]]

    # split train and test dataset
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=123)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # reset the suffled indexes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"X_train:\t{X_train.shape}")
    print(f"X_test:\t{X_test.shape}")
    print(f"y_train:\t{y_train.shape}")
    print(f"y_test:\t{y_test.shape}")

    # persist the train outputs
    logger.info("persist the train outputs")
    X_train.to_parquet(output_data_X_train)
    y_train.to_parquet(output_data_y_train)

    # persist the test outputs
    logger.info("persist the test outputs")
    X_test.to_parquet(output_data_X_test)
    y_test.to_parquet(output_data_y_test)

    # persist fit preprocessor
    logger.info("persist fit preprocessor")
    joblib.dump(data_prep, preprocessor)

    # persist the class labels
    logger.info("persist class labels")
    label_classes = {
        0: "absence of heart disease",
        1: "presence of heart disease",
    }
    joblib.dump(label_classes, labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        default="input_data/heart.xls",
    )
    parser.add_argument(
        "--preprocessor_path",
        type=str,
        default="./data/tmp/prepro",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="./data/tmp/labels",
    )
    parser.add_argument(
        "--output_data_X_train",
        type=str,
        default="./data/tmp/X_train",
    )
    parser.add_argument(
        "--output_data_y_train",
        type=str,
        default="./data/tmp/y_train",
    )
    parser.add_argument(
        "--output_data_X_test",
        type=str,
        default="./data/tmp/X_test",
    )
    parser.add_argument(
        "--output_data_y_test",
        type=str,
        default="./data/tmp/y_test",
    )
    args = parser.parse_args()

    preprocess_input_data(
        args.input_file_path,
        args.preprocessor_path,
        args.labels_path,
        args.output_data_X_train,
        args.output_data_y_train,
        args.output_data_X_test,
        args.output_data_y_test,
    )