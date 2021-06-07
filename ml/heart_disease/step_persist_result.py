import argparse

import pandas as pd

from ml.helpers.aml import AmlCustomHelper, logger


def upload_results_to_azureml_datastore(
    predictions: str,
    output_file_path: str,
):
    """Uploads the inference result file to the Azure Datastore.

    Args:
        predictions (str): Path to the dataset.
        output_file_path (str): Path to output the inference data.
    """

    # get datastore according to env
    aml_helper = AmlCustomHelper()
    datastore = aml_helper.ws.get_default_datastore()

    # upload files to AzureBlobDatastore
    logger.info("Upload predictions...")
    datastore.upload_files(
        files=[predictions],
        target_path=output_file_path,
        overwrite=True,
        show_progress=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_path",
        type=str,
        default="./data/tmp/pred.csv",
    )

    parser.add_argument(
        "--output_file_path",
        type=str,
        default="infer_data/result/",
    )
    args = parser.parse_args()

    upload_results_to_azureml_datastore(
        args.inference_path,
        args.output_file_path,
    )