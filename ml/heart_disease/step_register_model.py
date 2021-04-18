import argparse
import os
import shutil
import joblib
import sklearn
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.exceptions import AzureMLException, WebserviceException

from ml.helpers.aml import AmlCustomHelper, logger
from ml.heart_disease.constants import MODEL_NAME


def register_model_to_azureml(
    ws: Workspace, model_name: str, assets_dir: str, scores={}, description: str = ""
):
    """Register the model with it's assets to AzureML.

    Args:
        ws (Workspace): Workspace of the AzureML.
        model_name (str): Name to register the model.
        assets_dir (str): Directory containing the model and other assets, like scalers, preprocessors, etc.
        scores (dict, optional): Dictionary containing scores of the model, such as accuracy and f1-score. Defaults to {}.

    Returns:
        (Model): The object containing the registered model on AzureML.
    """
    # register the model to AzureML
    logger.info("Register the model to AzureML")
    model = Model.register(
        model_path=assets_dir,
        model_name=model_name,
        tags={"Model": "XGBClassifier", "Scaler": "None"},
        properties=scores,
        description=description,
        workspace=ws,
        model_framework=Model.Framework.SCIKITLEARN,
        model_framework_version=sklearn.__version__,
    )
    logger.info(f"Model {model_name} registered at version {model.version}")

    return model


def register_new_model(
    model_path: str,
    score_path: str,
    preprocessor_path: str,
    labels_path: str,
):
    """Gets all the model's assets and register the new model and it's assets to as an AzureML Model.

    Args:
        model_path (str): Path pointing to the model object.
        score_path (str): Path pointing to the score object.
        preprocessor_path (str): Path pointing to the preprocessor object.
        labels_path (str): Path pointing to the class labels object.

    Raises:
        AzureMLException: Error due to the final model having a worst f1-score than the old model.
    """
    logger.info("Proceeding with registration of new model on AzureML")

    # load current model scores
    logger.info("load the current model scores")
    scores = joblib.load(score_path)

    # get names
    logger.info("Get names")
    model_name = model_path.split("/")[-1]
    preprocessor_name = preprocessor_path.split("/")[-1].split(".")[0]
    score_name = score_path.split("/")[-1].split(".")[0]
    labels_name = labels_path.split("/")[-1].split(".")[0]

    logger.info("Model_name:\t{}".format(model_name))
    logger.info("Preprocessor_name:\t{}".format(preprocessor_name))
    logger.info("Score_name:\t{}".format(score_name))
    logger.info("Labels_name:\t{}".format(labels_name))

    # copyfiles to ASSETS_DIR
    aml_helper = AmlCustomHelper()
    if os.path.exists(aml_helper.ASSETS_DIR) and os.path.isdir(aml_helper.ASSETS_DIR):
        shutil.rmtree(aml_helper.ASSETS_DIR)
        logger.info(f"Deleted previous folder found: {aml_helper.ASSETS_DIR}")

    logger.info(f"Copy files to: {aml_helper.ASSETS_DIR}")
    os.makedirs(aml_helper.ASSETS_DIR, exist_ok=True)
    shutil.copyfile(model_path, os.path.join(aml_helper.ASSETS_DIR, model_name))
    shutil.copyfile(
        preprocessor_path,
        os.path.join(aml_helper.ASSETS_DIR, preprocessor_name),
    )
    shutil.copyfile(score_path, os.path.join(aml_helper.ASSETS_DIR, score_name))
    shutil.copyfile(labels_path, os.path.join(aml_helper.ASSETS_DIR, labels_name))

    # Register the model on AzureML
    register_model_to_azureml(
        aml_helper.ws,
        MODEL_NAME,
        aml_helper.ASSETS_DIR,
        scores,
        description="Heart Disease Classification Model",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--preprocessor_fit",
        type=str,
        default="./data/tmp/prepro",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="./data/tmp/labels",
    )
    args = parser.parse_args()

    register_new_model(
        args.trained_model,
        args.model_score,
        args.preprocessor_fit,
        args.labels_path,
    )