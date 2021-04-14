#!/usr/bin/env python3
"""
Creates a pipeline for training

"""
import argparse
import os
import sys

from azureml.pipeline.core import PipelineData, PipelineParameter
from azureml.pipeline.core.builder import StepSequence
from azureml.pipeline.core.pipeline import Pipeline
from azureml.pipeline.steps.python_script_step import PythonScriptStep

from ml.helpers.aml import AmlCustomHelper
from ml.heart_disease.constants import (
    EXPERIMENT_NAME_TRAIN,
    LABELS_NAME,
    MODEL_NAME,
    PIPELINE_DESCRIPTION_TRAIN,
    PIPELINE_NAME_TRAIN,
    PREPROCESSOR_NAME,
    SCORE_NAME,
)

SOURCE_PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/"))


def get_run_config():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.get_default_run_config(
        pip_packages=[
            *aml_helper.DEFAULT_PIP_PACKAGES,
            "imblearn==0.0",
            "lightgbm==2.3.0",
            "matplotlib==3.2.1",
            "seaborn==0.11.0",
        ]
    )


def get_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    datastore = aml_helper.ws.get_default_datastore()

    compute_target = aml_helper.get_default_compute_target()

    run_config = get_run_config()

    # Pipeline Parameters are passed when submitting the pipeline
    INPUT_TRAINING_FILE_PATH = PipelineParameter(
        name="INPUT_TRAINING_FILE_PATH",
        default_value="input_data/heart.xls",
    )

    # Data to Flow Among Pipeline's Steps
    preprocessor_path = PipelineData(
        "preprocessor_path", datastore=datastore, output_mode="mount"
    )
    labels_path = PipelineData("labels_path", datastore=datastore, output_mode="mount")
    X_train = PipelineData("X_train", datastore=datastore, output_mode="mount")
    y_train = PipelineData("y_train", datastore=datastore, output_mode="mount")
    X_test = PipelineData("X_test", datastore=datastore, output_mode="mount")
    y_test = PipelineData("y_test", datastore=datastore, output_mode="mount")

    # Pipeline steps
    step_validate_data = PythonScriptStep(
        name="step_validate_data",
        script_name="./ml/heart_disease/step_validate_training_data.py",
        compute_target=compute_target,
        arguments=[
            "--input_file_path",
            INPUT_TRAINING_FILE_PATH,
        ],
        inputs=[],
        outputs=[],
        allow_reuse=False,
        runconfig=run_config,
        params=aml_helper.get_default_step_params(),
    )

    step_preprocess_data = PythonScriptStep(
        name="step_preprocess_data",
        script_name="./ml/heart_disease/step_preprocess_train_data.py",
        compute_target=compute_target,
        arguments=[
            "--input_file_path",
            INPUT_TRAINING_FILE_PATH,
            "--preprocessor_path",
            preprocessor_path,
            "--labels_path",
            labels_path,
            "--output_data_X_train",
            X_train,
            "--output_data_y_train",
            y_train,
            "--output_data_X_test",
            X_test,
            "--output_data_y_test",
            y_test,
        ],
        inputs=[],
        outputs=[preprocessor_path, labels_path, X_train, y_train, X_test, y_test],
        allow_reuse=False,
        runconfig=run_config,
        params=aml_helper.get_default_step_params(),
    )

    pipeline = Pipeline(
        workspace=aml_helper.ws,
        steps=StepSequence(steps=[step_validate_data, step_preprocess_data]),
    )
    return pipeline


def deploy_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    aml_helper.deploy_pipeline(
        pipeline_name=PIPELINE_NAME_TRAIN,
        pipeline_description=PIPELINE_DESCRIPTION_TRAIN,
        pipeline=get_pipeline(),
    )


def validate_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.validate_pipeline(pipeline=get_pipeline())


def run_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.submit_pipeline(
        pipeline_name=PIPELINE_NAME_TRAIN,
        experiment_name=EXPERIMENT_NAME_TRAIN,
        pipeline=get_pipeline(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", default=True, action="store_true")
    parser.add_argument("--deploy", default=False, action="store_true")
    parser.add_argument("--run", default=False, action="store_true")
    args = parser.parse_args()

    if args.deploy:
        deploy_pipeline()
        sys.exit(0)

    if args.run:
        run_pipeline()
        sys.exit(0)

    if args.validate:
        validate_pipeline()
        sys.exit(0)
