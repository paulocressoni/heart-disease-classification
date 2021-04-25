#!/usr/bin/env python3
"""
Creates a pipeline for batch inference

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
    EXPERIMENT_NAME_INFER,
    PIPELINE_DESCRIPTION_INFER,
    PIPELINE_NAME_INFER,
)

SOURCE_PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/"))


def get_run_config():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.get_default_run_config(
        pip_packages=[
            *aml_helper.DEFAULT_PIP_PACKAGES,
            "xgboost==1.3.3",
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
    INPUT_FILE_PATH = PipelineParameter(
        name="INPUT_FILE_PATH",
        default_value="infer_data/dataset_to_infer.parquet",
    )

    # Data to Flow Among Pipeline's Steps
    original_dataset = PipelineData(
        "original_dataset", datastore=datastore, output_mode="mount"
    )
    preprocessed_dataset = PipelineData(
        "preprocessed_dataset", datastore=datastore, output_mode="mount"
    )

    # Pipeline steps
    step_validate_inference_data = PythonScriptStep(
        name="step_validate_inference_data",
        script_name="./ml/heart_disease/step_validate_inference_data.py",
        compute_target=compute_target,
        arguments=[
            "--input_file_path",
            INPUT_FILE_PATH,
        ],
        inputs=[],
        outputs=[],
        allow_reuse=False,
        runconfig=run_config,
        params=aml_helper.get_default_step_params(),
    )

    step_preprocess_infer_data = PythonScriptStep(
        name="step_preprocess_infer_data",
        script_name="./ml/heart_disease/step_preprocess_infer_data.py",
        compute_target=compute_target,
        arguments=[
            "--input_file_path",
            INPUT_FILE_PATH,
            "--transformed_data_path",
            preprocessed_dataset,
            "--original_data_path",
            original_dataset,
        ],
        inputs=[],
        outputs=[preprocessed_dataset, original_dataset],
        allow_reuse=False,
        runconfig=run_config,
        params=aml_helper.get_default_step_params(),
    )

    pipeline = Pipeline(
        workspace=aml_helper.ws,
        steps=StepSequence(
            steps=[step_validate_inference_data, step_preprocess_infer_data]
        ),
    )
    return pipeline


def deploy_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    aml_helper.deploy_pipeline(
        pipeline_name=PIPELINE_NAME_INFER,
        pipeline_description=PIPELINE_DESCRIPTION_INFER,
        pipeline=get_pipeline(),
    )


def validate_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.validate_pipeline(pipeline=get_pipeline())


def run_pipeline():
    aml_helper = AmlCustomHelper(source_path=SOURCE_PATH)
    return aml_helper.submit_pipeline(
        pipeline_name=PIPELINE_NAME_INFER,
        experiment_name=EXPERIMENT_NAME_INFER,
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
