import itertools
import json
import logging
import operator
import os
from copy import Error
from functools import reduce
from glob import glob
from shutil import copy
from time import sleep

from azureml.core import Dataset, Datastore, RunConfiguration, Workspace
from azureml.core.authentication import (
    AzureCliAuthentication,
    ServicePrincipalAuthentication,
)
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute.aks import AksCompute
from azureml.core.environment import DEFAULT_CPU_IMAGE
from azureml.core.model import Model
from azureml.core.runconfig import Run, RunConfiguration
from azureml.data.dataset_error_handling import DatasetValidationError
from azureml.exceptions import UserErrorException
from azureml.pipeline.core import PipelineEndpoint
from azureml.pipeline.core._restclients.aeva.models.error_response import (
    ErrorResponseException,
)
from dotenv import load_dotenv

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SOURCE_PATH = pathname = "/".join(
    os.path.dirname(os.path.realpath(__file__)).split("/")[:-1]
)

load_dotenv(override=True)


class InvalidAuthenticationException(Exception):
    def __init__(self, error_msg, error_response=None):
        super(InvalidAuthenticationException, self).__init__(error_msg)
        self.error_response = error_response


class PipelineException(Exception):
    def __init__(self, error_msg, error_response=None):
        super(PipelineException, self).__init__(error_msg)
        self.error_response = error_response


class AmlCustomHelper:
    """Class to aid in integration with Azure Machine Learning (AML)
    Use this class to return and iterate with the main features of AML.
    Instantiating this class you will already be authenticated with Azure and also with AML.
    Look at the method docstrings to learn more about each available operation.
    """

    DEFAULT_PIP_PACKAGES = [
        "azure-cli-core==2.16.0",
        "azureml-sdk==1.15.0",
        "azureml-mlflow==1.15.0",
        "azureml-defaults==1.15.0",
        "scikit-learn==0.23.2",
        "pandas==1.1.1",
        "joblib==0.16.0",
        "inference-schema==1.1.0",
        "numpy==1.19.2",
        "python-dotenv==0.14.0",
        "pyjwt==1.7.1",
    ]

    def __init__(self, source_path: str = SOURCE_PATH) -> None:
        """Initializes the class, defining the main attributes that will be used in a shared way with the other methods

        In this method there is also the authentication and initialization of the variable "ws" that represents the AML Workspace

        Args:
            source_path (str): Source path of the source code that is calling the helper
                     For example:
                        The build_amlignore_file uses this value to generate a list of files
                        that will be ignored when uploading to the AML
        """
        self.source_path = source_path
        self.use_service_principal = os.getenv("FORCE_AUTH_SERVICE_PRINCIPAL") == "TRUE"
        self.TENANT_ID = os.getenv("TENANT_ID")
        self.ACTIVE_DIRECTORY_CLIENT_ID = os.getenv("ACTIVE_DIRECTORY_CLIENT_ID")
        self.KV_SERVICE_PRINCIPAL_SECRET_NAME = os.getenv(
            "KV_SERVICE_PRINCIPAL_SECRET_NAME",
            "amlheartdiseas2396477080",
        )
        self.SERVICE_PRINCIPAL_SECRET = os.getenv("SERVICE_PRINCIPAL_SECRET")
        self.SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
        self.RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME")
        self.AML_WORKSPACE_NAME = os.getenv("AML_WORKSPACE_NAME")
        self.VNET_NAME = os.getenv("VNET_NAME")
        self.SUBNET_NAME = os.getenv("SUBNET_NAME")
        self.VALIDATE_ONLY = str(os.getenv("VALIDATE_ONLY")) == "TRUE"
        self.STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
        self.ASSETS_DIR = os.getenv(
            "ASSETS_DIR", default=os.path.join(self.source_path, "assets")
        )
        self.PIPELINE_NAME = os.getenv("PIPELINE_NAME")
        self.EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

        ### Runtime Variables are prefixed with AML_PARAMETER
        # Overwrite attributes if running on AML Workspace,
        # allowing us to run locally and on cloud smoothly
        for k, v in os.environ.items():
            if k.startswith("AML_PARAMETER_"):
                self.__setattr__(k.replace("AML_PARAMETER_", ""), v)

        self.run_context = Run.get_context(allow_offline=True)
        self.ws = self.get_workspace()
        self.env = os.getenv("ENV")

    def build_amlignore_file(self):
        """Build .amlignore file dynamicly

        Avoid unecessary upload, the .amligore file is used to manage which files will be used.

        To prevent unnecessary files from being included in the snapshot,
        make an ignore file (.gitignore or .amlignore) in the directory.
        Add the files and directories to exclude to this file.
        For more information on the syntax to use inside this file, see syntax and patterns for .gitignore.
        The .amlignore file uses the same syntax.
        If both files exist, the .amlignore file takes precedence.

        More info:
        https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#submit-the-experiment
        """
        copy("./.gitignore", "./.amlignore")
        paths_to_ignore: list = [
            *[
                "venv",
                "notebooks",
            ],
            *[
                p
                for p in glob(f"ml/*", recursive=True)
                if p
                not in [
                    "ml/helpers",
                    f"{'/'.join(self.source_path.split('/')[-2:])}",
                ]
            ],
        ]
        with open("./.amlignore", "a") as f_amlignore:
            f_amlignore.write("\n".join(paths_to_ignore))

    def get_default_step_params(self):
        """Pre Pipeline Deployment local step, inject environment variables

        Remember that you should prefix "AML_PARAMETER_" variables in order to use inside the steps
        """
        params = {}

        for k in [
            "SUBSCRIPTION_ID",
            "RESOURCE_GROUP_NAME",
            "AML_WORKSPACE_NAME",
            "STORAGE_ACCOUNT_NAME",
        ]:
            params[k] = self.__getattribute__(k)

        return params

    def get_credentials(self):
        """Returns the credential that will be used to authenticate with Azure and AML

         There are two ways to authenticate to Azure through this helper, via Service Principal or Azure Cli

         Azure Cli is normally used for local development, so the credential is inferred from the lib "az cli" of your machine.

         Service Principal is required during deployment and / or executions in the clusters, for this it is necessary to provide the parameters below:
            - ACTIVE_DIRECTORY_CLIENT_ID
            - SERVICE_PRINCIPAL_SECRET
            - TENANT_ID

        Returns:
            _ (ServicePrincipalAuthentication|AzureCliAuthentication) : Azure Authentication object
        """
        if (
            self.run_context._run_id.startswith("OfflineRun_")
            and not self.use_service_principal
        ):
            # Via Azure Cli Credentials
            logger.info(
                "[!]Authenticating using Azure Cli Credentials - Only local development environment"
            )
            return AzureCliAuthentication()

        if (
            self.ACTIVE_DIRECTORY_CLIENT_ID
            and (self.KV_SERVICE_PRINCIPAL_SECRET_NAME or self.SERVICE_PRINCIPAL_SECRET)
            and self.TENANT_ID
        ):
            service_principal_secret = (
                self.SERVICE_PRINCIPAL_SECRET
                or self.run_context.get_secret(
                    name=self.KV_SERVICE_PRINCIPAL_SECRET_NAME
                )
            )
            logger.info("[!]Authenticating via Service Principal Credentials")
            return ServicePrincipalAuthentication(
                tenant_id=self.TENANT_ID,
                service_principal_id=self.ACTIVE_DIRECTORY_CLIENT_ID,
                service_principal_password=service_principal_secret,
            )

        raise InvalidAuthenticationException("Invalid authentication method")

    def get_workspace(self):
        """Returns the AML workspace

         If the code is being executed locally (OfflineRun), a new object of the Workspace class is instantiated.

         However, if we are already executing the code inside a pipeline, it is not necessary to instantiate it,
         just access from the execution context.

         Alternatively, it is possible to use Workspace.from_config () if you are using an AML compute instance as well.
        Returns:
            _ (azureml.core.Workspace) : Workspace do Azure Machine Learning
        """
        if not self.run_context._run_id.startswith("OfflineRun_"):
            return self.run_context.experiment.workspace

        return Workspace(
            subscription_id=self.SUBSCRIPTION_ID,
            resource_group=self.RESOURCE_GROUP_NAME,
            workspace_name=self.AML_WORKSPACE_NAME,
            auth=self.get_credentials(),
        )

    def get_aks_compute_target(self):
        """Returns the Azure Kubernetes Service that is registered with AML

        Returns:
            _ (AksCompute) : AML computing resource
        """
        return AksCompute(self.ws, f"aksdataocean{self.env}")

    def get_default_compute_target(self):
        """Returns the default Compute Cluster for running pipelines

         Remembering that this cluster is shared for training and inference

        Returns:
            _ (azureml.core.ComputeTarget) : AML computing resource
        """
        return self.get_compute_target(
            aml_compute_target_name="compute-cluster",
            aml_vm_type="Standard_DS3_v2",
            min_nodes=1,
            max_nodes=4,
        )

    def get_compute_target(
        self,
        aml_compute_target_name: str = None,
        aml_vm_type: str = None,
        min_nodes: int = 1,
        max_nodes: int = 4,
    ):
        """Criar ou retornar AzureML compute target

        Args:
            aml_compute_target_name (str): Name of the compute cluster|instance|other
            aml_resource_group_name (str): Name of the Resource Group from the Compute cluster|instance|other
            aml_vm_type (str): Size of the Compute cluster|instance|other when creation is needed
            min_nodes (int, optional): Minimum active nodes to a Compute Cluster. Defaults to 1.
            max_nodes (int, optional): Maximum active nodes to a Compute Cluster. Defaults to 4.
        """

        if aml_compute_target_name in self.ws.compute_targets:
            return AmlCompute(workspace=self.ws, name=aml_compute_target_name)

        # if the compute target not exists
        logger.info(
            f"Compute target {aml_compute_target_name} not found. Creating then."
        )

        compute_config = AmlCompute.provisioning_configuration(
            vm_size=aml_vm_type,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        aml_compute = ComputeTarget.create(
            self.ws, aml_compute_target_name, compute_config
        )
        aml_compute.wait_for_completion(show_output=True)
        return aml_compute

    def get_pipeline_endpoint(self, pipeline_name):
        """Returns the Pipeline Endpoint for a specific published pipeline.

        Returns:
            _ (azureml.pipeline.core.PipelineEndpoint) : AML Pipeline Endpoint object
        """
        pipeline_endpoint = PipelineEndpoint.get(workspace=self.ws, name=pipeline_name)
        return pipeline_endpoint

    def submit_pipeline(
        self, pipeline_name: str = None, experiment_name: str = None, pipeline=None
    ):
        """Run a pipeline in AML.

         This method allows the execution of a pipeline already deployed in AML or a pipeline under development.

         To run a pipeline that is not deployed in the environment, just provide the "pipeline" argument.

        Args:
            pipeline_name (str): Name of the pipeline
            experiment_name (str): Name of the experiment related to the pipeline
            pipeline (azureml.pipeline.core.Pipeline): Pipeline type object (if it is necessary to run a pipeline that is not yet deployed)

        Returns:
            _ (azureml.pipeline.core.run.PipelineRun) : PipelineRun instance containing pipeline execution information
        """
        if not pipeline:
            pipeline = PipelineEndpoint.get(workspace=self.ws, name=pipeline_name)

        self.build_amlignore_file()

        pipeline_run = pipeline.submit(experiment_name)
        while pipeline_run.get_status() not in [
            "Finished",
            "Failed",
        ]:
            logger.info(
                f"[!]Experiment: {experiment_name} - Pipeline: {pipeline_name} - Run ID: {pipeline_run.id} not in terminal state"
            )
            sleep(30)

        logger.info(
            f"[!]Experiment: {experiment_name} - Pipeline: {pipeline_name} - Run ID: {pipeline_run.id} - State {pipeline_run.get_status()}"
        )
        return pipeline_run

    def validate_pipeline(self, pipeline=None):
        """Validate pipeline and identify potential errors, such as unconnected inputs.

        This method makes it possible to validate the specification of a pipeline.

        Args:
            pipeline (azureml.pipeline.core.Pipeline): Pipeline type object containing the pipeline implementation to be validated

        Returns:
            _ (list) : List of errors or alerts generated by pipeline validation

        Raise:
            _ (PipelineException) : If the error list is not empty, an error will be thrown preventing the process from continuing
        """
        # validate pipeline
        pipeline_error_list = pipeline.validate()

        if len(pipeline_error_list) > 0:
            logger.info(f"[!]Pipeline errors {pipeline_error_list}.")
            logger.info("[!]Pipeline was NOT published to Azure ML.")
            raise PipelineException(
                error_msg="Pipeline errors found.", error_response=pipeline_error_list
            )

        return pipeline_error_list

    def deploy_pipeline(
        self, pipeline_name=None, pipeline_description="", pipeline=None
    ):
        """This method validates and deploys a pipeline

         If the pipeline already exists, we will update the PipelineEndpoint information otherwise a new PipelineEndpoint is created.

        Args:
            pipeline_name (str): Pipeline name to be deployed
            pipeline_description (str): Summary description of the pipeline purpose
            pipeline (azureml.pipeline.core.Pipeline): Pipeline instance object

        Returns:
            _ (PipelineEndpoint) : Instance of the PipelineEndpoint object containing information for pipeline execution

        Raise:
            _ (PipelineException) : If the error list is not empty, an error will be thrown preventing the process from continuing
        """
        if pipeline is None or pipeline_name is None:
            raise Error("[!]Please, provide the pipeline name and pipeline value")

        self.build_amlignore_file()

        logger.info(f"[!]Deploying pipline {pipeline_name} - {pipeline_description}")

        self.validate_pipeline(pipeline=pipeline)

        if self.VALIDATE_ONLY:
            logger.info("[!]No errors found while validating the pipeline")
            logger.info("[!]Pipeline was NOT published to Azure ML.")
            return None

        # No errors found while validating the pipeline
        # publish the pipeline to AzureML
        logger.info("[!]Publishing the pipeline...")
        published_pipeline = pipeline.publish(
            name=pipeline_name,
            description=pipeline_description,
            continue_on_step_failure=False,
        )

        try:
            pipeline_endpoint = PipelineEndpoint.get(
                workspace=self.ws, name=pipeline_name
            )
            pipeline_endpoint.add_default(pipeline=published_pipeline)
            logger.info("[!]Add as default endpoint.")
        except ErrorResponseException:
            pipeline_endpoint = PipelineEndpoint.publish(
                workspace=self.ws,
                name=pipeline_name,
                description=pipeline_description,
                pipeline=published_pipeline,
            )
            logger.info("[!]Add endpoint.")

        logger.info("[!]Pipeline was published to Azure ML.")
        return pipeline_endpoint

    def get_default_run_config(self, pip_packages=DEFAULT_PIP_PACKAGES):
        """Sets the default package configuration for the pipeline.

        Args:
            pip_packages (str): List of packages to be installed on the AML machine
                                By default, we have already started with the variable DEFAULT_PIP_PACKAGES

        Returns:
            _ (RunConfiguration) : RunConfiguration object instance
        """
        logger.info("On Create Run Config...")
        run_config = RunConfiguration()
        run_config.environment.docker.enabled = True
        run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
        run_config.environment.python.user_managed_dependencies = False
        for pip_pkg in pip_packages:
            logger.info(f"Installing {pip_pkg}...")
            run_config.environment.python.conda_dependencies.add_pip_package(pip_pkg)
        logger.info("Run config done!")
        return run_config

    def get_datastore(self, datastore_name: str = None) -> Datastore:
        """Gets the datastore by it's name.

        Args:
            datastore_name (str, optional): Name of the Datastore. Defaults to None.

        Raises:
            Error: When no name is given an error is raised

        Returns:
            Datastore: Datastore object
        """
        if datastore_name is None:
            raise Error("Please, provide the datastore_name value")

        return Datastore.get(workspace=self.ws, datastore_name=datastore_name)

    def get_config(self, file_config_path="ml/helpers/config.json") -> json:
        """Loads a config json file.

        Args:
            file_config_path (str, optional): Path to the json file. Defaults to "ml/helpers/config.json".

        Returns:
            json: config json file
        """
        with open(file_config_path, "r") as cf:
            return json.load(cf)

    def get_active_models(self):
        """Gets the list of registered models listed on active experiments found in an config json file.

        Returns:
            [Model]: List of registered models of active experiments
        """
        config = self.get_config()
        declared_models = list(
            itertools.chain.from_iterable(
                [exp["models"] for exp in config["experiments"] if exp["active"]]
            )
        )
        models = [
            m for m in Model.list(self.ws, latest=True) if m.name in declared_models
        ]
        return models
