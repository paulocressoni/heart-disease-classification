#!/usr/bin/env bash
set -eux pipefail

sudo apt install jq

RESOURCE_GROUP_NAME="general-resource"
SUBSCRIPTION_ID=$(az account show | jq -cr .id)
TENANT_ID=$(az account show | jq -cr .homeTenantId)

DOTENV_FILE_PATH=".env"

echo $"ENV=dev" > ".env"
echo $"RESOURCE_GROUP_NAME=${RESOURCE_GROUP_NAME}" >> ${DOTENV_FILE_PATH}
echo $"SUBSCRIPTION_ID=${SUBSCRIPTION_ID}" >> ${DOTENV_FILE_PATH}
echo $"TENANT_ID=${TENANT_ID}" >> ${DOTENV_FILE_PATH}
echo $"AML_WORKSPACE_NAME=aml-heart-disease" >> ${DOTENV_FILE_PATH}
echo $"STORAGE_ACCOUNT_NAME=my-storage" >> ${DOTENV_FILE_PATH}
echo $"KV_SERVICE_PRINCIPAL_SECRET_NAME=amlheartdiseas2396477080" >> ${DOTENV_FILE_PATH}
echo $"AML_VNET_NAME=" >> ${DOTENV_FILE_PATH}
echo $"AML_SUBNET_NAME=" >> ${DOTENV_FILE_PATH}
