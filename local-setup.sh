#!/usr/bin/env bash
set -euxo pipefail

# the azure subscription as argument
YOUR_AZURE_SUBSCRIPTION=$1

# install virtualenv on linux (only first run)
sudo apt install virtualenv

# build a virtual environment and activate it
virtualenv -p python3 ./venv

. ./venv/bin/activate

# install the required libs
pip install -r ./requirements.txt

pip install -r ./ml/requirements.txt

az login

az account set --subscription $YOUR_AZURE_SUBSCRIPTION

