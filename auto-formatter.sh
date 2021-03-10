#!/usr/bin/env bash
set -euxo pipefail

# activate the virtual env
. ./venv/bin/activate

PATH_TO_CHECK=$@

# auto adjust file
echo "Adjusting..."
autoflake -r -i --remove-all-unused-imports ${PATH_TO_CHECK}
isort --recursive ${PATH_TO_CHECK}
black ${PATH_TO_CHECK}
   
# validate file
echo "Validating..."
autoflake --check --exclude "venv" -r -i --remove-all-unused-imports ${PATH_TO_CHECK}
isort --check-only --recursive ${PATH_TO_CHECK}
black --exclude "notebooks" --check ${PATH_TO_CHECK}
