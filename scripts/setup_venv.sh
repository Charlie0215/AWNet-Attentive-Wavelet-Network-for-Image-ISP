#!/bin/bash

set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

python3 -m venv $ROOT_DIR/venvs/awnet

source $ROOT_DIR/venvs/awnet/bin/activate

pip intall -r $ROOT_DIR/resources/requirements.txt

echo "$ROOT_DIR" >> artifacts/venv/lib/python3.9/site-packages/awnet.pth
