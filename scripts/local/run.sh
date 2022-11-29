#!/bin/bash
set -ux

if [[ $# != 1 ]]; then
    echo "usage: bash $0 run_conf"
    exit -1
fi

run_conf=$1
source ${run_conf}

export PYTHONPATH=$(dirname "$0")/../..:${PYTHONPATH:-}

${run_script} ${run_conf}