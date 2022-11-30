#!/bin/bash
set -ux

if [[ $# == 1 ]]; then
  run_conf=$1
  source ${run_conf}
elif [[ $# > 1 ]]; then
  echo "usage: bash $0 [run_conf]"
  exit -1
fi

mkdir -p ${save_path}

python \
  ./DDPM/scripts/train.py \
  --task ${task} \
  --model ${model} \
  --config_path ${config_path} \
  --vocab_path ${vocab_path} \
  --spm_model_file ${spm_model_file} \
  --save_path ${save_path} \
  --input_file ${input_file}
