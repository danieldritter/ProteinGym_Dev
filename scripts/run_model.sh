#!/bin/bash

export DATA_FOLDER="../example_data/mutations"
export MSA_FOLDER="../example_data/msas"
export REFERENCE_FILE="../example_data/reference_file.csv"
# export MODEL_CONFIG="../example_configs/site_independent_model.json"
export MODEL_CONFIG="../example_configs/esm2_8m_model.json"
export MODEL_NAME="esm_model"
export EXPERIMENT_INDEX=0
# python run_model.py --data_folder=$DATA_FOLDER --alignment_folder=$MSA_FOLDER \
python run_model.py --data_folder=$DATA_FOLDER \
--reference_file=$REFERENCE_FILE --model_config=$MODEL_CONFIG --experiment_index=$EXPERIMENT_INDEX \
--model_name=$MODEL_NAME
