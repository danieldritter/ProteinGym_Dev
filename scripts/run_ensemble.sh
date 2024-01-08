#!/bin/bash

export DATA_FOLDER="../example_data/mutations"
export MSA_FOLDER="../example_data/msas"
# export REFERENCE_FILE="../example_data/DMS_substitutions.csv"
export REFERENCE_FILE="../example_data/DMS_indels.csv"
# export MODEL_CONFIG="../example_configs/site_independent_model.json"
export MODEL_CONFIGS="../example_configs/esm2_8m_model.json ../example_configs/esm2_35m_model.json ../example_configs/site_independent_model.json "
export MODEL_TYPES="esm_model esm_model site_independent_model"
export EXPERIMENT_INDEX=0
# python run_model.py --data_folder=$DATA_FOLDER --alignment_folder=$MSA_FOLDER \
python run_ensemble.py --data_folder=$DATA_FOLDER \
--reference_file=$REFERENCE_FILE --model_configs $MODEL_CONFIGS --experiment_index=$EXPERIMENT_INDEX \
--model_types $MODEL_TYPES --alignment_folder=$MSA_FOLDER

