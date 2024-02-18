#!/bin/bash

export DATA_FOLDER="../example_data/mutations"
export MSA_FOLDER="../example_data/msas"
export MSA_WEIGHT_FOLDER="../example_data/msa_weights"
export REFERENCE_FILE="../example_data/DMS_substitutions.csv"
# export REFERENCE_FILE="../example_data/DMS_indels.csv"
export MODEL_CONFIGS="../example_configs/esm2_8m_model.json ../example_configs/esm2_35m_model.json ../example_configs/site_independent_model.json ../example_configs/tranception_s_model.json"
export EXPERIMENT_INDEX=20
python run_ensemble.py --data_folder=$DATA_FOLDER --alignment_folder=$MSA_FOLDER --weights_folder $MSA_WEIGHT_FOLDER \
--reference_file=$REFERENCE_FILE --model_configs $MODEL_CONFIGS --experiment_index=$EXPERIMENT_INDEX 

