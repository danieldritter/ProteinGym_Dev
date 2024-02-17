#!/bin/bash

export DATA_FOLDER="../example_data/mutations"
export MSA_FOLDER="../example_data/msas"
export MSA_WEIGHT_FOLDER="../example_data/msa_weights"
export REFERENCE_FILE="../example_data/DMS_substitutions.csv"
# export MODEL_CONFIG="../example_configs/site_independent_model.json"
# export MODEL_CONFIG="../example_configs/esm2_8m_model.json"
export MODEL_CONFIG="../example_configs/tranception_s_model.json"
# export MODEL_CONFIG="../example_configs/msa_transformer_model.json"
export EXPERIMENT_INDEX=20
python run_model.py --data_folder=$DATA_FOLDER --alignment_folder=$MSA_FOLDER --weights_folder=$MSA_WEIGHT_FOLDER \
--reference_file=$REFERENCE_FILE --model_config=$MODEL_CONFIG --experiment_index=$EXPERIMENT_INDEX
