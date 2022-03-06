#!/bin/bash

# conda env create -f environment.yml
# conda activate mega-nerf

# shell script variable
PATH_PWD=$PWD
DATASET_NAME="building"

CONFIG_YAML=${PATH_PWD}/configs/mega-nerf/${DATASET_NAME}.yaml
DATASET_PATH=${PATH_PWD}/../ksi_data/mega-nerf/building
MASK_PATH=${DATASET_PATH}/output_mask

GRID_X=1
GRID_Y=$GRID_X

#
python mega_nerf/train.py --config_file $CONFIG_YAML --exp_name $EXP_PATH --dataset_path $DATASET_PATH --chunk_paths $SCRATCH_PATH --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX}