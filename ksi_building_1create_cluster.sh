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
python scripts/create_cluster_masks.py --config $CONFIG_YAML --dataset_path $DATASET_PATH --output $MASK_PATH --grid_dim $GRID_X $GRID_Y

