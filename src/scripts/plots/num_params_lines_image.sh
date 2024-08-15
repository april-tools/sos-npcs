#!/bin/bash

PYSCRIPT="scripts.plots.num_params_lines"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/image-data-complete}"
MODELS="${MODELS:-"MPC;SOS;ExpSOS"}"
DATASETS="${DATASETS:-"MNIST FashionMNIST CelebA"}"

for dataset in $DATASETS
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "MNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "FashionMNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "CelebA" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal --legend --move-legend-outside --xlabel"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --models "$MODELS" --metric bpd
done


for dataset in $DATASETS
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "MNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "FashionMNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "CelebA" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --metric bpd --train
done
