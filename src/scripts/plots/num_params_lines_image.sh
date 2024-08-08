#!/bin/bash

PYSCRIPT="scripts.plots.num_params_lines"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/image-data-sos}"

for dataset in MNIST FashionMNIST
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "MNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "FashionMNIST" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal --legend --move-legend-outside"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --metric bpd --train
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --metric bpd
done
