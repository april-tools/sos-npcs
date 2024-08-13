#!/bin/bash

PYSCRIPT="scripts.plots.num_params_lines"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/uci-data-exp-sos}"

for dataset in hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "hepmass" ]
  then
    OTHER_FLAGS="--ylabel"
  elif [ "$dataset" == "miniboone" ]
  then
    OTHER_FLAGS="--legend"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --train &
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS
done
