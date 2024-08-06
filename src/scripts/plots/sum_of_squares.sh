#!/bin/bash

PYSCRIPT="scripts.plots.sum_of_squares"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/sos-npcs}"

for dataset in hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "power" ]
  then
    OTHER_FLAGS="--ylabel"
  elif [ "$dataset" == "gas" ]
  then
    OTHER_FLAGS=""
  elif [ "$dataset" == "hepmass" ]
  then
    OTHER_FLAGS=""
  elif [ "$dataset" == "miniboone" ]
  then
    OTHER_FLAGS="--legend"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --train
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS
done
