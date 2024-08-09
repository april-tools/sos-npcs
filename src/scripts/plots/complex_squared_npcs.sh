#!/bin/bash

PYSCRIPT="scripts.plots.complex_squared_npcs"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/sos-npcs}"

for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "hepmass" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "miniboone" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  else
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --train
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS
done
