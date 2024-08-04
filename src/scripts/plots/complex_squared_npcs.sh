#!/bin/bash

PYSCRIPT="scripts.plots.sos.complex_squared_npcs"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/complex-squared-npcs}"

for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "power" ]
  then
    OTHER_FLAGS="--ylabel"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --train &
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS
done
