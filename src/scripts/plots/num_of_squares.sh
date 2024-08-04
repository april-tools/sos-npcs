#!/bin/bash

PYSCRIPT="scripts.plots.sos.num_of_squares"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/num-of-squares-1-to-n}"

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
