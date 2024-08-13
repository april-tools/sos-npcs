#!/bin/bash

PYSCRIPT="scripts.plots.complex_squared_npcs_curves"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/sos-npcs-fixed-size}"
MAX_EPOCHS=100

for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"

  for lr in 0.0005 0.001 0.005
  do
    if [ "$dataset" == "power" ]
    then
      OTHER_FLAGS="--ylabel --ylabel-horizontal"
    elif [ "$dataset" == "miniboone" ]
    then
      OTHER_FLAGS="--legend"
    else
      OTHER_FLAGS="--ylabel --ylabel-horizontal"
    fi
    if [ "$lr" == "0.0005" ]
    then
      OTHER_FLAGS="$OTHER_FLAGS --title"
    fi
    python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" \
      --learning-rate $lr --max-epochs $MAX_EPOCHS \
      $OTHER_FLAGS
  done
done
