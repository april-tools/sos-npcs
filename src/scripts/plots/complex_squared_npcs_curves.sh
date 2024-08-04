#!/bin/bash

PYSCRIPT="scripts.plots.sos.complex_squared_npcs_curves"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/complex-squared-npcs}"
MAX_EPOCHS=200

for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"

  for lr in 0.0001 0.0005 0.001 0.005
  do
    if [ "$dataset" == "power" ]
    then
      OTHER_FLAGS="--ylabel"
    elif [ "$dataset" == "miniboone" ]
    then
      OTHER_FLAGS="--legend"
    else
      OTHER_FLAGS=""
    fi
    if [ "$lr" == "0.0001" ]
    then
      OTHER_FLAGS="$OTHER_FLAGS --title"
    fi
    python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" \
      --learning-rate $lr --max-epochs $MAX_EPOCHS \
      $OTHER_FLAGS
  done
done
