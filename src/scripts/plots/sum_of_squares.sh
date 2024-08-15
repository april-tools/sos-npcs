#!/bin/bash

PYSCRIPT="scripts.plots.sum_of_squares"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/sos-npcs-fixed-size}"


for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "hepmass" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "miniboone" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal --legend --move-legend-outside"
  elif [ "$dataset" == "power" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
    elif [ "$dataset" == "gas" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal --legend --move-legend-outside"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --plot-single-squares
done


for dataset in power gas hepmass miniboone
do
  echo "Processing results relative to data set $dataset"
  if [ "$dataset" == "hepmass" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  elif [ "$dataset" == "miniboone" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal --legend --move-legend-outside"
  elif [ "$dataset" == "power" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
    elif [ "$dataset" == "gas" ]
  then
    OTHER_FLAGS="--ylabel --ylabel-horizontal"
  else
    OTHER_FLAGS=""
  fi
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" $OTHER_FLAGS --plot-single-squares --train
done
