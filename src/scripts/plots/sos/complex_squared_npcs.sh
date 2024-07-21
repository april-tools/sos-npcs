#!/bin/bash

PYSCRIPT="scripts.plots.sos.complex_squared_npcs"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/complex-squared-npcs.float64}"

for dataset in hepmass miniboone
do
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" --train --ylabel &
  python -m "$PYSCRIPT" "$TBOARD_PATH" "$dataset" --ylabel
done
