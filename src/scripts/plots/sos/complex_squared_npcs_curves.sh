#!/bin/bash

PYSCRIPT="scripts.plots.sos.complex_squared_npcs_curves"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/complex-squared-npcs}"
MAX_EPOCHS=300

for dataset in hepmass miniboone
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" --max-epochs $MAX_EPOCHS --ylabel
done
