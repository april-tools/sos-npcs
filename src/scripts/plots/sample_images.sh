#!/bin/bash

PYSCRIPT="scripts.plots.sampling"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/image-data-sos}"
DEVICE=${DEVICE:-cuda}

for dataset in MNIST
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS \
    "SOS-C_RGqt_R1_K16_KI16_OAdam_LR0.01_BS256" "1723375039.486632" --exp-alias "complex" \
    --complex --num-units 16 --device $DEVICE
done
