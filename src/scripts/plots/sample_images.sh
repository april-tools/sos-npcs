#!/bin/bash

PYSCRIPT="scripts.plots.sampling"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/image-data-sos}"
DEVICE=${DEVICE:-cuda}

for dataset in MNIST
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS \
    "SOS-C_RGqt_R1_K64_KI64_OAdam_LR0.01_BS256" "1723292540.283073" --exp-alias "complex" \
    --complex --num-units 64 --device $DEVICE
done
