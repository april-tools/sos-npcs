#!/bin/bash

PYSCRIPT="scripts.plots.sampling"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/image-data-sos}"
DEVICE=${DEVICE:-cuda}

for dataset in MNIST
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS \
    "SOS-C_RGqt_R1_K128_KI128_OAdam_LR0.01_BS256" "1723383902.313948" --exp-alias "complex" \
    --complex --num-units 128 --device $DEVICE
done
