#!/bin/bash

PYSCRIPT="scripts.plots.sampling"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/image-data-sos}"
DEVICE=${DEVICE:-cuda}

for dataset in MNIST
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS \
    "SOS_RGqt_R1_K128_KI128_OAdam_LR0.01_BS256" "1723292540.023343" --exp-alias "real" \
    --num-units 128 --device $DEVICE
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS \
    "SOS-C_RGqt_R1_K64_KI64_OAdam_LR0.01_BS256" "1723292540.283073" --exp-alias "complex" \
    --num-units 64 --device $DEVICE
done
