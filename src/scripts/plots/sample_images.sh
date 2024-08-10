#!/bin/bash

PYSCRIPT="scripts.plots.sampling"
CHECKP_PATH="${CHECKP_PATH:-checkpoints/image-data-sos}"
DEVICE=${DEVICE:-cuda}

for dataset in MNIST
do
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" MPC --num-units 128 --device $DEVICE
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS --num-units 128 --device $DEVICE
  python -m "$PYSCRIPT" "$CHECKP_PATH" "$dataset" SOS --num-units 64 --complex --device $DEVICE
done
