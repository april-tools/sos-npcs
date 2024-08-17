#!/bin/bash

PYSCRIPT="scripts.benchmarks.benchmark"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/image-data-complete}"


BENCHMARK_FLATES="--num-iterations 5 --burn-in-iterations 1 --batch-size 256"


python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST MPC --num-units "16 32 64 128 256 512"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST SOS --num-units "16 32 64 128 256 512"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST SOS --complex --num-units "8 16 32 64 128 256"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST ExpSOS --complex --mono-num-units 8 --num-units "8 16 32 64 128 256"

#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA MPC --num-units "16 32 64 128 256"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA SOS --num-units "16 32 64 128"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA SOS --complex --num-units "8 16 32 64 128"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA ExpSOS --complex --mono-num-units 8 --num-units "4 8 16 32"
