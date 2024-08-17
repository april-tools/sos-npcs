#!/bin/bash

PYSCRIPT="scripts.benchmarks.benchmark"
TBOARD_PATH="${TBOARD_PATH:-tboard-runs/image-data-complete}"
DATA_PATH="${DATA_PATH:-datasets}"

BENCHMARK_FLAGS="--backprop --num-iterations 5 --burn-in-iterations 1 --batch-size 256 --data-path $DATA_PATH"


python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST MPC $BENCHMARK_FLAGS --num-units "16 32 64 128 256 512"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST SOS $BENCHMARK_FLAGS --num-units "16 32 64 128 256 512"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST SOS $BENCHMARK_FLAGS --complex --num-units "8 16 32 64 128 256"
python -m "$PYSCRIPT" "$TBOARD_PATH" MNIST ExpSOS $BENCHMARK_FLAGS --complex --mono-num-units 8 --num-units "8 16 32 64 128 256"

#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA MPC $BENCHMARK_FLAGS --num-units "16 32 64 128 256"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA SOS $BENCHMARK_FLAGS --num-units "16 32 64 128"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA SOS $BENCHMARK_FLAGS --complex --num-units "8 16 32 64 128"
#python -m "$PYSCRIPT" "$TBOARD_PATH" CelebA ExpSOS $BENCHMARK_FLAGS --complex --mono-num-units 8 --num-units "4 8 16 32"
