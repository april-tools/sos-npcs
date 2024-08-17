#!/bin/bash

PYSCRIPT="scripts.plots.benchmark"
CSV_PATH="${CSV_PATH:-benchmarks}"

python -m "$PYSCRIPT" "$CSV_PATH" MNIST \
  --ylabel --ylabel-horizontal  \
  --xlabel --legend --move-legend-outside

python -m "$PYSCRIPT" "$CSV_PATH" CelebA \
  --ylabel --ylabel-horizontal \
  --xlabel --legend --move-legend-outside
