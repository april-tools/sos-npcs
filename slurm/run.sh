#!/bin/bash

# Find a suitable scratch directory
SCRATCH_DIR="/disk/scratch_big"
if [ ! -w "$SCRATCH_DIR" ]
then
	SCRATCH_DIR="/disk/scratch"
fi
SCRATCH_DIR="$SCRATCH_DIR/$USER"

echo "Running job on partition '$SLURM_JOB_PARTITION' and node '$SLURMD_NODENAME'"
echo "Using scratch directory '$SCRATCH_DIR'"

echo "Copying datasets to the scratch directory ..."

SCRATCH_DATA_PATH="$SCRATCH_DIR/datasets"
mkdir -p "$LOCAL_DATA_PATH" || exit 1
rsync -r -a --update --compress --progress "$DATA_PATH/" "$SCRATCH_DATA_PATH"

echo "Setting up the result and checkpoints directories ..."

DESTINATION_PATH="$HOME/$PROJECT_NAME"
RESULTS_PATH="$SCRATCH_DIR/$SLURM_JOB_ID"
TBOARD_DIR="$RESULTS_PATH/tboard-runs/$EXPS_ID"
CHECKP_DIR="$RESULTS_PATH/checkpoints/$EXPS_ID"
DEST_TBOARD_DIR="$DESTINATION_PATH/tboard-runs"
DEST_CHECKP_DIR="$DESTINATION_PATH/checkpoints"

# Create local directories where to save model checkpoints and tensorboard logs
mkdir -p "$RESULTS_PATH" || exit 1
mkdir -p "$DEST_TBOARD_DIR" || exit 1
mkdir -p "$DEST_CHECKP_DIR" || exit 1

echo "Starting job ..."

# Get the command to run
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`"

# Activate the virtual environment, and launch the command.
# Then, rsync is used to copy the model checkpoints and tensorboard logs from the
# nodes local disk to the shared disk.
source "$VENV_PATH/bin/activate" || exit 1
$COMMAND --device cuda --data-path "$LOCAL_DATA_PATH" --tboard-path "$TBOARD_DIR" --checkpoint-path "$CHECKP_DIR" || exit 1
rsync -r -a --verbose --ignore-existing "$TBOARD_DIR" "$DEST_TBOARD_DIR/"
rsync -r -a --verbose --ignore-existing "$CHECKP_DIR" "$DEST_CHECKP_DIR/"

# Cleanup before exiting
rm -rf "$RESULTS_PATH"
rmdir "$SCRATCH_DIR"
