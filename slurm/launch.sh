#!/bin/bash

export PROJECT_NAME="sos-npcs"
export PYTHONPATH=${PYTHONPATH:-src}

# These flags need to be updated accordingly:
# EXPS_ID: some identifier for the experiments
# VENV_PATH: the path containing the pip virtual environment
# DATA_PATH: the path containing the data
export EXPS_ID=${EXPS_ID:-exps}
export VENV_PATH=${VENV_PATH:-venv}

# The Slurm partition to use, e.g.,
#PARTITION=PGR-Standard
PARTITION=${PARTITION:-PGR-Standard}
#PARTITION=${PARTITION:-Teach-Standard}
# An optional list of Slurm node to exclude, e.g.,
#EXCL_NODES=${EXCL_NODES:-busynode[01-07]}
EXCL_NODES=${EXCL_NODES:-crannog[01-07],damnii[05-08]}
# The maximum number of parallel jobs to dispatch
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-20}

# Resources and maximum execution time
NUM_CPUS=2
NUM_GPUS=1
TIME=167:59:00
#TIME=79:59:00

JOB_NAME="$PROJECT_NAME-$EXPS_ID"
LOG_DIRECTORY="slurm/logs/$PROJECT_NAME/$EXPS_ID"
LOG_OUTPUT="$LOG_DIRECTORY/%j.out"
EXPS_FILE="$1"
NUM_EXPS=`cat ${EXPS_FILE} | wc -l`

echo "Creating slurm logging directory $LOG_DIRECTORY"
mkdir -p "$LOG_DIRECTORY"

echo "Slurm job settings"
echo "Partition: $PARTITION"
echo "Excl nodes: $EXCL_NODES"

sbatch --job-name $JOB_NAME --output "$LOG_OUTPUT" --error "$LOG_OUTPUT" \
  --partition "$PARTITION" --nodes 1 --ntasks 1 \
  --cpus-per-task $NUM_CPUS --gres=gpu:$NUM_GPUS \
  --time $TIME --exclude="$EXCL_NODES" \
  --array=1-${NUM_EXPS}%${MAX_PARALLEL_JOBS} \
  slurm/run.sh "$EXPS_FILE"

