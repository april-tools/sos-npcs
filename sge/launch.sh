#!/bin/bash

export PROJECT_NAME="sos-npcs"
export PYTHONPATH=${PYTHONPATH:-src}

# These flags need to be updated accordingly:
# EXPS_ID: some identifier for the experiments
# VENV_PATH: the path containing the pip virtual environment
# DATA_PATH: the path containing the data
export EXPS_ID=${EXPS_ID:-exps}
export VENV_PATH=${VENV_PATH:-venv}

# The maximum number of parallel jobs to dispatch
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-20}

# Resources and maximum execution time
NUM_GPUS=1
TIME=167:59:00
#TIME=79:59:00

JOB_NAME="$PROJECT_NAME-$EXPS_ID"
LOG_DIRECTORY="sge/logs/$PROJECT_NAME/$EXPS_ID"
EXPS_FILE="$1"
NUM_EXPS=`cat ${EXPS_FILE} | wc -l`

echo "Creating slurm logging directory $LOG_DIRECTORY"
mkdir -p "$LOG_DIRECTORY"

#echo "SGE job settings"

qsub -N $JOB_NAME -o "$LOG_DIRECTORY" -e "$LOG_DIRECTORY" -cwd \
  -v PROJECT_NAME,VENV_PATH \
  -q gpu -pe gpu-a100 $NUM_GPUS -l h_vmem=16G \
  -l h_rt="$TIME" \
  -t "1-${NUM_EXPS}" -tc $MAX_PARALLEL_JOBS \
  sge/run.sh "$EXPS_FILE"
