#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J digits.sh
#SBATCH --time=1-00:00:00

# ======== Module, Virtualenv and Other Dependencies ======
source ../../tokyotech_cluster_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

# # ======== Data Copy ========
# t0=$(date +%s)
# cp -r "${DATA_DIR_PATH}/office_home" "$HINADORI_LOCAL_SCRATCH"
# t1=$(date +%s)
# echo "Time for dataset stage-in: $((t1 - t0)) sec"

# ======== Configuration ========
PROGRAM="train.py"
pushd ../../../src/digits
PYTHON_ARGS=$@

# ======== Execution ========
CMD="python ${PROGRAM} ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd