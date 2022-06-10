# ======== Configuration ========
WANDB_ENTITY='rlopt'
WANDB_PROJECT_NAME='debug_oh_ot_study'

METHOD="JUmbOT"
MODEL_ARCH="ResNet50"
DATA_SET="office_home"
TEST_INTERVAL="1000"
BATCH_SIZE="65"
# ======== Loop  ========

SHELL_ARGS="${METHOD}\
            --wandb_entity ${WANDB_ENTITY} \
            --wandb_project_name ${WANDB_PROJECT_NAME} \
            --net ${MODEL_ARCH} \
            --dset ${DATA_SET} \
            --test_interval ${TEST_INTERVAL}\
            --batch_size ${BATCH_SIZE}
            "

CMD="ybatch run.sh ${SHELL_ARGS}"
echo $CMD
eval $CMD
