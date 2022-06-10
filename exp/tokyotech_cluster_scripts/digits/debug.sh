# ======== Configuration ========
WANDB_ENTITY='rlopt'
WANDB_PROJECT_NAME='debug_digits_ot_study'
# ======== Loop  ========

SHELL_ARGS="--wandb_entity ${WANDB_ENTITY} \
            --wandb_project_name ${WANDB_ENTITY}
            "

CMD="ybatch run.sh ${SHELL_ARGS}"
echo $CMD
eval $CMD
