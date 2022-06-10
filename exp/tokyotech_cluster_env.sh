# ResNet Model Path
# Ref https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
MODEL_WEIGHTS_PATH="/home/hiroki11x/workspace/MixOT/data/pretrained/resnet50-0676ba61.pth"
export MODEL_WEIGHTS_PATH

# Pyenv VirtualEnv Environment
PYTHON_PATH="/home/hiroki11x/dl/virtualenv_py387/bin"
export PYTHON_PATH

# Dataset Dir (which include office_home and visda)
DATA_DIR_PATH="/mnt/nfs/datasets/OOD"
export DATA_DIR_PATH

# Cluster
CLUSTER_NAME="tokyotech_cluster"
export CLUSTER_NAME

# # ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/11.0
module load cudnn/cuda-11.0/8.0