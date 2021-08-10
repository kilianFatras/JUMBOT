#!/bin/bash -l
#
# Des paramètres sbatch peuvent être spécifiés dans ce fichier sous la forme : #SBATCH <param>
# Les paramètres d'appels sont prioritaires
#
# > man sbatch
#
# Nom du job 
#SBATCH --job-name=OH_UOT_1_sqe
#
# Fichier de sortie d'exécution
#SBATCH --output=office_home_UOT_1_sqe.log
#
#
# Valeurs par défaut : 
#
# --gres=gpu:0 --cpus-per-task=1 --cpus-per-gpu=2 --partition=shortrun --time=30
#
 
conda activate python37
setcuda 10.2

######### FIRST source dataset ##########
python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Art.txt --t_dset_path ./data/office-home/Clipart.txt --batch_size 65 --output_dir "A_C_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Art.txt --t_dset_path ./data/office-home/Product.txt --batch_size 65 --output_dir "A_P_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Art.txt --t_dset_path ./data/office-home/Real_World.txt --batch_size 65 --output_dir "A_R_UOT_1_sqe_clean"

######### SECOND source dataset ##########
python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Clipart.txt --t_dset_path ./data/office-home/Art.txt --batch_size 65 --output_dir "C_A_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Clipart.txt --t_dset_path ./data/office-home/Product.txt --batch_size 65 --output_dir "C_P_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Clipart.txt --t_dset_path ./data/office-home/Real_World.txt --batch_size 65 --output_dir "C_R_UOT_1_sqe_clean"

######### Third source dataset ##########
python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Product.txt --t_dset_path ./data/office-home/Art.txt --batch_size 65 --output_dir "P_A_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Product.txt --t_dset_path ./data/office-home/Clipart.txt --batch_size 65 --output_dir "P_C_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Product.txt --t_dset_path ./data/office-home/Real_World.txt --batch_size 65 --output_dir "P_R_UOT_1_sqe_clean"


######### Fourth source dataset ##########
python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Real_World.txt --t_dset_path ./data/office-home/Art.txt --batch_size 65 --output_dir "R_A_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Real_World.txt --t_dset_path ./data/office-home/Clipart.txt --batch_size 65 --output_dir "R_C_UOT_1_sqe_clean"

python3 /share/home/fatras/MBUOT/domain_adaptation/competitor/jumbot/train_jumbot.py UOT --gpu_id [0,1] --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Real_World.txt --t_dset_path ./data/office-home/Product.txt --batch_size 65 --output_dir "R_P_UOT_1_sqe_clean"