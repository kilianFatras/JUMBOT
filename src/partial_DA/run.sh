#!/bin/bash -l
#
# Des paramètres sbatch peuvent être spécifiés dans ce fichier sous la forme : #SBATCH <param>
# Les paramètres d'appels sont prioritaires
#
# > man sbatch
#
# Nom du job 
#SBATCH --job-name=PDA_OH_UOT_all
#
# Fichier de sortie d'exécution
#SBATCH --output=results/OH_PDA_UOT_reproduced.log
#
# Autres paramètres utiles :
# - réservation de gpus : --gres=gpu:[1-4]
# - réservation de cpus : --cpus-per-task, -c [1-n]
# - choix du type de GPU : --constraint titan|2080ti
# - date/heure de démarrage : --begin, -b <date> (ex: 20200801, 20:00, now+2d ...) 
# - date/heure deadline : --deadline <date>
# - positionnement du répertoire : --chdir, -d <dir>
# - redirection d'entrée : --input, -i <file>
# - nom du job : --job-name, -J <jobname>
# - notifications mail : --mail-type=BEGIN,FAIL,END,TIME_LIMIT_80
# - spécification adresse mail : --mail-user=mailaddress
# - redirections de sortie : --output, -o <file> --open-mode=append|truncate
# - partition : --partition, -p shortrun|longrun
# - durée maximale : --time, -t [HH:]MM
# - vérification sans lancement : --test-only
#
# Valeurs par défaut : 
#
# --gres=gpu:0 --cpus-per-task=1 --cpus-per-gpu=2 --partition=shortrun --time=30
#
 
conda activate python37
setcuda 10.2

python run_JUMBOT.py --s 0 --t 1 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 0 --t 2 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 0 --t 3 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 1 --t 0 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 1 --t 2 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 1 --t 3 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 2 --t 0 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 2 --t 1 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 2 --t 3 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 3 --t 0 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 3 --t 1 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
python run_JUMBOT.py --s 3 --t 2 --dset office_home --net ResNet50 --output reproduced_uot --gpu_id [0,1] 
