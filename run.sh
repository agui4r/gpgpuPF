#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --gres=gpu:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

make

./main

#stdbuf -oL -eL nsys profile --stats true -o report --force-overwrite true ./main
