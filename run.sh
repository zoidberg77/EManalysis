#!/bin/bash
#
#SBATCH --cpus-per-task 16 # number of nodes
#SBATCH --mem 64g # memory pool for all cores
#SBATCH -o emanalysis.out # STDOUT
#SBATCH -e emanalysis.out # STDERR
#SBATCH --gres=gpu:1
/home/sidakk95cs/.conda/envs/EManalysis/bin/python main.py --cfg configs/mouseA2.yaml --mode ptcprep
