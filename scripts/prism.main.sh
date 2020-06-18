#!/bin/bash

#$ -l mem_free=192G,h_vmem=8G
#$ -pe serial 24
#$ -e /home/maximl/output/calibration.err
#$ -o /home/maximl/output/calibration.out

source ~/venvs/ml/bin/activate

python $HOME/experiments_neurips/main.py -o $HOME/results/experiments_neurips/run1

