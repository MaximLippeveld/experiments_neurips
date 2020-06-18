#!/bin/bash

#PBS -l nodes=1:ppn=36
#PBS -l walltime=10:00:00

module load Python/3.7
source $VSC_DATA/venv/bin/activate

python $VSC_HOME/experiments_neurips/main.py -o ~/vsc_data_vo/results/dirichletcal/run1
