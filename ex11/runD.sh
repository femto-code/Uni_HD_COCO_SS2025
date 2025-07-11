#!/bin/bash

#SBATCH --job-name=cc11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:00:00
#SBATCH --partition=exercise-hpdc


spack env activate cuda
spack load cuda@12.1.1
nvcc -O2 -o versionD versionD.cu
srun ./versionD