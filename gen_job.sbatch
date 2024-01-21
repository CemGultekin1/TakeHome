#!/bin/bash
#SBATCH --job-name=gencod2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/cg3306/new_loc/TakeHome/logs/gen2_%a.out
#SBATCH --error=/scratch/cg3306/new_loc/TakeHome/logs/gen2_%a.err
module purge
singularity exec --nv --overlay overlay-15GB-500K.ext3:ro\
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "\
            source /ext3/source.sh;\
            python learning/genetic.py 4 2;\
            "