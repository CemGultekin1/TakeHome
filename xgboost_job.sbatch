#!/bin/bash
#SBATCH --job-name=xgboost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --output=xg.out
#SBATCH --error=xg.err
#SBATCH --time=18:00:00

module purge
singularity exec --nv --overlay overlay-15GB-500K.ext3:ro\
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "\
            source /ext3/source.sh;\
            python xgboost_learning/hyperparam.py;\
            "