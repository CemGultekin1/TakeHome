#!/bin/bash
#SBATCH --job-name=xgboost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --array=2,4,6,8,10,12,14,16,18,20
#SBATCH --output=/scratch/cg3306/new_loc/TakeHome/logs/xg_%a.out
#SBATCH --error=/scratch/cg3306/new_loc/TakeHome/logs/xg_%a.err

module purge
singularity exec --nv --overlay overlay-15GB-500K.ext3:ro\
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "\
            source /ext3/source.sh;\
            python xgboost_learning/hyperparam.py $SLURM_ARRAY_TASK_ID;\
            "