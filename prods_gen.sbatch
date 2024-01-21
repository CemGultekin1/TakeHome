#!/bin/bash
#SBATCH --job-name=prods
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30GB
#SBATCH --time=1:00:00
#SBATCH --array=1-32
#SBATCH --output=/scratch/cg3306/new_loc/TakeHome/logs/xg_%a.out
#SBATCH --error=/scratch/cg3306/new_loc/TakeHome/logs/xg_%a.err

module purge
singularity exec --nv --overlay overlay-15GB-500K.ext3:ro\
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "\
            source /ext3/source.sh;\
            python learning/gradboost.py $SLURM_ARRAY_TASK_ID;\
            "