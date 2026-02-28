#!/bin/bash
#SBATCH --nodes=4 # nodes
#SBATCH --ntasks-per-node=1 # tasks per node
#SBATCH --cpus-per-task=1 # cores per task
##SBATCH --gres=gpu:4 # GPUs per node
##SBATCH --mem=494000 # mem per node (MB)
#SBATCH --time=00:30:00 # time limit (d-hh:mm:ss)
#SBATCH --account=tra26_minwinsc # account
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --qos=boost_qos_dbg # quality of service

module purge
module load python
module load cuda/11.8

source /leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/pyenvs/MLLM_challenge/bin/activate

srun python test_amber.py
