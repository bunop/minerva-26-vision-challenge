# Activate all stuff in the environment
module purge
module load python
module load cuda

export HF_HUB_CACHE="./MLLM_challenge/hf_models"
export HF_HOME="./MLLM_challenge/hf_models"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

source MLLM_challenge/pyenvs/MLLM_challenge/bin/activate
