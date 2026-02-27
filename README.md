
# Vision Challenge 2026

This repository contains the code and instructions for the Zero-Shot MLLM Agent Challenge, where participants will use the Qwen2.5-VL-7B model to perform vision-language tasks. Follow the setup guide to get started with loading the model and performing inference on your own images and questions

## Public folder

Go to `/leonardo/pub/usertrain/a08trc22/Vision_Challenge` to find this project
on leonardo.

## Github repository

Here's the public Github repository for this project: <https://github.com/bunop/minerva-26-vision-challenge->

## Start an interactive session on leonardo

To start an interactive session on leonardo, use the following command:

```bash
srun -N 1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 \
    -t 00:30:00 -p boost_usr_prod -q boost_qos_dbg \
    -A tra26_minwinsc --pty /bin/bash
```

## Activate the environment

To activate the environment, run the following command in your terminal:

```bash
source activate.sh
```

## Execute the test script

```bash
python setup_and_inference.py
```

## Execute the EVQA test script

Call the `test_evqa.py` script to run the end-to-end inference pipeline for the
EVQA task. The output will be saved to `evqa_test_output.txt`.

```bash
srun -N 1 --ntasks-per-node=1 --cpus-per-task=1 --mem=64gb \
    --gres=gpu:1 -t 00:30:00 -p boost_usr_prod -q boost_qos_dbg \
    -A tra26_minwinsc --pty /bin/bash
source activate.sh
python test_evqa.py | tee evqa_test_output.txt
```