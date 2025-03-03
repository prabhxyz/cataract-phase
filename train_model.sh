#!/usr/bin/env bash

# In this script, I'm running the training script for the cataract surgery phase recognition model.
# I can specify hyperparameters here or rely on defaults in the Python script.

# Run the training script
python3 src/train.py \
    --data_dir "Cataract-1k-Phase" \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir "checkpoints"
