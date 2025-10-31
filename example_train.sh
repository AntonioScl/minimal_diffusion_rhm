#!/bin/bash

# Example training script for minimal diffusion RHM
# Train a discrete diffusion model on RHM data

python main.py \
    --num_features 8 \
    --num_classes 8 \
    --num_synonyms 3 \
    --tuple_size 2 \
    --num_layers 3 \
    --seed_rules 42 \
    --train_size 1024 \
    --test_size 1024 \
    --batch_size 64 \
    --print_period 100 \
    --output example_experiment \
