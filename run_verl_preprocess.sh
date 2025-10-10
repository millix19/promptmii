#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=verl_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --time=6:00:00
#SBATCH --output=logs/verl_preprocess_%j.out
#SBATCH --error=logs/verl_preprocess_%j.err

# Run the VERL data preprocessing script with configurable training example ratios

# V8 script with multi-pass training example ratios, same subset of test examples each pass
# Pass 1: 100% of datasets get 5 examples 
# Pass 2: 30% of datasets get additional 10 examples
# Pass 3: 20% of datasets get additional 20 examples 
# Pass 4: 10% of datasets get additional 50 examples

python examples/data_preprocess/prompt_gen_v8.py \
    --input_dir /data/group_data/prompt-gen/v3_3_processed \
    --output_dir /data/group_data/prompt-gen/verl_preprocess_v8_multipass_test20 \
    --train_example_ratios "5:1.0,10:0.30,20:0.20,50:0.10" \
    --balance_test_labels disabled \
    --balance_train_labels disabled \
    --num_test_examples 20 \
    --seed 42

# Run with qwen meta prompt template
python examples/data_preprocess/prompt_gen_v8.py \
    --input_dir /data/group_data/prompt-gen/v3_3_processed \
    --output_dir /data/group_data/prompt-gen/verl_preprocess_v8_multipass_meta3_test20 \
    --train_example_ratios "5:1.0,10:0.30,20:0.20,50:0.10" \
    --meta_prompt_template v7 \
    --balance_test_labels disabled \
    --balance_train_labels disabled \
    --num_test_examples 20 \
    --seed 42

# Testing
# python examples/data_preprocess/prompt_gen_v8.py \
#     --input_dir /data/user_data/emilyx/prompt_gen/v3_3_processed \
#     --output_dir /data/group_data/prompt-gen/test/verl_preprocess_v8_multipass_test2 \
#     --train_example_ratios "5:1.0,10:0.30,20:0.20,50:0.10" \
#     --num_test_examples 2 \
#     --max_datasets 5 \
#     --seed 42
