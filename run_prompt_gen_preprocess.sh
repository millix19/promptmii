#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=prompt_gen_v3
#SBATCH --output=logs/prompt_gen_v3_%J.out
#SBATCH --error=logs/prompt_gen_v3_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --mem=512G

export PYTHONUNBUFFERED=1

python prompt_gen_preprocess.py \
    --save_dir /data/user_data/emilyx/prompt_gen/v3_3 \
    --num_train_examples 1000 \
    --num_test_examples 200 \
    --limit_datasets 7000 \
    --chunk_size 50

echo "Prompt generation completed!" 

# Testing
# python prompt_gen_preprocess.py \
#     --save_dir out/test \
#     --num_train_examples 5 \
#     --num_test_examples 5 \
#     --limit_datasets 100 \
#     --chunk_size 5