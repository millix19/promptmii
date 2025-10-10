#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=prompt_gen_postprocess
#SBATCH --output=logs/prompt_gen_postprocess_%J.out
#SBATCH --error=logs/prompt_gen_postprocess_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

# The output dataset is https://huggingface.co/datasets/milli19/promptmii-dataset

export PYTHONUNBUFFERED=1

# Run the post-processing script
python prompt_gen_postprocess.py \
    --input_dir /data/user_data/emilyx/prompt_gen/v3_3 \
    --output_dir /data/user_data/emilyx/prompt_gen/v3_3_processed \
    --max_configs 10 \
    --validation_ratio 0.1

echo "Post-processing completed!"

# Optional: Test with smaller dataset
# python prompt_gen_postprocess.py \
#     --input_dir out/test \
#     --output_dir out/test_processed \
#     --max_configs 5 \
#     --validation_ratio 0.2 