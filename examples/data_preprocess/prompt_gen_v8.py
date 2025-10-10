#!/usr/bin/env python3
"""
Downstream usage script for VERL training preprocessing.
Step 3: Convert post-processed prompt generation data to VERL training format.

This script:
1. Loads the train/validation/test splits from prompt_gen_postprocess.py
2. Reformats data into VERL training format using the best meta prompt template (v3)
3. Uses multi-pass approach to create training examples with varying context counts
4. Saves to train/validation/test parquet files

Multi-pass approach:
- Pass 1: All datasets get 1 training example with 5 context examples
- Pass 2: 30% of datasets get additional training example with 10 context examples  
- Pass 3: 20% of datasets get additional training example with 20 context examples
- Pass 4: 10% of datasets get additional training example with 50 context examples
"""

import argparse
import os
import random
from datasets import load_from_disk, Dataset
from verl.utils.hdfs_io import makedirs
import json
from collections import defaultdict


def parse_train_example_ratios(ratios_str):
    """
    Parse training example ratios string into a dictionary.
    
    Args:
        ratios_str: String like "5:1.0,10:0.30,20:0.20,50:0.10"
    
    Returns:
        dict: {example_count: probability} e.g., {5: 1.0, 10: 0.30, 20: 0.20, 50: 0.10}
    """
    if not ratios_str:
        return {}
    
    ratios = {}
    for pair in ratios_str.split(','):
        if ':' not in pair:
            raise ValueError(f"Invalid ratio format: {pair}. Expected format: 'count:probability'")
        
        count_str, prob_str = pair.split(':', 1)
        try:
            count = int(count_str.strip())
            prob = float(prob_str.strip())
            ratios[count] = prob
        except ValueError as e:
            raise ValueError(f"Invalid ratio format: {pair}. Count must be int, probability must be float. Error: {e}")
    
    return ratios


def format_examples(dataset_samples, label_names):
    """Format dataset samples for meta prompt insertion."""
    formatted = []
    for s in dataset_samples:
        assert isinstance(s, dict), f"Sample is not a dict: {s}"
        assert 'label' in s, f"Sample missing 'label' key: {s}"
        
        # Handle case where label is already a string (not an index)
        if isinstance(s["label"], str):
            label_str = s["label"]
        else:
            # Handle case where label is an integer index (NEVER)
            label_str = label_names[s["label"]]
        
        formatted.append(f'Text: "{s["text"]}"\nLabel: {label_str}\n\n')
    return "".join(formatted)


def format_meta_prompt_v3(dataset_samples, label_names):
    """Format the meta prompt using the v3 template from baseline_eval_v2_llama3.py."""
    examples_str = format_examples(dataset_samples, label_names)
    label_names_str = ", ".join(label_names)
    
    # v3 template from baseline_eval_v2_llama3.py
    meta_prompt = (
        "You are helping to create a prompt for a language model to classify text inputs. The model should choose one label from the following options: {label_names}.\n\n"
        "Here are some example inputs and their correct labels:\n{examples}\n\n"
        "Write an instruction that:\n"
        "- Describes the classification task in a way that generalizes to new inputs.\n"
        "- Points out any useful clues or strategies for making the decision.\n"
        "- Clearly tells the model to respond with only the label name, and not to include any explanation or additional text.\n\n"
        "Provide only the instruction, not the examples or labels."
    ).format(examples=examples_str, label_names=label_names_str)
    
    return meta_prompt


def format_meta_prompt_v7(dataset_samples, label_names):
    """Format the meta prompt using the v7 template from baseline_eval_v3_modular.py."""
    examples_str = format_examples(dataset_samples, label_names)
    label_names_str = ", ".join(label_names)
    
    # v7 template from baseline_eval_v3_modular.py
    meta_prompt = (
        "You are designing a clear instruction for a data annotator to classify text inputs into one of these labels: {label_names_str}\n\n"
        "Here are some example inputs and their correct labels:\n{examples}\n\n"
        "Your task is to write a concise instruction that:\n"
        "- Defines the classification task and clearly explains the meaning of each label.\n"
        "- Provides general labeling strategies and decision rules so annotators can correctly handle unseen inputs.\n"
        "- Highlights common pitfalls, tricky edge cases, and misconceptions to reduce labeling errors.\n"
        "- Keeps the instruction reasonably concise and focused — avoid unnecessary repetition or overly long explanations.\n\n"
    ).format(examples=examples_str, label_names_str=label_names_str)
    
    return meta_prompt


def subsample_examples(examples, num_examples):
    """Subsample examples to a specified number."""
    if len(examples) <= num_examples:
        return examples
    return random.sample(examples, num_examples)


def subsample_examples_balanced_by_label(examples, num_examples, label_names):
    """Subsample examples to balance label coverage as evenly as possible.
    
    Strategy:
    1. Bucket examples by label
    2. Allocate floor(num_examples / num_labels) per label (capped by availability)
    3. Distribute the remaining examples round-robin over labels with remaining capacity
    """
    def _normalize_label(label_value, label_names):
        """Return label string for either string or index label fields."""
        if isinstance(label_value, str):
            return label_value
        return label_names[label_value]

    if len(examples) <= num_examples:
        return examples
    
    # Bucket by normalized label
    label_to_examples = defaultdict(list)
    for ex in examples:
        label_str = _normalize_label(ex.get("label"), label_names)
        label_to_examples[label_str].append(ex)
    
    labels = sorted(label_to_examples.keys())
    if not labels:
        return random.sample(examples, num_examples)
    
    num_labels = len(labels)
    base_per_label = num_examples // num_labels
    
    selected = []
    remaining_pool_by_label = {}
    
    # allocate base_per_label
    for label in labels:
        bucket = label_to_examples[label]
        take = min(base_per_label, len(bucket))
        if take > 0:
            chosen = random.sample(bucket, take)
            selected.extend(chosen)
        remaining_pool_by_label[label] = [ex for ex in bucket if ex not in selected]
    
    # Collect deficits where a label had fewer than base_per_label
    selected_count = len(selected)
    needed = max(0, num_examples - selected_count)
    
    if needed > 0:
        # round-robin order preferring labels
        labels_by_capacity = sorted(labels, key=lambda l: len(remaining_pool_by_label[l]), reverse=True)
        idx = 0
        while needed > 0 and any(len(remaining_pool_by_label[l]) > 0 for l in labels_by_capacity):
            label = labels_by_capacity[idx % len(labels_by_capacity)]
            if remaining_pool_by_label[label]:
                ex = random.choice(remaining_pool_by_label[label])
                remaining_pool_by_label[label].remove(ex)
                selected.append(ex)
                needed -= 1
            idx += 1
    
    # If still short (all buckets exhausted), fall back to random over full set excluding already selected
    if len(selected) < num_examples:
        already = set(id(ex) for ex in selected)
        remaining_global = [ex for ex in examples if id(ex) not in already]
        if remaining_global:
            fill = min(num_examples - len(selected), len(remaining_global))
            selected.extend(random.sample(remaining_global, fill))
    
    # If we somehow overshot, but I think this shouldn't happen
    if len(selected) > num_examples:
        selected = random.sample(selected, num_examples)
    
    return selected

def process_to_verl_format(example, subset_idx, num_train_examples=5, num_test_examples=5, meta_prompt_template="v3", balance_train_labels=False):
    """Convert a post-processed example to VERL training format."""
    # Extract data from the post-processed format
    dataset_name = example['dataset_name']
    config_name = example.get('config_name', '')
    split_name = example.get('split_name', '')
    train_examples = example['train_examples']
    test_examples = example['test_examples']
    label_names = example['label_names']
    
    # Get index if available, otherwise use dataset_name as identifier
    idx = example.get('index', dataset_name)
    
    # Subsample examples
    if balance_train_labels:
        train_examples = subsample_examples_balanced_by_label(train_examples, num_train_examples, label_names)
    else:
        train_examples = subsample_examples(train_examples, num_train_examples)
    
    # Format meta prompt using selected template
    if meta_prompt_template == "v7":
        meta_prompt = format_meta_prompt_v7(train_examples, label_names)
    else:  # default to v3
        meta_prompt = format_meta_prompt_v3(train_examples, label_names)
    
    # Create unique identifier for this subset
    subset_id = f"{dataset_name}_{config_name}_{split_name}_subset_{subset_idx}"
    
    # Return in VERL format
    return {
        "data_source": "hf-text-classification",
        "prompt": [
            {
                "role": "user",
                "content": meta_prompt,
            }
        ],
        "ability": "prompt-generation",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "split_name": split_name,
            "subset_id": subset_id,
            "subset_index": subset_idx,
            "input_examples": train_examples,
            "test_examples": test_examples,
            "label_mapping": label_names,
            "index": idx,
            "train_examples_count": num_train_examples,
            "meta_prompt_template": meta_prompt_template,
        },
    }


def process_split_multi_pass(dataset, train_example_ratios, num_test_examples=5, max_datasets=None, seed=42, meta_prompt_template="v3", fanout=1, balance_test_labels=True, balance_train_labels=False):
    """Process a dataset split using multi-pass approach.
    
    Args:
        dataset: Dataset to process
        train_example_ratios: Dict mapping example counts to probabilities
        num_test_examples: Number of test examples to use
        max_datasets: Maximum number of datasets to process
        seed: Random seed for reproducibility
        meta_prompt_template: Template version to use
        fanout: Number of different samples to generate per dataset per pass
        balance_test_labels: If True, balance label distribution in test examples
        balance_train_labels: If True, balance label distribution in train examples
    """
    print(f"Processing {len(dataset)} examples using multi-pass approach...")
    
    # Limit number of rows if specified
    if max_datasets is not None:
        print(f"before: {len(dataset)}")
        dataset = dataset.select(range(min(max_datasets, len(dataset))))
        print(f"Limited to {len(dataset)} rows for processing")
    
    # Set seed for reproducible assignments
    random.seed(seed)
    
    # Pre-sample test examples for each dataset to ensure consistency across passes
    print("Pre-sampling test examples for consistency across passes...")
    dataset_with_test_examples = []
    for row in dataset:
        # Sample test examples once per dataset
        if balance_test_labels:
            test_examples = subsample_examples_balanced_by_label(row['test_examples'], num_test_examples, row['label_names'])
        else:
            test_examples = subsample_examples(row['test_examples'], num_test_examples)
        # Create a copy of the row with the sampled test examples
        row_copy = row.copy()
        row_copy['test_examples'] = test_examples
        dataset_with_test_examples.append(row_copy)
    
    # Track statistics
    stats = defaultdict(int)
    processed_data = []
    
    # Sort ratios by example count for logical ordering
    sorted_ratios = sorted(train_example_ratios.items())
    
    # Process each pass
    for pass_idx, (example_count, probability) in enumerate(sorted_ratios):
        print(f"\nPass {pass_idx + 1}: Processing with {example_count} context examples (probability: {probability})")
        
        # Each pass processes a subset of datasets based on probability
        num_datasets_to_process = int(len(dataset_with_test_examples) * probability)
        datasets_to_process = random.sample(dataset_with_test_examples, num_datasets_to_process)
        print(f"  Processing {len(datasets_to_process)} datasets (selected {probability*100:.1f}%)")
        
        # Process each selected dataset with fanout
        for row_idx, row in enumerate(datasets_to_process):
            dataset_name = row['dataset_name']
            config_name = row.get('config_name', '')
            split_name = row.get('split_name', '')
            
            # Generate multiple fanout copies for each dataset
            for fanout_idx in range(fanout):
                try:
                    # Create unique subset index that includes pass and fanout information
                    unique_subset_idx = f"pass_{pass_idx}_{row_idx}_fanout_{fanout_idx}"
                    
                    # Create a copy of the row for this fanout to ensure independent sampling
                    row_copy = row.copy()
                    
                    processed_example = process_to_verl_format(
                        row_copy, 
                        unique_subset_idx,
                        num_train_examples=example_count,
                        num_test_examples=num_test_examples,
                        meta_prompt_template=meta_prompt_template,
                        balance_train_labels=balance_train_labels,
                    )
                    
                    # Add pass and fanout information to metadata
                    processed_example['extra_info']['pass_index'] = pass_idx
                    processed_example['extra_info']['pass_example_count'] = example_count
                    processed_example['extra_info']['fanout_index'] = fanout_idx
                    processed_example['extra_pass_probability'] = probability
                    
                    processed_data.append(processed_example)
                    stats[example_count] += 1
                    
                except Exception as e:
                    print(f"Error processing pass {pass_idx}, dataset {row_idx+1}, fanout {fanout_idx} ({dataset_name}): {e}")
                    continue
        
        print(f"  Pass {pass_idx + 1} completed: {len(datasets_to_process) * fanout} training examples created")
    
    print(f"\nSuccessfully processed {len(processed_data)} total training examples")
    print(f"Training example distribution: {dict(stats)}")
    
    # Calculate expected vs actual dataset size
    original_size = len(dataset)
    final_size = len(processed_data)
    expansion_ratio = final_size / original_size
    print(f"Dataset size: {original_size} → {final_size} (expansion ratio: {expansion_ratio:.2f}x)")
    
    return Dataset.from_list(processed_data)


def save_verl_datasets(train_dataset, val_dataset, test_dataset, output_dir, args=None, train_example_ratios=None):
    """Save the VERL format datasets to parquet files and CSV for debugging."""
    print(f"Saving VERL datasets to {output_dir}...")
    
    makedirs(output_dir, exist_ok=True)
    
    # Save to parquet files
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))
    
    # Save to CSV files for easy debugging
    print("Saving CSV files for debugging...")
    
    def save_csv_with_expanded_columns(dataset, filename):
        """Save dataset to CSV with extra_info columns expanded for readability."""
        if len(dataset) == 0:
            print(f"Warning: {filename} has no data to save")
            return
        
        # Convert to pandas DataFrame
        import pandas as pd
        df = pd.DataFrame(dataset)
        
        # Expand extra_info into separate columns
        if 'extra_info' in df.columns:
            # Extract key fields from extra_info
            df['dataset_name'] = df['extra_info'].apply(lambda x: x.get('dataset_name', '') if isinstance(x, dict) else '')
            df['config_name'] = df['extra_info'].apply(lambda x: x.get('config_name', '') if isinstance(x, dict) else '')
            df['split_name'] = df['extra_info'].apply(lambda x: x.get('split_name', '') if isinstance(x, dict) else '')
            df['subset_id'] = df['extra_info'].apply(lambda x: x.get('subset_id', '') if isinstance(x, dict) else '')
            df['subset_index'] = df['extra_info'].apply(lambda x: x.get('subset_index', '') if isinstance(x, dict) else '')
            df['index'] = df['extra_info'].apply(lambda x: x.get('index', '') if isinstance(x, dict) else '')
            df['train_examples_count'] = df['extra_info'].apply(lambda x: x.get('train_examples_count', '') if isinstance(x, dict) else '')
            df['pass_index'] = df['extra_info'].apply(lambda x: x.get('pass_index', '') if isinstance(x, dict) else '')
            df['pass_example_count'] = df['extra_info'].apply(lambda x: x.get('pass_example_count', '') if isinstance(x, dict) else '')
            df['fanout_index'] = df['extra_info'].apply(lambda x: x.get('fanout_index', '') if isinstance(x, dict) else '')
            
            # Extract input_examples, test_examples, and label_mapping
            df['input_examples'] = df['extra_info'].apply(lambda x: str(x.get('input_examples', '')) if isinstance(x, dict) else '')
            df['test_examples'] = df['extra_info'].apply(lambda x: str(x.get('test_examples', '')) if isinstance(x, dict) else '')
            df['label_mapping'] = df['extra_info'].apply(lambda x: str(x.get('label_mapping', '')) if isinstance(x, dict) else '')
            
            # Drop the original extra_info column
            df = df.drop('extra_info', axis=1)
        
        # Extract prompt content as context
        if 'prompt' in df.columns:
            df['context'] = df['prompt'].apply(lambda x: x[0]['content'] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and 'content' in x[0] else str(x))
            # Drop the original prompt column
            df = df.drop('prompt', axis=1)
        
        # Reorder columns for better readability
        column_order = [
            'dataset_name', 'config_name', 'split_name', 'subset_id', 'subset_index', 'index',
            'train_examples_count', 'pass_index', 'pass_example_count', 'fanout_index', 'context', 
            'input_examples', 'test_examples', 'label_mapping', 'data_source', 'ability'
        ]
        
        # Only include columns that exist in the dataframe
        final_columns = [col for col in column_order if col in df.columns]
        df = df[final_columns]
        
        # Save to CSV in the output directory
        csv_path = os.path.join(output_dir, filename)
        df.to_csv(csv_path, index=False, escapechar='\\')
        print(f"Saved {filename} to {csv_path} with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    
    save_csv_with_expanded_columns(train_dataset, "train_debug.csv")
    save_csv_with_expanded_columns(val_dataset, "validation_debug.csv")
    save_csv_with_expanded_columns(test_dataset, "test_debug.csv")
    
    # Save metadata
    metadata = {
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "test_examples": len(test_dataset),
        "total_examples": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "format": "verl-training",
        "meta_prompt_template": "v3",  # Default, will be updated below
        "approach": "multi-pass",
        "train_example_ratios": train_example_ratios or {},
        "max_datasets": None
    }
    
    # Update metadata with actual values if available
    if args is not None:
        if hasattr(args, 'max_datasets'):
            metadata['max_datasets'] = args.max_datasets
        if hasattr(args, 'meta_prompt_template'):
            metadata['meta_prompt_template'] = args.meta_prompt_template
        if hasattr(args, 'balance_test_labels'):
            metadata['balance_test_labels'] = args.balance_test_labels
        if hasattr(args, 'balance_train_labels'):
            metadata['balance_train_labels'] = args.balance_train_labels
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved VERL datasets to {output_dir}")
    print(f"Metadata: {metadata}")


def main():
    parser = argparse.ArgumentParser(description="Convert post-processed prompt generation data to VERL training format using multi-pass approach")
    parser.add_argument("--input_dir", required=True, help="Input directory containing train/validation/test splits from prompt_gen_postprocess.py")
    parser.add_argument("--output_dir", required=True, help="Output directory for VERL format datasets")
    parser.add_argument("--train_example_ratios", type=str, required=True, help="Training example ratios in format 'count:prob,count:prob' (e.g., '5:1.0,10:0.30,20:0.20,50:0.10')")
    parser.add_argument("--num_test_examples", type=int, default=5, help="Number of test examples to subsample")
    parser.add_argument("--max_datasets", type=int, default=None, help="Maximum number of datasets to process (useful for testing)")
    parser.add_argument("--meta_prompt_template", type=str, default="v3", choices=["v3", "v7"], help="Meta prompt template to use (default: v3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fanout", type=int, default=1, help="Number of different samples to generate per dataset per pass (default: 1)")
    parser.add_argument("--balance_test_labels", type=str, choices=["enabled","disabled"], default="enabled", help="Balance label distribution in test selection: enabled|disabled (default: enabled)")
    parser.add_argument("--balance_train_labels", type=str, choices=["enabled","disabled"], default="disabled", help="Balance label distribution in train selection: enabled|disabled (default: disabled)")
    
    args = parser.parse_args()
    args.balance_test_labels = (args.balance_test_labels == "enabled")
    args.balance_train_labels = (args.balance_train_labels == "enabled")
    print(f"Balance test labels: {args.balance_test_labels}")
    print(f"Balance train labels: {args.balance_train_labels}")
    
    # Parse training example ratios
    try:
        train_example_ratios = parse_train_example_ratios(args.train_example_ratios)
        print(f"Using training example ratios: {train_example_ratios}")
    except ValueError as e:
        print(f"Error parsing train_example_ratios: {e}")
        return 1
    
    # Load splits from post-processing
    print(f"Loading splits from {args.input_dir}...")
    train_dataset = load_from_disk(os.path.join(args.input_dir, "train"))
    val_dataset = load_from_disk(os.path.join(args.input_dir, "validation"))
    test_dataset = load_from_disk(os.path.join(args.input_dir, "test"))
    
    print(f"Loaded splits:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    # Process each split using multi-pass approach
    print("Converting to VERL format using multi-pass approach...")
    print(f"Using meta prompt template: {args.meta_prompt_template}")
    
    train_verl = process_split_multi_pass(train_dataset, train_example_ratios, args.num_test_examples, args.max_datasets, args.seed, args.meta_prompt_template, args.fanout, args.balance_test_labels, args.balance_train_labels)
    val_verl = process_split_multi_pass(val_dataset, train_example_ratios, args.num_test_examples, args.max_datasets, args.seed, args.meta_prompt_template, args.fanout, args.balance_test_labels, args.balance_train_labels)
    test_verl = process_split_multi_pass(test_dataset, train_example_ratios, args.num_test_examples, args.max_datasets, args.seed, args.meta_prompt_template, args.fanout, args.balance_test_labels, args.balance_train_labels)
    
    # Save VERL format datasets
    save_verl_datasets(train_verl, val_verl, test_verl, args.output_dir, args, train_example_ratios)
    
    print("VERL training preprocessing completed successfully!")


if __name__ == "__main__":
    main()