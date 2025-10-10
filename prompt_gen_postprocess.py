#!/usr/bin/env python3
"""
Post-processing script for prompt generation datasets.
Step 2: Second stage pre-processing after initial dataset extraction.

This script:
1. Limits max 10 configs per dataset
2. Splits into training, validation, and custom test sets
3. Saves as a new HuggingFace dataset
"""

import argparse
import os
import random
from collections import defaultdict
from datasets import load_from_disk, Dataset, concatenate_datasets
from verl.utils.hdfs_io import makedirs
import json
from prompt_gen_preprocess import load_concatenated_dataset





def limit_configs_per_dataset(dataset, max_configs=10):
    """
    Limit the number of configs per dataset to max_configs.
    Keeps the first max_configs configs for each dataset (in order).
    
    Args:
        dataset: The input dataset
        max_configs: Maximum number of configs per dataset
        
    Returns:
        Filtered dataset
    """
    print(f"Limiting configs to {max_configs} per dataset...")
    
    # Simple approach: keep first max_configs configs per dataset
    dataset_configs = defaultdict(list)
    for i, example in enumerate(dataset):
        dataset_name = example['dataset_name']
        dataset_configs[dataset_name].append(i)
    
    # Keep only first max_configs configs per dataset
    indices_to_keep = []
    for dataset_name, indices in dataset_configs.items():
        indices_to_keep.extend(indices[:max_configs])
    
    # Create filtered dataset
    filtered_dataset = dataset.select(indices_to_keep)
    
    print(f"Original dataset: {len(dataset)} examples")
    print(f"Filtered dataset: {len(filtered_dataset)} examples")
    
    # Show breakdown
    dataset_counts = defaultdict(int)
    for example in filtered_dataset:
        dataset_counts[example['dataset_name']] += 1
    
    print(f"Datasets with configs:")
    for dataset_name, count in sorted(dataset_counts.items()):
        print(f"  {dataset_name}: {count} configs")
    
    return filtered_dataset


def split_datasets(dataset, validation_ratio=0.1, test_datasets=None):
    """
    Split datasets into training, validation, and test sets.
    
    Args:
        dataset: Input dataset
        validation_ratio: Ratio of datasets for validation (default: 0.1)
        test_datasets: List of dataset names to put in test set
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print("Splitting datasets into train/validation/test sets...")
    
    if test_datasets is None:
        test_datasets = [
            "fancyzhx/ag_news",
            "CogComp/trec", 
            "legacy-datasets/banking77",
            "community-datasets/yahoo_answers_topics",
            "fancyzhx/dbpedia_14"
        ]
    
    # Group by dataset name
    dataset_groups = defaultdict(list)
    for i, example in enumerate(dataset):
        dataset_name = example['dataset_name']
        dataset_groups[dataset_name].append(i)
    
    # Separate test datasets
    test_indices = []
    non_test_indices = []
    
    for dataset_name, indices in dataset_groups.items():
        if dataset_name in test_datasets:
            test_indices.extend(indices)
        else:
            non_test_indices.extend(indices)
    
    # Split remaining datasets into train/validation
    random.seed(42)  # For reproducibility
    random.shuffle(non_test_indices)
    
    split_point = int(len(non_test_indices) * validation_ratio)
    val_indices = non_test_indices[:split_point]
    train_indices = non_test_indices[split_point:]
    
    # Create split datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    print(f"Split results:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    # Show dataset breakdown for each split
    for split_name, split_dataset in [("Training", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        unique_datasets = set()
        for example in split_dataset:
            unique_datasets.add(example['dataset_name'])
        print(f"  {split_name} datasets: {len(unique_datasets)}")
    
    return train_dataset, val_dataset, test_dataset


def save_splits(train_dataset, val_dataset, test_dataset, output_dir):
    """
    Save the split datasets to disk.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        output_dir: Output directory
    """
    print(f"Saving splits to {output_dir}...")
    
    makedirs(output_dir, exist_ok=True)
    
    # Save each split
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    test_dataset.save_to_disk(os.path.join(output_dir, "test"))
    
    # Save metadata
    metadata = {
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "test_examples": len(test_dataset),
        "total_examples": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "train_datasets": len(set(example['dataset_name'] for example in train_dataset)),
        "validation_datasets": len(set(example['dataset_name'] for example in val_dataset)),
        "test_datasets": len(set(example['dataset_name'] for example in test_dataset))
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved splits to {output_dir}")
    print(f"Metadata: {metadata}")


def main():
    parser = argparse.ArgumentParser(description="Post-process prompt generation datasets")
    parser.add_argument("--input_dir", required=True, help="Input directory containing chunked datasets")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed splits")
    parser.add_argument("--max_configs", type=int, default=10, help="Maximum configs per dataset")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of datasets for validation")
    parser.add_argument("--test_datasets", nargs="+", default=None, help="List of dataset names for test set")
    
    args = parser.parse_args()
    
    # Load concatenated dataset
    print(f"Loading dataset from {args.input_dir}...")
    dataset = load_concatenated_dataset(args.input_dir)
    
    # Step 1: Limit configs per dataset
    dataset = limit_configs_per_dataset(dataset, args.max_configs)
    
    # Step 2: Split into train/validation/test
    train_dataset, val_dataset, test_dataset = split_datasets(
        dataset, 
        validation_ratio=args.validation_ratio,
        test_datasets=args.test_datasets
    )
    
    # Step 3: Save splits
    save_splits(train_dataset, val_dataset, test_dataset, args.output_dir)
    
    print("Post-processing completed successfully!")


if __name__ == "__main__":
    main() 