#!/usr/bin/env python3
"""
View the output of prompt_gen_preprocess.py
Supports both chunked and single dataset formats.
"""

from datasets import load_from_disk, concatenate_datasets
import os
from collections import defaultdict
import argparse


def load_dataset_from_dir(save_dir):
    """Load dataset from directory, handling both chunked and single formats."""
    # Check if it's a chunked dataset
    chunk_dirs = [d for d in os.listdir(save_dir) if d.startswith('chunk_') and os.path.isdir(os.path.join(save_dir, d))]

    if chunk_dirs:
        # Chunked format
        print(f"Found {len(chunk_dirs)} chunks, loading...")
        datasets = []
        for chunk_dir in sorted(chunk_dirs):
            chunk_path = os.path.join(save_dir, chunk_dir)
            datasets.append(load_from_disk(chunk_path))
        dataset = concatenate_datasets(datasets)
        print(f"Loaded chunked dataset from {save_dir}")
    else:
        # Single dataset format
        dataset = load_from_disk(save_dir)
        print(f"Loaded single dataset from {save_dir}")

    return dataset


def view_prompt_gen_output(save_dir, num_examples=3):
    """View the output dataset from prompt_gen_preprocess.py"""

    # Check if the dataset exists
    if not os.path.exists(save_dir):
        print(f"Dataset directory {save_dir} does not exist.")
        print("Please run prompt_gen_preprocess.py first with:")
        print(f"python prompt_gen_preprocess.py --save_dir {save_dir}")
        return

    try:
        # Load dataset
        dataset = load_dataset_from_dir(save_dir)

        print(f"Dataset loaded successfully!")
        print(f"Dataset info: {dataset}")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {dataset.features}")
        print("\n" + "="*80)

        # Display first few examples
        print(f"First {num_examples} examples:")
        for i, example in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
            print(f"\n{'='*80}")
            print(f"Example {i+1}:")
            print(f"  Dataset name: {example['dataset_name']}")
            print(f"  Config name: {example['config_name']}")
            print(f"  Split name: {example['split_name']}")
            print(f"  Input column: {example['input_column']}")
            print(f"  Label column: {example['label_column']}")
            print(f"  Number of train examples: {len(example['train_examples'])}")
            print(f"  Number of test examples: {len(example['test_examples'])}")
            print(f"  Label names ({len(example['label_names'])}): {example['label_names']}")

            # Show first train example
            if example['train_examples']:
                first_train = example['train_examples'][0]
                print(f"\n  First train example:")
                text_preview = first_train['text'][:150] + "..." if len(first_train['text']) > 150 else first_train['text']
                print(f"    Text: {text_preview}")
                print(f"    Label: {first_train['label']}")

            # Show first test example
            if example['test_examples']:
                first_test = example['test_examples'][0]
                print(f"\n  First test example:")
                text_preview = first_test['text'][:150] + "..." if len(first_test['text']) > 150 else first_test['text']
                print(f"    Text: {text_preview}")
                print(f"    Label: {first_test['label']}")

        print("\n" + "="*80)
        print("Dataset Statistics:")
        print(f"Total examples: {len(dataset)}")

        # Count unique datasets, configs, and splits
        unique_datasets = set()
        unique_configs = set()
        unique_splits = set()
        dataset_counts = defaultdict(int)
        input_columns = defaultdict(int)
        label_columns = defaultdict(int)

        for example in dataset:
            unique_datasets.add(example['dataset_name'])
            unique_configs.add(f"{example['dataset_name']}/{example['config_name']}")
            unique_splits.add(example['split_name'])
            dataset_counts[example['dataset_name']] += 1
            input_columns[example['input_column']] += 1
            label_columns[example['label_column']] += 1

        print(f"Unique datasets: {len(unique_datasets)}")
        print(f"Unique configs: {len(unique_configs)}")
        print(f"Unique splits: {len(unique_splits)}")

        # Show breakdown by dataset
        print(f"\nTop 10 datasets by config count:")
        for dataset_name, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dataset_name}: {count} configs")

        # Column analysis
        print(f"\nColumn usage:")
        print(f"Input columns:")
        for col, count in sorted(input_columns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {count} times")

        print(f"Label columns:")
        for col, count in sorted(label_columns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {count} times")

        # Label name statistics
        all_label_names = set()
        num_labels_list = []
        for example in dataset:
            all_label_names.update(example['label_names'])
            num_labels_list.append(len(example['label_names']))

        print(f"\nLabel statistics:")
        print(f"Total unique label names across all datasets: {len(all_label_names)}")
        print(f"Average labels per dataset: {sum(num_labels_list) / len(num_labels_list):.2f}")
        print(f"Min labels: {min(num_labels_list)}")
        print(f"Max labels: {max(num_labels_list)}")

        # Show some sample label names
        sample_labels = list(all_label_names)[:10]
        print(f"Sample label names: {sample_labels}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the dataset was created successfully by running prompt_gen_preprocess.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View prompt_gen_preprocess.py output")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory containing the preprocessed dataset")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to display")

    args = parser.parse_args()

    view_prompt_gen_output(save_dir=args.save_dir, num_examples=args.num_examples)
