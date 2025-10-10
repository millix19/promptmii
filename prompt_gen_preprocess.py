"""
Preprocess text classification datasets for prompt generation training.
Uses LLM to identify input and label columns dynamically.
"""

import argparse
import os
from datasets import load_dataset, load_dataset_builder, Dataset, get_dataset_config_names, get_dataset_split_names, Features, Value, Sequence
from tqdm import tqdm
from huggingface_hub import HfApi
from verl.utils.hdfs_io import copy, makedirs
import random
import litellm
from typing import List, Dict, Any, Tuple
from verl.utils.api_config import get_litellm_config


# LiteLLM configuration
API_KEY, BASE_URL = get_litellm_config()


def query_litellm(model, messages):
    """Query LiteLLM API."""
    try:
        client = litellm.completion(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=model,
            messages=messages
        )
        return client.choices[0].message.content.strip()
    except Exception as e:
        print("INFO litellm issue:", e)
        return ""


def get_text_classification_datasets(limit=10000):
    """Fetch dataset names tagged with 'text-classification'."""
    api = HfApi()
    datasets = api.list_datasets(filter="task_categories:text-classification", sort="trending_score")
    datasets = list(datasets)
    return [ds.id for ds in datasets[:limit]]


def identify_columns_with_llm(dataset_name, config, dataset_description, examples, dataset_columns, model="litellm_proxy/neulab/gpt-4.1-mini-2025-04-14"):
    """
    Use LiteLLM to identify input and label columns in a Hugging Face dataset.
    """
    prompt = f"""
You are an expert data scientist. Given a dataset for text classification, your task is to identify:

1. The name of the column that contains the **input text** to be classified.
2. The name of the column that contains the **label** (category or class) for classification.

Return your answer in this exact format:
input_column: <column_name>
label_column: <column_name>

Dataset name: {dataset_name}
Config: {config}
Description: {dataset_description}

Columns: {', '.join(dataset_columns)}

Examples:
"""

    for i, ex in enumerate(examples):
        prompt += f"\nExample {i+1}:\n"
        for k, v in ex.items():
            v_str = str(v)
            if isinstance(v_str, str) and len(v_str) > 200:
                v_str = v_str[:200] + "..."
            prompt += f"  {k}: {v_str}\n"

    prompt += "\nNow return only the two column names in the specified format."
    
    # Send request via LiteLLM
    content = query_litellm(model, messages=[{"role": "user", "content": prompt}])

    # Parse response
    input_column, label_column = None, None
    for line in content.splitlines():
        if line.lower().startswith("input_column:"):
            input_column = line.split(":", 1)[1].strip()
        elif line.lower().startswith("label_column:"):
            label_column = line.split(":", 1)[1].strip()

    if not input_column or not label_column:
        raise ValueError(f"LLM failed to extract columns. Response:\n{content}")

    return input_column, label_column


def get_label_names(dataset, label_column):
    """Get label names from the dataset builder."""
    try:
        label_names = dataset.features[label_column].names
        # Convert to strings to ensure compatibility
        if label_names is not None:
            return [str(name) for name in label_names]
        return label_names
    except Exception as e:
        print(f"INFO Counting label names from dataset due to: {e}")
        return None


def get_label_value(example, label_column, label_names):
    """Get the string representation of a label value."""
    label_value = example[label_column]
    
    # If we have label_names (ClassLabel case), convert index to string
    if label_names is not None and isinstance(label_value, int):
        return str(label_names[label_value])
    
    # Otherwise, just return the string representation
    return str(label_value)


def get_dataset_features():
    """Get the features schema for the processed dataset."""
    return Features({
        'dataset_name': Value('string'),
        'config_name': Value('string'),
        'split_name': Value('string'),
        'train_examples': Sequence({
            'text': Value('string'),
            'label': Value('string')
        }),
        'test_examples': Sequence({
            'text': Value('string'),
            'label': Value('string')
        }),
        'label_names': Sequence(Value('string')),
        'input_column': Value('string'),
        'label_column': Value('string')
    })


def process_dataset_with_config(dataset_name, config, num_train_examples, num_test_examples):
    """Process a single dataset config."""
    try:
        # Load dataset builder
        builder = load_dataset_builder(dataset_name, config)
        
        # Get available splits using get_dataset_split_names, if train not in, use first split
        available_splits = get_dataset_split_names(dataset_name, config)
        if 'train' not in available_splits:
            split_name = available_splits[0]  # Use first split
            print(f"INFO Train split not found for {dataset_name} (config: {config}), using '{split_name}' split instead.")
        else:
            split_name = 'train'
            print(f"INFO Using 'train' split for {dataset_name} (config: {config})")
        
        # Load dataset with streaming and shuffle
        dataset = load_dataset(dataset_name, config, split=split_name, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=5000)
        # Get dataset description and examples for LLM
        dataset_description = builder.info.description or "" # this is always empty
        examples = list(dataset.take(3))
        # dataset_columns = list(dataset.features.keys())
        dataset_columns = list(examples[0].keys())
        
        # Use LLM to identify columns
        input_column, label_column = identify_columns_with_llm(
            dataset_name, config, dataset_description, examples, dataset_columns
        )
        
        print(f"INFO Dataset: {dataset_name}, Config: {config}")
        print(f"INFO Identified columns - Input: {input_column}, Label: {label_column}")
        
        # Take train/test examples first
        all_examples = list(dataset.take(num_train_examples + num_test_examples))
        
        if len(all_examples) < num_train_examples + num_test_examples:
            print(f"Skipping {dataset_name} (config: {config}) has insufficient examples")
            return None
        
        # Get label names from the actual sampled examples
        label_names = get_label_names(dataset, label_column) #TODO this could be list of ints
        if label_names is None: # TODO this part is slow
            # Need to determine unique values from the sampled data
            unique_labels = set()
            for ex in all_examples:
                unique_labels.add(str(ex[label_column]))
            label_names = list(unique_labels)
        
        print(f"INFO Label names (first 10 of {len(label_names)}): {label_names[:10]}")
        
        # Check if number of labels is too high (likely wrong label column)
        total_examples_needed = num_train_examples + num_test_examples
        if len(label_names) > 0.5 * total_examples_needed:
            print(f"Skipping {dataset_name} (config: {config}) - too many unique labels ({len(label_names)}) for {total_examples_needed} examples. LLM likely identified wrong label column.")
            return None
        # Also skip if only one label
        if len(label_names) == 1:
            print(f"Skipping {dataset_name} (config: {config}) - only one label.")
            return None
        
        # Split into train and test
        train_examples = all_examples[:num_train_examples]
        test_examples = all_examples[num_train_examples:num_train_examples + num_test_examples]
        
        # Process examples to get proper label values
        processed_train_examples = []
        for ex in train_examples:
            processed_ex = {
                "text": str(ex[input_column]),
                "label": str(get_label_value(ex, label_column, label_names))
            }
            processed_train_examples.append(processed_ex)
        
        processed_test_examples = []
        for ex in test_examples:
            processed_ex = {
                "text": str(ex[input_column]),
                "label": str(get_label_value(ex, label_column, label_names))
            }
            processed_test_examples.append(processed_ex)
        
        return {
            "dataset_name": dataset_name, # string
            "config_name": config, # string
            "split_name": split_name, # string  
            "train_examples": processed_train_examples,
            "test_examples": processed_test_examples, 
            "label_names": label_names, # list of strings
            "input_column": input_column, # string
            "label_column": label_column # string   
        }
        
    except Exception as e:
        print(f"Skipping {dataset_name} (config: {config}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, help="Directory to save the dataset (e.g., /data/user_data/emilyx/prompt_gen_dataset)")
    parser.add_argument("--num_train_examples", type=int, default=5, help="Number of training examples per dataset.")
    parser.add_argument("--num_test_examples", type=int, default=5, help="Number of test examples per dataset.")
    parser.add_argument("--limit_datasets", type=int, default=20, help="Limit the number of datasets to process.")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of datasets per chunk.")
    parser.add_argument("--resume", type=bool, default=True, help="Whether to resume from existing processed data (default: True).")

    args = parser.parse_args()

    # Set up base output directory
    base_output_path = args.save_dir
    makedirs(base_output_path, exist_ok=True)

    # Fetch dataset names
    dataset_names = get_text_classification_datasets(limit=args.limit_datasets)
    dataset_names = dataset_names[3000:] # WARNING: harcorded resume
    
    # RESUME LOGIC: Load all existing processed data (if resuming)
    processed_datasets = set()
    chunk_counter = 0
    
    if args.resume:
        # Load all existing chunks and collect processed dataset names
        while os.path.exists(os.path.join(base_output_path, str(chunk_counter))):
            chunk_dir = os.path.join(base_output_path, str(chunk_counter))
            try:
                dataset = Dataset.load_from_disk(chunk_dir)
                for item in dataset:
                    processed_datasets.add(item['dataset_name'])
                print(f"Loaded chunk {chunk_counter} with {len(dataset)} datasets")
                chunk_counter += 1
            except Exception as e:
                print(f"Error loading chunk {chunk_counter}: {e}")
                break
        
        print(f"Resuming from chunk {chunk_counter}. Already processed {len(processed_datasets)} unique datasets.")
    else:
        print("Starting over - ignoring existing processed data.")
    
    # Initialize processed data list
    processed_data = []

    for idx, dataset_name in enumerate(tqdm(dataset_names, desc="Processing datasets")):
        # Skip if dataset was already processed (if resuming)
        if args.resume and dataset_name in processed_datasets:
            print(f"Skipping already processed: {dataset_name}")
            print("--------------------------------")      
            continue
        
        # Get all configs for this dataset
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception as e:
            print(f"Skipping cannot get dataset configs for {dataset_name}: {e}")
            configs = [None]

        if not configs:
            configs = [None]  # Use None to indicate no specific config
        
        for config in configs:
            
            result = process_dataset_with_config(
                dataset_name, config, 
                args.num_train_examples, 
                args.num_test_examples
            )
            
            if result is not None:
                processed_data.append(result)
                print(f"Valid dataset processed ({idx+1}/{len(dataset_names)}): {dataset_name} (config: {config}). Total valid datasets: {chunk_counter * args.chunk_size + len(processed_data)}")
                
                # Save chunk when we reach chunk_size
                if len(processed_data) >= args.chunk_size:
                    try:
                        print(f"Attempting to save chunk {chunk_counter} with {len(processed_data)} datasets to {chunk_dir}")
                        chunk_dir = os.path.join(base_output_path, str(chunk_counter))
                        # Use shared features schema to avoid type inference issues
                        # features = get_dataset_features()
                        dataset = Dataset.from_list(processed_data) #, features=features)
                        dataset.save_to_disk(chunk_dir)
                        print(f"Saved chunk {chunk_counter} with {len(processed_data)} datasets to {chunk_dir}")
                        chunk_counter += 1
                        processed_data = []  # Clear memory
                    except Exception as e:
                        print(f"Error saving chunk {chunk_counter}: {e}")
                        processed_data = []  # Clear failed data to prevent accumulation
                        print("--------------------------------")    
                        continue
                    
        print("--------------------------------")      

    # Final save of remaining data
    if processed_data:
        chunk_dir = os.path.join(base_output_path, str(chunk_counter))
        # Use shared features schema to avoid type inference issues
        features = get_dataset_features()
        dataset = Dataset.from_list(processed_data, features=features)
        print(f"Saved final chunk {chunk_counter} with {len(processed_data)} datasets to {chunk_dir}")
        dataset.save_to_disk(chunk_dir)
        print(f"Total chunks saved: {chunk_counter + 1}")
    else:
        print("No new datasets were successfully processed.")


def load_concatenated_dataset(base_dir):
    """
    Load and concatenate all chunk directories from a base directory.
    
    Args:
        base_dir: Base directory containing numbered chunk directories (0, 1, 2, ...)
        
    Returns:
        Concatenated dataset
    """
    from datasets import concatenate_datasets
    
    # Find all chunk directories
    chunk_dirs = []
    chunk_num = 0
    while os.path.exists(os.path.join(base_dir, str(chunk_num))):
        chunk_dirs.append(os.path.join(base_dir, str(chunk_num)))
        chunk_num += 1
    
    if not chunk_dirs:
        raise ValueError(f"No chunk directories found in {base_dir}")
    
    print(f"Found {len(chunk_dirs)} chunk directories")
    
    # Load each chunk
    datasets = []
    for chunk_dir in chunk_dirs:
        print(f"Loading {chunk_dir}")
        dataset = Dataset.load_from_disk(chunk_dir)
        datasets.append(dataset)
    
    # Concatenate all datasets
    concatenated_dataset = concatenate_datasets(datasets)
    print(f"Concatenated dataset has {len(concatenated_dataset)} total datasets")
    
    return concatenated_dataset


if __name__ == "__main__":
    main() 