import litellm
from verl.utils.api_config import get_litellm_config
from typing import List, Dict, Any

# Note: max_tokens values are now passed as parameters to functions

# Get API configuration
API_KEY, BASE_URL = get_litellm_config()

def get_model_temperature(model_name, desired_temperature):
    """Get the appropriate temperature for a given model"""
    if "o3" in model_name.lower():
        # O-series models only support temperature=1
        return 1.0
    else:
        return desired_temperature

def query_api_chat_batch(messages_list, max_tokens=100, temperature=0.0, model=None, vllm_url=None):
    """Replace vLLM calls with LiteLLM API calls"""
    try:
        # Use global temperature function
        adjusted_temperature = get_model_temperature(model, temperature)
        if adjusted_temperature != temperature:
            print(f"[INFO] Model {model} temperature adjusted from {temperature} to {adjusted_temperature}")
            
        responses = litellm.batch_completion(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=model,
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=adjusted_temperature,
            drop_params=True  # Drop unsupported parameters gracefully
        )
        return [response["choices"][0]["message"]["content"].strip() for response in responses]
    except Exception as e:
        print(f"Error querying API: {e}")
        return [""] * len(messages_list)

def format_examples(dataset_samples, label_names):
    s = ""
    for ex in dataset_samples:
        label = ex["label"]
        s += f"Input: {ex['text']}\nLabel: {label}\n"
    return s

def generate_instruction_api(train_examples, label_names, meta_prompt_template, api_model, max_tokens_instruction=1024):
    """Generate instruction using API model - functionally identical to generate_instruction"""
    examples_str = format_examples(train_examples, label_names)
    label_names_str = ", ".join(label_names)  # Match the vLLM version exactly
    meta_prompt = meta_prompt_template.format(examples=examples_str, label_names_str=label_names_str)
    messages = [{"role": "user", "content": meta_prompt}]
    response = query_api_chat_batch([messages], max_tokens=max_tokens_instruction, temperature=0.0, model=api_model, vllm_url=None)  # Match temperature=0.0
    return response[0]  # Match the vLLM version exactly

def apply_instruction_batch_api(examples, instruction, api_model, max_tokens_prediction=10):
    """Apply instruction to examples using API model - functionally identical to apply_instruction_batch"""
    messages_list = []
    for ex in examples:
        messages = [{"role": "user", "content": instruction + f"Input: {ex['text']}"}]  # Match the vLLM version exactly
        messages_list.append(messages)
    responses = query_api_chat_batch(messages_list, max_tokens=max_tokens_prediction, temperature=0.0, model=api_model, vllm_url=None)  # Match temperature=0.0
    return responses

def count_tokens_in_instruction(instruction, model_name):
    """Count the number of tokens in an instruction using the model's tokenizer"""
    try:
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(model_name)
        tokens = tokenizer.encode(instruction)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Could not count tokens for {model_name}: {e}")
        # Fallback: rough estimation (1 token ≈ 4 characters for English)
        return len(instruction) // 4
