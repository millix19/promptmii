import asyncio
import httpx
import time
import os
from sklearn.metrics import f1_score
import numpy as np

## SGLang fast version with high concurrency using proven baseline architecture
## Uses the same exact logic as v4 but with SGLang backend for 5x performance improvement
## Adds a custom line after the generated instruction to ensure the model only outputs valid label options, similar to baseline v3.
## adds truncate prompt tokens
## Uses F1 score instead of accuracy for evaluation

async def query_sglang_batch_async(prompts, max_tokens=90, max_concurrency=128, timeout_seconds=60.0):
    """
    Call the SGLang router endpoint for batch inference with async concurrency.
    Uses the same proven architecture as the fast baseline evaluation.
    
    Args:
        prompts: List of input prompts.
        max_tokens: Maximum number of tokens to generate for each prompt.
        max_concurrency: Maximum number of concurrent requests (128 like proven baseline).
        timeout_seconds: Timeout for each request.

    Returns:
        List of generated responses.
    """
    # Use SGLang router endpoint (same as baseline)
    sglang_host = os.environ.get("SGLANG_SERVER_ADDR", "localhost")
    sglang_port = os.environ.get("SGLANG_SERVER_PORT", "8100")  # Default to prediction port
    url = f"http://{sglang_host}:{sglang_port}/v1/chat/completions"
    model = os.environ.get("SGLANG_MODEL_NAME", "default")

    async def _run():
        sem = asyncio.Semaphore(max_concurrency)
        limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
        timeout = httpx.Timeout(timeout_seconds)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            async def one_call(prompt):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                }
                
                async with sem:
                    try:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        result = resp.json()
                        if "choices" in result and result["choices"]:
                            msg = result["choices"][0].get("message", {})
                            content = msg.get("content")
                            if content and content.strip():
                                return content
                            else:
                                return ""
                        else:
                            return ""
                    except Exception as e:
                        print(f"Error querying SGLang: {type(e).__name__}: {str(e)}")
                        print(f"  URL: {url}")
                        print(f"  Payload: {payload}")
                        if hasattr(e, 'response'):
                            print(f"  Status Code: {e.response.status_code}")
                            print(f"  Response: {e.response.text}")
                        return ""

            tasks = [one_call(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
        # If already inside an event loop, create a task and wait
        task = asyncio.create_task(_run())
        return await task
    except RuntimeError:
        # No running loop: safe to asyncio.run
        return asyncio.run(_run())


def query_sglang_batch(prompts, max_tokens=90):
    """
    Synchronous wrapper for async batch query function.
    Maintains compatibility with existing code while providing SGLang async performance.
    """
    return asyncio.run(query_sglang_batch_async(prompts, max_tokens))


def compute_f1_score(predictions, labels, label_names):
    """
    Compute F1 score by comparing predictions with labels.
    """
    # Handle edge cases
    if not predictions or not labels:
        return 0.0
    
    # Convert predictions and labels to numeric indices for F1 calculation
    label_to_idx = {label.lower().strip(): i for i, label in enumerate(label_names)}
    
    pred_indices = []
    label_indices = []
    
    for pred, label in zip(predictions, labels):
        pred_clean = pred[:100].lower().strip()  # Limit prediction length and clean
        label_clean = label.lower().strip()
        
        # Convert prediction to index
        if pred_clean in label_to_idx:
            pred_idx = label_to_idx[pred_clean]
        else:
            # If prediction doesn't match any label, assign to first label as fallback
            pred_idx = 0
        
        # Convert label to index
        if label_clean in label_to_idx:
            label_idx = label_to_idx[label_clean]
        else:
            # If label doesn't match any label name, assign to first label as fallback
            label_idx = 0
        
        pred_indices.append(pred_idx)
        label_indices.append(label_idx)
    
    # Calculate F1 score
    try:
        f1 = f1_score(label_indices, pred_indices, average='macro', zero_division=0)
        # Handle nan values
        if f1 != f1:  # Check for nan
            return 0.0
        return f1
    except Exception as e:
        print(f"Error computing F1 score: {e}")
        return 0.0

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute rewards for multiple rollouts in batch using SGLang with proven fast architecture.
    This version uses SGLang's prefill/decode disaggregation for optimal performance while maintaining
    the exact same logic as v4. Adds a custom line after the generated instruction to ensure
    the model only outputs valid label options, similar to baseline v3.
    Uses F1 score instead of accuracy for evaluation.
    
    Args:
        data_sources: List of dataset names (not used directly here).
        solution_strs: List of generated instructions.
        ground_truths: List of ground truth responses.
        extra_infos: List of metadata containing test examples and label mappings.

    Returns:
        List of rewards for the batch.
    """
    # Validate inputs
    if not all("test_examples" in info and "label_mapping" in info for info in extra_infos):
        raise ValueError("Missing required fields in extra_info: 'test_examples' and 'label_mapping'.")

    # Track timing for performance analysis
    start_time = time.time()
    
    # Prepare batched prompts and labels (EXACT SAME LOGIC AS V4)
    prompts = []
    labels_list = []
    for solution_str, extra_info in zip(solution_strs, extra_infos):
        test_examples = extra_info["test_examples"]
        label_mapping = extra_info["label_mapping"]
        
        # Add custom line after the generated instruction (baseline v3 style)
        custom_line = f"Only return one of these options: {', '.join(label_mapping)}. Do not output \"Label:\" or any extra text.\n\n"
        final_instruction = solution_str + "\n" + custom_line
        
        for example in test_examples:
            input_text = f"{final_instruction}Input: {example['text']}"
            prompts.append(input_text)
            ## in the v2 dataset, the label is the label name
            labels_list.append(example["label"])
            ## careful: in the v1 dataset, the label is the index, so we need to convert it to the label name
            # labels_list.append(label_mapping[example["label"]])

    print(f"[SGLANG PERFORMANCE] Processing {len(prompts)} prompts with SGLang async concurrency (128 concurrent)...")
    
    # Batch API call to SGLang router with high concurrency (proven baseline architecture)
    predictions = query_sglang_batch(prompts)
    
    inference_time = time.time() - start_time
    print(f"[SGLANG PERFORMANCE] Inference completed in {inference_time:.2f}s for {len(prompts)} prompts ({len(prompts)/inference_time:.1f} prompts/s)")

    # Compute rewards using F1 score instead of accuracy
    rewards = []
    idx = 0
    for solution_str, extra_info in zip(solution_strs, extra_infos):
        test_examples = extra_info["test_examples"]
        label_mapping = extra_info["label_mapping"]
        batch_predictions = predictions[idx:idx + len(test_examples)]
        batch_labels = labels_list[idx:idx + len(test_examples)]
        idx += len(test_examples)
        reward = compute_f1_score(batch_predictions, batch_labels, label_mapping)
        rewards.append(reward)

    total_time = time.time() - start_time
    print(f"[SGLANG PERFORMANCE] Total reward computation: {total_time:.2f}s")
    
    return rewards

if __name__ == "__main__":
    # Example inputs
    data_sources = ["dataset1", "dataset2"]
    solution_strs = [
        "Classify the text into one of the following categories: positive, negative.",
        "Determine if the text is about sports, politics, or technology."
    ]
    ground_truths = [
    ]
    extra_infos = [
        {
            "test_examples": [{"text": "I love this!", "label": "positive"}, {"text": "This is bad.", "label": "negative"}],
            "label_mapping": ["positive", "negative"]
        },
        {
            "test_examples": [{"text": "The team won the match.", "label": "sports"}, {"text": "The election results are out.", "label": "politics"}],
            "label_mapping": ["sports", "politics", "technology"]
        }
    ]

    # Compute rewards
    rewards = compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    print("Rewards:", rewards)