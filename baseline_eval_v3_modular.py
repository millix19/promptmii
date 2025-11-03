import random
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import os
import argparse
from verl.utils import hf_tokenizer
from typing import List, Dict, Any, Optional
import asyncio
import httpx

# Global variables for max tokens
MAX_TOKENS_INSTRUCTION = 2048
MAX_TOKENS_PREDICTION = 10

# Default context length (in tokens)
DEFAULT_CONTEXT_LENGTH = 32768
SAFETY_MARGIN = 2000

class ContextLengthManager:
    """Manages context length calculations and example limits per dataset."""
    
    def __init__(self, context_length=32768, safety_margin=1000):
        self.context_length = context_length
        self.safety_margin = safety_margin
        self.cache = {}
    
    def get_adjusted_n(self, requested_n, train_examples, instruction_template, label_names, dataset_key):
        """Get adjusted n that fits within context"""
        if dataset_key in self.cache:
            max_safe = self.cache[dataset_key]
        else:
            max_safe = self._calculate_max_examples(train_examples, instruction_template, label_names)
            self.cache[dataset_key] = max_safe
        
        actual_n = min(requested_n, max_safe)
        
        if actual_n != requested_n:
            print(f"[CONTEXT_ADJUSTMENT] {dataset_key}: {requested_n} → {actual_n}")
        
        return actual_n, requested_n
    
    def _calculate_max_examples(self, train_examples, instruction_template, label_names):
        """Calculate max examples that fit within context"""
        # Count base instruction tokens (without examples)
        base_instruction = instruction_template.format(examples="", label_names_str=", ".join(label_names))
        base_tokens = self._count_tokens(base_instruction)
        
        # Count custom line tokens
        custom_line = f"Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"
        custom_tokens = self._count_tokens(custom_line)
        
        # Start from 0 and add examples incrementally
        total_tokens = base_tokens + custom_tokens
        
        for i, example in enumerate(train_examples):
            example_text = f"Input: {example['text']}\nLabel: {example['label']}\n"
            example_tokens = self._count_tokens(example_text)
            
            # Check if adding this example would exceed the limit
            if total_tokens + example_tokens > self.context_length - self.safety_margin:
                return max(0, i - 1)  # Return the number of examples that fit, minus 1 for input
            
            total_tokens += example_tokens
        
        # All examples fit, but still subtract 1 for input
        return max(0, len(train_examples) - 1)
    
    def _count_tokens(self, text):
        """Count tokens in text using the model's tokenizer"""
        try:
            tokenizer = hf_tokenizer("Qwen/Qwen2.5-7B-Instruct")  # Use a default model for tokenization
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            return len(text) // 4  # Fallback estimation

def query_local_vllm_chat_batch(
    messages_list,
    max_tokens=100,
    temperature=0.0,
    model=None,
    vllm_url=None,
    *,
    max_concurrency: int = 128,
    timeout_seconds: float = 60.0,
):
    async def _run():
        sem = asyncio.Semaphore(max_concurrency)
        limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
        timeout = httpx.Timeout(timeout_seconds)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            async def one_call(messages):
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                async with sem:
                    try:
                        resp = await client.post(vllm_url, json=payload)
                        resp.raise_for_status()
                        result = resp.json()
                        if "choices" in result and result["choices"]:
                            msg = result["choices"][0].get("message", {})
                            content = msg.get("content")
                            if content and content.strip():
                                return content
                            else:
                                print(f"Warning: Empty response for payload: {payload}")
                                return ""
                        else:
                            print(f"Warning: No choices in response: {result}")
                            return ""
                    except Exception as e:
                        print(f"Error querying vLLM: {e}")
                        return ""

            tasks = [one_call(messages) for messages in messages_list]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
        # If already inside an event loop, create a task and wait.
        return loop.run_until_complete(_run())
    except RuntimeError:
        # No running loop: safe to asyncio.run
        return asyncio.run(_run())
    except Exception:
        # If run_until_complete failed because loop is already running, fall back to a nested loop runner
        # that still preserves behavior in notebook environments.
        # Note: This keeps things simple and avoids extra dependencies like nest_asyncio.
        return asyncio.get_event_loop().run_until_complete(_run())

def naive_instruction_prompt(label_names):
    return f"Classify the Input. Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"

def format_examples(dataset_samples, label_names):
    s = ""
    for ex in dataset_samples:
        label = ex["label"]
        s += f"Input: {ex['text']}\nLabel: {label}\n"
    return s

def in_context_learning_prompt(train_examples, label_names, include_naive_instruction=True):
    s = ""
    for ex in train_examples:
        label = ex["label"]
        s += f"Input: {ex['text']}\nLabel: {label}\n"
    if include_naive_instruction:
        s += "Now classify the next input. " \
            f"Only return one of these options: {', '.join(label_names)}. " \
            "Do not output \"Label:\" or any extra text.\n\n"
    return s

# generated instruction baseline
META_PROMPT_V3 = (
    "You are helping to create a prompt for a language model to classify text inputs. The model should choose one label from the following options: {label_names_str}.\n\n"
    "Here are some example inputs and their correct labels:\n{examples}\n\n"
    "Write an instruction that:\n"
    "- Describes the classification task in a way that generalizes to new inputs.\n"
    "- Points out any useful clues or strategies for making the decision.\n"
    "- Clearly tells the model to respond with only the label name, and not to include any explanation or additional text.\n\n"
    "Provide only the instruction, not the examples or labels."
)

META_PROMPT_V4 = (
    "You are helping to create a prompt for a language model to classify text inputs.\n"
    "\n"
    "I will provide the label set and examples below, and I would like you to write an instruction below:\n"
    "- Describes the classification task in a way that generalizes to new inputs.\n"
    "- Points out any useful clues or strategies for making the decision.\n"
    "- In particular, lists pairs or sets of labels that may be confusing, and describe rules (with examples) about how to distinguish the difference between them.\n"
    "- Clearly tells the model to respond with only the label name, and not to include any explanation or additional text.\n"
    "\n"
    "The model should choose one label from the following options: {label_names_str}\n"
    "\n"
    "Here are some example inputs and their correct labels:\n{examples}\n"
    )

META_PROMPT_V5 = (
    "You are designing a clear instruction for a data annotator to classify text inputs into one of these labels: {label_names_str}\n\n"
    "Here are some example inputs and their correct labels:\n{examples}\n\n"
    "Your task is to write a concise instruction that:\n"
    "- Defines the classification task and clearly explains the meaning of each label.\n"
    "- Provides general labeling strategies and decision rules so annotators can correctly handle unseen inputs.\n"
    "- Highlights common pitfalls, tricky edge cases, and misconceptions to reduce labeling errors.\n"
    "- Keeps the instruction reasonably concise and focused — avoid unnecessary repetition or overly long explanations.\n\n"
)

META_PROMPT_V6 = (
    "You are helping to create a prompt for a language model to classify text inputs. "
    "The model should choose one label from the following options: {label_names_str}.\n\n"
    "Here are some example inputs and their correct labels:\n{examples}\n\n"
    "Write an instruction that:\n"
    "- Describes the classification task in a way that generalizes to new inputs.\n"
    "- Points out any useful clues or strategies for making the decision.\n"
    "- You do not need to specify the exact output format — a line enforcing the format "
    "will be automatically added at the end.\n\n"
    "Provide only the instruction, not the examples or labels."
)

def generate_instruction(train_examples, label_names, meta_prompt_template, model, vllm_url):
    examples_str = format_examples(train_examples, label_names)
    label_names_str = ", ".join(label_names)
    meta_prompt = meta_prompt_template.format(examples=examples_str, label_names_str=label_names_str)
    
    messages = [{"role": "user", "content": meta_prompt}]
    response = query_local_vllm_chat_batch([messages], max_tokens=MAX_TOKENS_INSTRUCTION, temperature=0.0, model=model, vllm_url=vllm_url)
    return response[0]

def apply_instruction_batch(examples, instruction, model, vllm_url):
    messages_list = []
    for ex in examples:
        messages = [{"role": "user", "content": instruction + f"Input: {ex['text']}"}]
        messages_list.append(messages)
    return query_local_vllm_chat_batch(messages_list, max_tokens=MAX_TOKENS_PREDICTION, temperature=0.0, model=model, vllm_url=vllm_url)

def apply_icl_prompt_batch(examples, icl_prompt, model, vllm_url):
    messages_list = []
    for ex in examples:
        messages = [{"role": "user", "content": icl_prompt + f"Input: {ex['text']}"}]
        messages_list.append(messages)
    return query_local_vllm_chat_batch(messages_list, max_tokens=MAX_TOKENS_PREDICTION, temperature=0.0, model=model, vllm_url=vllm_url)

def count_tokens_in_instruction(instruction, model_name):
    tokenizer = hf_tokenizer(model_name)
    return len(tokenizer.encode(instruction))

class BaselineRunner:
    """Modular baseline runner that supports selective execution and append/overwrite modes."""
    
    def __init__(self, instruction_model: str, prediction_model: str, instruction_vllm_url: str, prediction_vllm_url: str = None):
        self.instruction_model = instruction_model
        self.prediction_model = prediction_model
        self.instruction_vllm_url = instruction_vllm_url
        # Use instruction URL for prediction if not specified (backward compatibility)
        self.prediction_vllm_url = prediction_vllm_url if prediction_vllm_url else instruction_vllm_url
        # Initialize context length manager (will be updated with actual values later)
        self.context_manager = ContextLengthManager()
    
    def run_naive_baseline(self, test_subset: List[Dict], label_names: List[str], 
                          dataset_info: Dict[str, Any], requested_n: int, actual_n: int) -> List[Dict]:
        """Run naive instruction baseline."""
        naive_instruction = naive_instruction_prompt(label_names)
        naive_token_count = count_tokens_in_instruction(naive_instruction, self.prediction_model)
        
        print(f"[INFO] Baseline: naive, Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_instruction_batch(test_subset, naive_instruction, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": "naive",
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": None,
                "prediction_model": self.prediction_model,
                "instruction": naive_instruction,
                "instruction_tokens": naive_token_count,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_generated_instruction_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                                         label_names: List[str], dataset_info: Dict[str, Any], 
                                         requested_n: int, actual_n: int, meta_prompt_template: str, baseline_name: str) -> List[Dict]:
        """Run generated instruction baseline with specified meta prompt."""
        gen_instruction = generate_instruction(train_subset, label_names, meta_prompt_template, 
                                            self.instruction_model, self.instruction_vllm_url)
        custom_line = f"Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"
        final_instruction = gen_instruction + "\n" + custom_line
        final_instruction_tokens = count_tokens_in_instruction(final_instruction, self.prediction_model)
        
        print(f"[INFO] Baseline: {baseline_name}, Instr model: {self.instruction_model}, Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_instruction_batch(test_subset, final_instruction, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": baseline_name,
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": self.instruction_model,
                "prediction_model": self.prediction_model,
                "instruction": final_instruction,
                "instruction_tokens": final_instruction_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_generated_instruction_icl_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                                             label_names: List[str], dataset_info: Dict[str, Any], 
                                             requested_n: int, actual_n: int, meta_prompt_template: str, baseline_name: str) -> List[Dict]:
        """Run generated instruction + ICL baseline with specified meta prompt."""
        gen_instruction = generate_instruction(train_subset, label_names, meta_prompt_template, 
                                            self.instruction_model, self.instruction_vllm_url)
        # No need for custom_line since in_context_learning_prompt already includes the constraint
        gen_icl_prompt = gen_instruction + "\n" + in_context_learning_prompt(train_subset, label_names)
        gen_icl_tokens = count_tokens_in_instruction(gen_icl_prompt, self.prediction_model)
        
        print(f"[INFO] Baseline: {baseline_name}, Instr model: {self.instruction_model}, Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_icl_prompt_batch(test_subset, gen_icl_prompt, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": baseline_name,
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": self.instruction_model,
                "prediction_model": self.prediction_model,
                "instruction": gen_icl_prompt,
                "instruction_tokens": gen_icl_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_icl_only_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                             label_names: List[str], dataset_info: Dict[str, Any], requested_n: int, actual_n: int) -> List[Dict]:
        """Run ICL only baseline."""
        icl_only_prompt = in_context_learning_prompt(train_subset, label_names)
        icl_only_tokens = count_tokens_in_instruction(icl_only_prompt, self.prediction_model)
        
        print(f"[INFO] Baseline: naive+icl, Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_icl_prompt_batch(test_subset, icl_only_prompt, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": "naive+icl",
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": None,
                "prediction_model": self.prediction_model,
                "instruction": icl_only_prompt,
                "instruction_tokens": icl_only_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_api_generated_instruction_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                                             label_names: List[str], dataset_info: Dict[str, Any], 
                                             requested_n: int, actual_n: int, meta_prompt_template: str, baseline_name: str) -> List[Dict]:
        """Run generated instruction baseline using API model for instruction generation."""
        # Import the API helper functions
        from api_instruction_helper import generate_instruction_api, count_tokens_in_instruction
        
        # Use API for instruction generation with proper max_tokens
        gen_instruction = generate_instruction_api(train_subset, label_names, meta_prompt_template, self.instruction_model, MAX_TOKENS_INSTRUCTION)
        
        custom_line = f"Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"
        final_instruction = gen_instruction + "\n" + custom_line
        final_instruction_tokens = count_tokens_in_instruction(final_instruction, self.prediction_model)
        
        print(f"[INFO] Baseline: {baseline_name}, Instr model: {self.instruction_model} (API), Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_instruction_batch(test_subset, final_instruction, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": baseline_name,
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": self.instruction_model,
                "prediction_model": self.prediction_model,
                "instruction": final_instruction,
                "instruction_tokens": final_instruction_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_api_generated_instruction_icl_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                                                 label_names: List[str], dataset_info: Dict[str, Any], 
                                                 requested_n: int, actual_n: int, meta_prompt_template: str, baseline_name: str) -> List[Dict]:
        """Run generated instruction + ICL baseline using API model for instruction generation."""
        # Import the API helper functions
        from api_instruction_helper import generate_instruction_api, count_tokens_in_instruction
        
        # Use API for instruction generation with proper max_tokens
        gen_instruction = generate_instruction_api(train_subset, label_names, meta_prompt_template, self.instruction_model, MAX_TOKENS_INSTRUCTION)
        
        # No need for custom_line since in_context_learning_prompt already includes the constraint
        gen_icl_prompt = gen_instruction + "\n" + in_context_learning_prompt(train_subset, label_names)
        gen_icl_tokens = count_tokens_in_instruction(gen_icl_prompt, self.prediction_model)
        
        print(f"[INFO] Baseline: {baseline_name}, Instr model: {self.instruction_model} (API), Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_icl_prompt_batch(test_subset, gen_icl_prompt, self.prediction_model, self.prediction_vllm_url)
        
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": baseline_name,
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": self.instruction_model,
                "prediction_model": self.prediction_model,
                "instruction": gen_icl_prompt,
                "instruction_tokens": gen_icl_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results
    
    def run_GEPA_baseline(self, train_subset: List[Dict], test_subset: List[Dict], 
                         label_names: List[str], dataset_info: Dict[str, Any], 
                         requested_n: int, actual_n: int) -> List[Dict]:
        """
        Run GEPA baseline: use GEPA to optimize instruction, then classify test set.
        """
        import gepa
        from gepa.core.adapter import EvaluationBatch, GEPAAdapter
        custom_line = f"Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"

        # Split train_subset into train and val (1:2 ratio)  
        if len(train_subset) < 2:
            split_point = 1
        else:
            split_point = len(train_subset) // 3
        gepa_dataset = [{
                "input": ex["text"],
                "additional_context": {},
                "answer": ex["label"]
            } for ex in train_subset]
        if len(gepa_dataset) < 2:
            gepa_trainset = gepa_dataset
            gepa_valset = None
        else:
            gepa_trainset = gepa_dataset[:split_point]
            gepa_valset = gepa_dataset[split_point:]
        
        # Create classification adapter following GEPA's structure
        class ClassificationAdapter(GEPAAdapter):
            def __init__(self, prediction_model, prediction_vllm_url, label_names):
                self.prediction_model = prediction_model
                self.prediction_vllm_url = prediction_vllm_url
                self.label_names = label_names
                self.failure_score = 0.0
            
            def evaluate(self, batch, candidate, capture_traces=False):
                # Get the instruction from candidate (GEPA passes dict with instruction)
                system_content = list(candidate.values())[0] if candidate else ""
                
                # Apply instruction to batch
                messages_list = []
                for ex in batch:
                    user_content = f"Input: {ex['input']}"
                    messages = [{"role": "user", "content": system_content + custom_line + user_content}]
                    messages_list.append(messages)
                
                predictions = query_local_vllm_chat_batch(
                    messages_list, 
                    max_tokens=MAX_TOKENS_PREDICTION, 
                    temperature=0.0, 
                    model=self.prediction_model, 
                    vllm_url=self.prediction_vllm_url
                )
                
                # Calculate scores and outputs
                outputs = []
                scores = []
                trajectories = [] if capture_traces else None
                
                for ex, pred in zip(batch, predictions):
                    score = 1.0 if ex["answer"].lower().strip() == pred.lower().strip() else 0.0
                    
                    output = {"full_assistant_response": pred}
                    outputs.append(output)
                    scores.append(score)
                    
                    if capture_traces:
                        trajectory = {
                            "data": ex,
                            "full_assistant_response": pred
                        }
                        trajectories.append(trajectory)
                
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
            
            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                """Required method for GEPA adapter"""
                ret_d = {}
                comp = components_to_update[0] if components_to_update else "system_prompt"
                
                items = []
                if eval_batch.trajectories:
                    for traj, score, output in zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs):
                        data = traj["data"]
                        generated_output = traj["full_assistant_response"]
                        
                        if score > 0.0:
                            feedback = f"The generated response is correct. The response includes the correct answer '{data['answer']}'"
                        else:
                            feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}'. Ensure that the correct answer is included in the response exactly as it is."
                        
                        item = {
                            "Inputs": data["input"],
                            "Generated Outputs": generated_output,
                            "Feedback": feedback,
                        }
                        items.append(item)
                
                ret_d[comp] = items
                return ret_d
        
        # Create adapter
        adapter = ClassificationAdapter(self.prediction_model, self.prediction_vllm_url, label_names)
        
        # Start with naive instruction as seed
        seed_instruction = naive_instruction_prompt(label_names)
        seed_candidate = {"system_prompt": seed_instruction}
        
        print(f"[GEPA] Train set size: {len(gepa_trainset)}")
        print(f"[GEPA] Val set size: {len(gepa_valset) if gepa_valset else 0}")

        gepa_result = gepa.optimize(
                    seed_candidate=seed_candidate,
                    trainset=gepa_trainset,
                    valset=gepa_valset,
                    adapter=adapter,
                    task_lm=None,  
                    reflection_lm=lambda prompt: query_local_vllm_chat_batch(
                        [[{"role": "user", "content": prompt}]],
                        max_tokens=MAX_TOKENS_INSTRUCTION,
                        temperature=0.0,
                        model=self.instruction_model,
                        vllm_url=self.instruction_vllm_url
                    )[0],
                    max_metric_calls=150,  
                    reflection_minibatch_size=min(3, len(gepa_trainset)),  
                )
        
        # Get the optimized instruction
        optimized_instruction = gepa_result.best_candidate['system_prompt'] + custom_line
        instruction_tokens = count_tokens_in_instruction(optimized_instruction, self.prediction_model)
        
        print(f"[GEPA] Optimization complete. Best score: {max(gepa_result.val_aggregate_scores) if gepa_result.val_aggregate_scores else 'N/A'}")
        print(f"[GEPA] Optimized instruction:\n{optimized_instruction}\n")
        
        # Apply optimized instruction to test set
        print(f"[INFO] Baseline: GEPA, Instr model: {self.instruction_model}, Prediction model: {self.prediction_model}, n: {actual_n} (requested: {requested_n})")
        preds = apply_instruction_batch(test_subset, optimized_instruction, self.prediction_model, self.prediction_vllm_url)
         
        # Format results
        results = []
        for ex_idx, (ex, pred) in enumerate(zip(test_subset, preds)):
            results.append({
                **dataset_info,
                "baseline": "GEPA",
                "n": requested_n,
                "actual_n": actual_n,
                "instruction_model": self.instruction_model,
                "prediction_model": self.prediction_model,
                "instruction": optimized_instruction,
                "instruction_tokens": instruction_tokens,
                "input": ex["text"],
                "answer": ex["label"],
                "prediction": pred,
                "correct": ex["label"].lower().strip() == pred.lower().strip()
            })
        return results

def load_existing_results(output_path: str) -> List[Dict]:
    """Load existing results if they exist."""
    if os.path.exists(output_path):
        try:
            print(f"[INFO] Loading existing results from {output_path}")
            existing_dataset = load_from_disk(output_path)
            return existing_dataset.to_pandas().to_dict('records')
        except (FileNotFoundError, ValueError) as e:
            print(f"[INFO] Directory exists but is not a valid Dataset directory: {e}")
            return []
    return []

def save_results(results: List[Dict], output_path: str):
    """Save results to timestamped subdirectory (no append mode)."""
    import time
    timestamp = int(time.time())
    run_path = f"{output_path}/data/run_{timestamp}"
    os.makedirs(run_path, exist_ok=True)
    
    results_dataset = Dataset.from_list(results)
    results_dataset.save_to_disk(run_path)
    print(f"[INFO] Saved {len(results)} results to {run_path}")

def main():
    parser = argparse.ArgumentParser(description="Run modular baseline eval v3 with vLLM chat API.")
    parser.add_argument('--eval_samples', type=int, default=200, help='Number of test examples to evaluate')
    parser.add_argument('--examples_to_test', type=str, default="5,10,20,50,100", help='Comma-separated list of n-shot values')
    parser.add_argument('--instruction_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help='Model name for instruction generation')
    parser.add_argument('--prediction_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help='Model name for prediction')
    parser.add_argument('--trained_instruction_model', type=str, default=None,
                       help='Trained model path for instruction generation (for trained_generated_instruction baseline)')
    parser.add_argument('--custom_baseline_name', type=str, default="trained_generated_instruction",
                       help='Custom name for the baseline (default: trained_generated_instruction)')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000/v1/chat/completions", 
                       help='vLLM server chat completions endpoint for instruction model')
    parser.add_argument('--prediction_vllm_url', type=str, default=None,
                       help='vLLM server chat completions endpoint for prediction model (uses instruction URL if not specified)')
    parser.add_argument('--dataset_path', type=str, default="/data/user_data/emilyx/prompt_gen/v3_3_processed/validation", 
                       help='Path to dataset')
    parser.add_argument('--output_path', type=str, default="/data/user_data/emilyx/prompt_gen/baseline_eval_v3_results", 
                       help='Path to save results')
    parser.add_argument('--max_datasets', type=int, default=20, 
                       help='Maximum number of datasets/configs to process (randomly sampled if dataset is larger)')
    parser.add_argument('--baselines', type=str, default="naive,generated_instruction,generated_instruction+icl,naive+icl,generated_instruction2,generated_instruction2+icl,generated_instruction3,generated_instruction4,trained_generated_instruction,trained_generated_instruction3,api_generated_instruction3", 
                       help='Comma-separated list of baselines to run')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible dataset/example selection')
    parser.add_argument('--max_tokens_instruction', type=int, default=1024, help='Max tokens for instruction generation')
    parser.add_argument('--max_tokens_prediction', type=int, default=10, help='Max tokens for prediction generation')
    parser.add_argument('--context_length', type=int, default=None, help='Override model context length (in tokens)')
    parser.add_argument('--safety_margin', type=int, default=1000, help='Safety margin for context length calculations (in tokens)')
    args = parser.parse_args()

    EVAL_SAMPLES = args.eval_samples
    EXAMPLES_TO_TEST = [int(x) for x in args.examples_to_test.split(",")]
    INSTRUCTION_MODEL = args.instruction_model
    PREDICTION_MODEL = args.prediction_model
    TRAINED_INSTRUCTION_MODEL = args.trained_instruction_model
    VLLM_URL = args.vllm_url
    PREDICTION_VLLM_URL = args.prediction_vllm_url
    dataset_path = args.dataset_path
    output_path = args.output_path
    MAX_DATASETS = args.max_datasets
    BASELINES_TO_RUN = args.baselines.split(",")
    
    # Update global token limits
    global MAX_TOKENS_INSTRUCTION, MAX_TOKENS_PREDICTION
    MAX_TOKENS_INSTRUCTION = args.max_tokens_instruction
    MAX_TOKENS_PREDICTION = args.max_tokens_prediction
    
    # Update context length constants if overridden
    global SAFETY_MARGIN, DEFAULT_CONTEXT_LENGTH
    SAFETY_MARGIN = args.safety_margin
    
    if args.context_length:
        DEFAULT_CONTEXT_LENGTH = args.context_length
        print(f"[INFO] Setting context length to {args.context_length}")
    
    # Set random seed for reproducible dataset/example selection
    random.seed(args.random_seed)

    print(f"[INFO] Running baselines: {BASELINES_TO_RUN}")
    print(f"[INFO] Instruction model: {INSTRUCTION_MODEL}")
    print(f"[INFO] Prediction model: {PREDICTION_MODEL}")
    if TRAINED_INSTRUCTION_MODEL:
        print(f"[INFO] Trained instruction model: {TRAINED_INSTRUCTION_MODEL}")
    
    # Show context length configuration
    print(f"[INFO] Context length: {DEFAULT_CONTEXT_LENGTH} tokens")
    print(f"[INFO] Safety margin: {SAFETY_MARGIN} tokens")
    print(f"[INFO] Available context for examples: {DEFAULT_CONTEXT_LENGTH - SAFETY_MARGIN} tokens")

    os.makedirs(output_path, exist_ok=True)

    # Initialize baseline runner
    runner = BaselineRunner(INSTRUCTION_MODEL, PREDICTION_MODEL, VLLM_URL, PREDICTION_VLLM_URL)
    
    # Update context length manager with the configured values
    runner.context_manager.context_length = DEFAULT_CONTEXT_LENGTH
    runner.context_manager.safety_margin = SAFETY_MARGIN

    print(f"[INFO] Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if len(dataset) > MAX_DATASETS:
        dataset = dataset.select(random.sample(range(len(dataset)), MAX_DATASETS))
    print(f"[INFO] Number of dataset/config rows: {len(dataset)}")
    
    all_results = []

    print(f"[INFO] Using vLLM server for instruction model: {INSTRUCTION_MODEL}, prediction model: {PREDICTION_MODEL}")
    
    for row_idx, row in enumerate(tqdm(dataset, desc=f"Datasets/Configs [{INSTRUCTION_MODEL} -> {PREDICTION_MODEL}]")):
        dataset_name = row.get("dataset_name", "")
        config_name = row.get("config_name", "")
        input_column = row.get("input_column", "")
        label_column = row.get("label_column", "")
        label_names = row["label_names"]
        train_examples = row["train_examples"]
        test_examples = row["test_examples"]

        print(f"[INFO] Processing row {row_idx}: dataset={dataset_name}, config={config_name}, train={len(train_examples)}, test={len(test_examples)}")

        # Always sample EVAL_SAMPLES test examples
        if len(test_examples) > EVAL_SAMPLES:
            test_subset = random.sample(test_examples, EVAL_SAMPLES)
        else:
            test_subset = test_examples

        dataset_info = {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "input_column": input_column,
            "label_column": label_column,
        }

        # Shuffle train examples once at the beginning (after seed is set)
        shuffled_train_examples = train_examples.copy()
        random.shuffle(shuffled_train_examples)
        
        for n in EXAMPLES_TO_TEST:
            if len(shuffled_train_examples) < n:
                print(f"[INFO] Skipping n={n} (not enough train examples)")
                continue
            
            # Create dataset key for context length caching
            dataset_key = f"{dataset_name}_{config_name}"
            
            # Get adjusted n value that fits within context
            actual_n, requested_n = runner.context_manager.get_adjusted_n(
                n, shuffled_train_examples, META_PROMPT_V3, label_names, dataset_key
            )
            
            # Skip if no examples can fit
            if actual_n == 0:
                print(f"[WARNING] Skipping n={n} - no examples fit within context length")
                continue
            
            train_subset = shuffled_train_examples[:actual_n]
            print(f"[INFO] Running baselines for n={actual_n} (requested: {requested_n})")

            # Run selected baselines
            for baseline in BASELINES_TO_RUN:
                baseline = baseline.strip()
                
                if baseline == "naive":
                    results = runner.run_naive_baseline(test_subset, label_names, dataset_info, requested_n, actual_n)
                
                elif baseline == "generated_instruction":
                    results = runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V3, "generated_instruction"
                    )
                
                elif baseline == "generated_instruction+icl":
                    results = runner.run_generated_instruction_icl_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V3, "generated_instruction+icl"
                    )
                
                elif baseline == "naive+icl":
                    results = runner.run_icl_only_baseline(train_subset, test_subset, label_names, dataset_info, requested_n, actual_n)
                
                elif baseline == "generated_instruction2":
                    results = runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V4, "generated_instruction2"
                    )
                
                elif baseline == "generated_instruction2+icl":
                    results = runner.run_generated_instruction_icl_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V4, "generated_instruction2+icl"
                    )
                
                elif baseline == "generated_instruction3":
                    results = runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V5, "generated_instruction3"
                    )
                
                elif baseline == "generated_instruction3+icl":
                    results = runner.run_generated_instruction_icl_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V5, "generated_instruction3+icl"
                    )
                
                elif baseline == "generated_instruction4":
                    results = runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V6, "generated_instruction4"
                    )
                
                elif baseline == "trained_generated_instruction":
                    if TRAINED_INSTRUCTION_MODEL is None:
                        print(f"[WARNING] trained_generated_instruction baseline requested but --trained_instruction_model not provided, skipping...")
                        continue
                    
                    # Use custom baseline name if provided
                    baseline_name = args.custom_baseline_name
                    print(f"[INFO] Using custom trained baseline name: {baseline_name}")
                    
                    # Create a temporary runner with trained model for instruction generation
                    trained_runner = BaselineRunner(TRAINED_INSTRUCTION_MODEL, PREDICTION_MODEL, VLLM_URL, PREDICTION_VLLM_URL)
                    results = trained_runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V3, baseline_name
                    )
                
                elif baseline == "trained_generated_instruction3":
                    if TRAINED_INSTRUCTION_MODEL is None:
                        print(f"[WARNING] trained_generated_instruction3 baseline requested but --trained_instruction_model not provided, skipping...")
                        continue
                    
                    # Use custom baseline name if provided
                    baseline_name = args.custom_baseline_name
                    print(f"[INFO] Using custom trained baseline name: {baseline_name}")
                    
                    # Create a temporary runner with trained model for instruction generation
                    trained_runner = BaselineRunner(TRAINED_INSTRUCTION_MODEL, PREDICTION_MODEL, VLLM_URL, PREDICTION_VLLM_URL)
                    results = trained_runner.run_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V5, baseline_name  # Use META_PROMPT_V5 like generated_instruction3
                    )
                
                elif baseline == "api_generated_instruction":
                    # Handle API-based instruction generation baseline
                    print(f"[INFO] Running API-based instruction generation baseline: {baseline}")
                    
                    # Use custom baseline name if provided
                    baseline_name = args.custom_baseline_name
                    print(f"[INFO] Using custom baseline name: {baseline_name}")
                    
                    # Use META_PROMPT_V3 for API-based instruction generation
                    results = runner.run_api_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V3, baseline_name
                    )
                
                elif baseline == "api_generated_instruction3":
                    # Handle API-based instruction generation baseline using META_PROMPT_V5 (same as generated_instruction3)
                    print(f"[INFO] Running API-based instruction generation baseline: {baseline}")
                    
                    # Use custom baseline name if provided
                    baseline_name = args.custom_baseline_name
                    print(f"[INFO] Using custom baseline name: {baseline_name}")
                    
                    # Use META_PROMPT_V5 for API-based instruction generation (same as generated_instruction3)
                    results = runner.run_api_generated_instruction_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V5, baseline_name
                    )
                
                elif baseline == "api_generated_instruction3+icl":
                    # Handle API-based instruction generation + ICL baseline using META_PROMPT_V5
                    print(f"[INFO] Running API-based instruction generation + ICL baseline: {baseline}")
                    
                    # Use custom baseline name if provided
                    baseline_name = args.custom_baseline_name
                    print(f"[INFO] Using custom baseline name: {baseline_name}")
                    
                    # Use META_PROMPT_V5 for API-based instruction generation + ICL
                    results = runner.run_api_generated_instruction_icl_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n, 
                        META_PROMPT_V5, baseline_name
                    )
                elif baseline == "GEPA":
                    results = runner.run_GEPA_baseline(
                        train_subset, test_subset, label_names, dataset_info, requested_n, actual_n
                    )

                else:
                    print(f"[WARNING] Unknown baseline: {baseline}, skipping...")
                    continue

                all_results.extend(results)
                print(f"[INFO] Completed baseline: {baseline} with n={actual_n} (requested: {requested_n})")
                # print the first 5 results
                for i, result in enumerate(results[:5]):
                    status = "✅ CORRECT" if result["correct"] else "❌ WRONG"
                    print(f"\n[{baseline} n={actual_n} (requested: {requested_n}) sample {i+1}] {status}")
                    for key, value in result.items():
                        print(f"{key}: {value}")
                    print("-" * 60)
    # Save all results at the end
    save_results(all_results, output_path)
    # Print context adjustment summary
    print(f"\n[CONTEXT_ADJUSTMENT_SUMMARY]")
    print(f"Total results: {len(all_results)}")
    
    # Count adjustments
    adjustments = 0
    for result in all_results:
        if result.get("n", 0) != result.get("actual_n", 0):
            adjustments += 1
    
    if adjustments > 0:
        print(f"Context adjustments made: {adjustments} results")
        print(f"Adjustment rate: {adjustments/len(all_results)*100:.1f}%")
    else:
        print("No context adjustments were needed")
    
    print(f"[INFO] Done. Results saved to {output_path}")

if __name__ == "__main__":
    main() 