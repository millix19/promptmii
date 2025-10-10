"""
PromptMII: Meta-Learning Instruction Induction for LLMs

A simple API to use PromptMII for automatic instruction generation on classification tasks.
Requires only 1 GPU - runs sequentially (generate instruction, then evaluate).
"""

from typing import List, Dict, Any, Optional, Union
from datasets import Dataset
from sklearn.metrics import f1_score
import pandas as pd
from vllm import LLM, SamplingParams

# Meta-prompt templates (model-specific from paper)
# Qwen models use META_PROMPT_QWEN
META_PROMPT_QWEN = (
    "You are designing a clear instruction for a data annotator to classify text inputs into one of these labels: {label_names_str}\n\n"
    "Here are some example inputs and their correct labels:\n{examples}\n\n"
    "Your task is to write a concise instruction that:\n"
    "- Defines the classification task and clearly explains the meaning of each label.\n"
    "- Provides general labeling strategies and decision rules so annotators can correctly handle unseen inputs.\n"
    "- Highlights common pitfalls, tricky edge cases, and misconceptions to reduce labeling errors.\n"
    "- Keeps the instruction reasonably concise and focused — avoid unnecessary repetition or overly long explanations.\n"
)

# Llama models use META_PROMPT_LLAMA
META_PROMPT_LLAMA = (
    "You are helping to create a prompt for a language model to classify text inputs. "
    "The model should choose one label from the following options: {label_names_str}.\n\n"
    "Here are some example inputs and their correct labels:\n{examples}\n\n"
    "Write an instruction that:\n"
    "- Describes the classification task in a way that generalizes to new inputs.\n"
    "- Points out any useful clues or strategies for making the decision.\n"
    "- Clearly tells the model to respond with only the label name, and not to include any explanation or additional text.\n\n"
    "Provide only the instruction, not the examples or labels."
)


class PromptMII:
    """
    PromptMII: Automatic instruction generation for classification tasks.

    Example usage:
        ```python
        from datasets import load_dataset
        from promptmii import PromptMII

        # Load your dataset
        dataset = load_dataset("ag_news", split="test")

        # Initialize PromptMII
        promptmii = PromptMII(
            instruction_model="milli19/promptmii-qwen-2.5-7b-instruct",
            prediction_model="Qwen/Qwen2.5-7B-Instruct"
        )

        # Run on your dataset
        results = promptmii.run(
            train_dataset=dataset[:100],
            test_dataset=dataset[100:200],
            text_column="text",
            label_column="label"
        )

        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"Generated Instruction: {results['instruction']}")
        results['predictions_df'].to_csv("predictions.csv")
        ```
    """

    def __init__(
        self,
        instruction_model: str = "milli19/promptmii-llama-3.1-8b-instruct",
        prediction_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization: float = 0.7,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize PromptMII.

        Args:
            instruction_model: HuggingFace model name for instruction generation
                             Llama models: "milli19/promptmii-llama-3.1-8b-instruct"
                             Qwen models: "milli19/promptmii-qwen-2.5-7b-instruct"
                             Or use base models
            prediction_model: HuggingFace model name for making predictions
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            tensor_parallel_size: Tensor parallelism size (default: 1)
        """
        self.instruction_model_name = instruction_model
        self.prediction_model_name = prediction_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

    def run(
        self,
        train_dataset: Union[Dataset, List[Dict], pd.DataFrame],
        test_dataset: Union[Dataset, List[Dict], pd.DataFrame],
        text_column: str = "text",
        label_column: str = "label",
        n_train_examples: Optional[int] = None,
        label_names: Optional[List[str]] = None,
        meta_prompt_template: Optional[str] = None,
        save_predictions_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run PromptMII on your classification dataset.

        Args:
            train_dataset: Training examples for instruction generation
            test_dataset: Test examples for evaluation
            text_column: Name of the text/input column
            label_column: Name of the label column
            n_train_examples: Number of training examples to use (default: use all)
            label_names: List of label names (auto-detected if None)
            meta_prompt_template: Meta-prompt template to use (auto-selected based on model if None)
            save_predictions_path: Path to save detailed predictions CSV (optional)

        Returns:
            Dictionary containing:
                - f1_score: Macro F1 score
                - accuracy: Accuracy score
                - instruction: Generated instruction
                - instruction_tokens: Number of tokens in instruction
                - predictions_df: DataFrame with all predictions
        """
        # Convert inputs to standard format
        train_examples = self._convert_to_dict_list(train_dataset, text_column, label_column)
        test_examples = self._convert_to_dict_list(test_dataset, text_column, label_column)

        # Auto-detect label names if not provided
        if label_names is None:
            label_names = self._auto_detect_labels(train_examples + test_examples)

        # Sample training examples if needed
        if n_train_examples is not None and n_train_examples < len(train_examples):
            import random
            train_examples = random.sample(train_examples, n_train_examples)

        # Auto-select meta-prompt template based on model if not provided
        if meta_prompt_template is None:
            if "llama" in self.instruction_model_name.lower():
                meta_prompt_template = META_PROMPT_LLAMA
                print(f"[PromptMII] Auto-selected Llama meta-prompt template")
            else:
                meta_prompt_template = META_PROMPT_QWEN
                print(f"[PromptMII] Auto-selected Qwen meta-prompt template")

        print(f"[PromptMII] Using {len(train_examples)} training examples, {len(test_examples)} test examples")
        print(f"[PromptMII] Label set: {label_names}")

        # Step 1: Generate instruction using PromptMII
        print(f"\n[PromptMII] Step 1: Loading instruction model ({self.instruction_model_name})...")
        instruction = self._generate_instruction(train_examples, label_names, meta_prompt_template)

        # Add custom enforcement line (as per paper)
        custom_line = f"Only return one of these options: {', '.join(label_names)}. Do not output \"Label:\" or any extra text.\n\n"
        final_instruction = instruction + "\n" + custom_line

        print(f"[PromptMII] Generated instruction:\n{final_instruction}\n")

        # Step 2: Make predictions using generated instruction
        print(f"[PromptMII] Step 2: Loading prediction model ({self.prediction_model_name})...")
        predictions, instruction_tokens = self._make_predictions(test_examples, final_instruction)

        print(f"[PromptMII] Instruction tokens: {instruction_tokens}")

        # Step 3: Compute metrics
        ground_truth = [ex["label"] for ex in test_examples]
        f1 = self._compute_f1(predictions, ground_truth, label_names)
        acc = self._compute_accuracy(predictions, ground_truth)

        print(f"[PromptMII] Results: F1={f1:.3f}, Accuracy={acc:.3f}")

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            "input": [ex["text"] for ex in test_examples],
            "ground_truth": ground_truth,
            "prediction": predictions,
            "correct": [g.lower().strip() == p.lower().strip() for g, p in zip(ground_truth, predictions)]
        })

        # Save if requested
        if save_predictions_path:
            predictions_df.to_csv(save_predictions_path, index=False)
            print(f"[PromptMII] Saved predictions to {save_predictions_path}")

        # Prepare results
        results = {
            "f1_score": f1,
            "accuracy": acc,
            "instruction": final_instruction,
            "instruction_tokens": instruction_tokens,
            "label_names": label_names,
            "n_train_examples": len(train_examples),
            "n_test_examples": len(test_examples),
            "predictions_df": predictions_df
        }

        return results

    def _convert_to_dict_list(self, dataset, text_column: str, label_column: str) -> List[Dict]:
        """Convert various dataset formats to list of dicts."""
        if isinstance(dataset, Dataset):
            return [{"text": row[text_column], "label": str(row[label_column])} for row in dataset]
        elif isinstance(dataset, pd.DataFrame):
            return [{"text": row[text_column], "label": str(row[label_column])} for _, row in dataset.iterrows()]
        elif isinstance(dataset, list):
            if isinstance(dataset[0], dict):
                return [{"text": row[text_column], "label": str(row[label_column])} for row in dataset]
            else:
                raise ValueError("List items must be dictionaries")
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def _auto_detect_labels(self, examples: List[Dict]) -> List[str]:
        """Auto-detect unique labels from examples."""
        unique_labels = sorted(list(set(ex["label"] for ex in examples)))
        return unique_labels

    def _generate_instruction(self, train_examples: List[Dict], label_names: List[str], meta_prompt_template: str) -> str:
        """Generate instruction using meta-prompt."""
        # Format training examples (same format as baseline evaluation)
        examples_str = ""
        for ex in train_examples:
            examples_str += f'Input: {ex["text"]}\nLabel: {ex["label"]}\n'

        # Create meta-prompt
        label_names_str = ", ".join(label_names)
        meta_prompt = meta_prompt_template.format(examples=examples_str, label_names_str=label_names_str)

        # Load instruction model
        llm = LLM(
            model=self.instruction_model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True
        )

        # Generate instruction using chat format
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

        # Apply chat template (same as vLLM server does)
        tokenizer = llm.get_tokenizer()
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": meta_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = llm.generate([chat_prompt], sampling_params)
        instruction = outputs[0].outputs[0].text.strip()

        # Free GPU memory
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        return instruction

    def _make_predictions(self, test_examples: List[Dict], instruction: str) -> tuple[List[str], int]:
        """Make predictions using the generated instruction."""
        # Load prediction model
        llm = LLM(
            model=self.prediction_model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True
        )

        # Get tokenizer
        tokenizer = llm.get_tokenizer()

        # Count instruction tokens
        instruction_tokens = len(tokenizer.encode(instruction))

        # Create prompts with chat format (same as vLLM server)
        chat_prompts = []
        for ex in test_examples:
            user_content = instruction + f"Input: {ex['text']}"
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True
            )
            chat_prompts.append(chat_prompt)

        # Generate predictions
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(chat_prompts, sampling_params)
        predictions = [output.outputs[0].text.strip() for output in outputs]

        # Free GPU memory
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        return predictions, instruction_tokens

    def _compute_f1(self, predictions: List[str], ground_truth: List[str], label_names: List[str]) -> float:
        """Compute macro F1 score."""
        label_to_idx = {label.lower().strip(): i for i, label in enumerate(label_names)}

        pred_indices = []
        gt_indices = []

        for pred, gt in zip(predictions, ground_truth):
            pred_clean = pred.lower().strip()
            gt_clean = gt.lower().strip()

            pred_idx = label_to_idx.get(pred_clean, 0)
            gt_idx = label_to_idx.get(gt_clean, 0)

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

        try:
            return f1_score(gt_indices, pred_indices, average='macro', zero_division=0)
        except:
            return 0.0

    def _compute_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute accuracy."""
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p.lower().strip() == g.lower().strip())
        return correct / len(predictions) if predictions else 0.0


# Convenience function for quick usage
def run_promptmii(
    train_dataset,
    test_dataset,
    text_column: str = "text",
    label_column: str = "label",
    instruction_model: str = "milli19/promptmii-llama-3.1-8b-instruct",
    prediction_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick convenience function to run PromptMII.

    Example:
        ```python
        from promptmii import run_promptmii
        from datasets import load_dataset

        dataset = load_dataset("ag_news", split="test")
        results = run_promptmii(
            train_dataset=dataset[:100],
            test_dataset=dataset[100:200]
        )
        print(f"F1: {results['f1_score']:.3f}")
        print(f"Instruction: {results['instruction']}")
        ```
    """
    promptmii = PromptMII(
        instruction_model=instruction_model,
        prediction_model=prediction_model
    )
    return promptmii.run(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        text_column=text_column,
        label_column=label_column,
        **kwargs
    )
