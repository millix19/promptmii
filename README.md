# PromptMII: Meta-Learning Instruction Induction for LLMs

**PromptMII** automatically generates task-specific instructions for classification tasks, achieving many-shot in-context learning (ICL) performance while using 3-13× fewer tokens.

[![Paper](https://img.shields.io/badge/Paper-ArXiv-blue)](https://arxiv.org/abs/2510.16932)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PromptMII-yellow)](https://huggingface.co/datasets/milli19/promptmii-dataset)
[![Models](https://img.shields.io/badge/🤗%20Models-PromptMII-yellow)](https://huggingface.co/milli19)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/promptmii.git
cd promptmii
pip install vllm pandas scikit-learn datasets
```

### Using PromptMII on Your Dataset

```python
from promptmii import PromptMII


# Initialize with our trained model
promptmii = PromptMII(
    instruction_model="milli19/promptmii-llama-3.1-8b-instruct",
    prediction_model="meta-llama/Llama-3.1-8B-Instruct"
)

# Run on your classification dataset
results = promptmii.run(
    train_dataset=your_train_data,  # List of {"text": ..., "label": ...} or HF dataset
    test_dataset=your_test_data,
    text_column="text",
    label_column="label"
)

print(f"F1 Score: {results['f1_score']:.3f}")
print(f"Generated Instruction: {results['instruction']}")
```

See [`test_promptmii.py`](./test_promptmii.py) for usage examples.


## 1. **PromptMII Dataset**

- Step 1: Initial Preprocessing (`run_prompt_gen_preprocess.sh`)
  -  Fetches public text classification datasets from HuggingFace

- Step 2: Postprocessing (`run_prompt_gen_postprocess.sh`)
  - Filter and splits preprocessed data for training and eval
  - Outputs saved to [HuggingFace](https://huggingface.co/datasets/milli19/promptmii-dataset)

- Step 3: VERL Format Conversion for Training (`run_verl_preprocess.sh`)
  - Converts data to VERL training format (< 5min runtime)


## 2. **Reproduce Training Results**

Environment setup: https://verl.readthedocs.io/en/latest/start/install.html#requirements

```bash
sbatch prompt_gen_grpo_flame_sglang_f1_explore.sh
```

**Trained PromptMII models:**
- Qwen: [`milli19/promptmii-qwen-2.5-7b-instruct`](https://huggingface.co/milli19/promptmii-qwen-2.5-7b-instruct)
- Llama: [`milli19/promptmii-llama-3.1-8b-instruct`](https://huggingface.co/milli19/promptmii-llama-3.1-8b-instruct)

## 3. **Reproduce Evaluation Results**

Install SGLang for evaluation:

```bash
pip install "sglang[all]"
pip install sglang-router
```

Choose the evaluation script based on what you want to test:

```bash
# Compare all baselines (naive, ICL, PromptMII-Zero)
sbatch run_baselines_modular_sglang.sh

# Evaluate trained PromptMII models
sbatch add_trained_instruction_baseline_sglang.sh

# Evaluate API-based instruction generation (GPT-4, Claude, etc.)
sbatch add_api_instruction_baseline_sglang.sh
```

## Acknowledgements

Training code is based on [VeRL](https://github.com/volcengine/verl)

## Citation
```
@misc{xiao2025promptmiimetalearninginstructioninduction,
      title={Prompt-MII: Meta-Learning Instruction Induction for LLMs}, 
      author={Emily Xiao and Yixiao Zeng and Ada Chen and Chin-Jou Li and Amanda Bertsch and Graham Neubig},
      year={2025},
      eprint={2510.16932},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.16932}, 
}
```
