"""
Quick test script for PromptMII to verify it works correctly.
"""

from promptmii import PromptMII
from datasets import load_dataset

def test_basic():
    """Test with a simple custom dataset"""
    print("=" * 80)
    print("Testing PromptMII with simple sentiment classification")
    print("=" * 80)

    # Simple test data
    train_data = [
        {"text": "I love this product! It's amazing!", "label": "positive"},
        {"text": "Great quality and fast shipping", "label": "positive"},
        {"text": "Best purchase I've made this year", "label": "positive"},
        {"text": "Terrible product, complete waste of money", "label": "negative"},
        {"text": "Very disappointed, does not work as advertised", "label": "negative"},
        {"text": "Poor quality, broke after one use", "label": "negative"},
    ]

    test_data = [
        {"text": "Excellent service and product!", "label": "positive"},
        {"text": "Highly recommend, very satisfied", "label": "positive"},
        {"text": "Awful experience, would not buy again", "label": "negative"},
        {"text": "Completely broken on arrival", "label": "negative"},
    ]

    # Initialize PromptMII (defaults to Llama)
    print("\nInitializing PromptMII with Llama model...")
    promptmii = PromptMII(
        instruction_model="milli19/promptmii-llama-3.1-8b-instruct",
        prediction_model="meta-llama/Llama-3.1-8B-Instruct"
    )

    # Run evaluation
    print("\nRunning PromptMII...")
    results = promptmii.run(
        train_dataset=train_data,
        test_dataset=test_data,
        text_column="text",
        label_column="label",
        save_predictions_path="test_predictions.csv"
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"F1 Score: {results['f1_score']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Instruction Tokens: {results['instruction_tokens']}")
    print(f"\nGenerated Instruction:")
    print("-" * 80)
    print(results['instruction'])
    print("-" * 80)

    print(f"\nPredictions:")
    print(results['predictions_df'].to_string())

    print(f"\nPredictions saved to: test_predictions.csv")
    print("\n✅ Test completed successfully!")

def test_qwen():
    """Test with Qwen model"""
    print("\n" + "=" * 80)
    print("Testing PromptMII with Qwen model")
    print("=" * 80)

    train_data = [
        {"text": "The movie was fantastic!", "label": "positive"},
        {"text": "Boring and too long", "label": "negative"},
    ]

    test_data = [
        {"text": "Amazing film!", "label": "positive"},
        {"text": "Waste of time", "label": "negative"},
    ]

    print("\nInitializing PromptMII with Qwen model...")
    promptmii = PromptMII(
        instruction_model="milli19/promptmii-qwen-2.5-7b-instruct",
        prediction_model="Qwen/Qwen2.5-7B-Instruct"
    )

    results = promptmii.run(
        train_dataset=train_data,
        test_dataset=test_data,
        text_column="text",
        label_column="label"
    )

    print(f"\nQwen Results: F1={results['f1_score']:.3f}, Accuracy={results['accuracy']:.3f}")
    print("\n✅ Qwen test completed successfully!")

def test_sst2():
    """Test with SST-2 dataset from Hugging Face"""
    print("\n" + "=" * 80)
    print("Testing PromptMII with SST-2 dataset from Hugging Face")
    print("=" * 80)

    # Load SST-2 dataset
    print("\nLoading SST-2 dataset from Hugging Face...")
    dataset = load_dataset("stanfordnlp/sst2")

    # Use a small subset for quick testing
    train_subset = dataset["train"].select(range(20))
    test_subset = dataset["validation"].select(range(200))

    print("\nInitializing PromptMII with Llama model...")
    promptmii = PromptMII(
        instruction_model="milli19/promptmii-llama-3.1-8b-instruct",
        prediction_model="meta-llama/Llama-3.1-8B-Instruct"
    )

    # Run evaluation
    print("\nRunning PromptMII on SST-2...")
    results = promptmii.run(
        train_dataset=train_subset,
        test_dataset=test_subset,
        text_column="sentence",
        label_column="label",
        save_predictions_path="sst2_predictions.csv"
    )

    # Display results
    print("\n" + "=" * 80)
    print("SST-2 RESULTS")
    print("=" * 80)
    print(f"F1 Score: {results['f1_score']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Instruction Tokens: {results['instruction_tokens']}")
    print(f"\nGenerated Instruction:")
    print("-" * 80)
    print(results['instruction'])
    print("-" * 80)

    print(f"\nPredictions saved to: sst2_predictions.csv")
    print("\n✅ SST-2 test completed successfully!")

if __name__ == "__main__":
    # Run basic test
    # test_basic()

    # Uncomment to test Qwen model
    # test_qwen()

    # Uncomment to test SST-2 dataset
    test_sst2()
