# data_loader.py
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def load_medical_dataset():
    """Load and preprocess the medical instruction dataset"""
    print("Loading AlpaCare-MedInstruct-52k dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k")
    
    # Split dataset (90/5/5)
    train_testvalid = dataset['train'].train_test_split(test_size=0.1, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_splits = {
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']
    }
    
    print(f"Train: {len(dataset_splits['train'])} samples")
    print(f"Test: {len(dataset_splits['test'])} samples") 
    print(f"Validation: {len(dataset_splits['validation'])} samples")
    
    return dataset_splits

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess dataset for training"""
    
    # Add medical disclaimer to every output
    disclaimed_outputs = []
    for output in examples['output']:
        disclaimed_output = output + "\n\n--- MEDICAL DISCLAIMER ---\nThis is for educational purposes only. Please consult a qualified healthcare professional for medical advice."
        disclaimed_outputs.append(disclaimed_output)
    
    # Tokenize inputs and outputs
    model_inputs = tokenizer(
        examples['instruction'],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        disclaimed_outputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    # Test the data loader
    dataset = load_medical_dataset()
    print("Dataset loaded successfully!")
