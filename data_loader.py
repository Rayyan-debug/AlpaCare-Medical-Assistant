from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def load_and_preprocess_data():
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
        'valid': test_valid['train']
    }
    
    print(f"Training samples: {len(dataset_splits['train'])}")
    print(f"Testing samples: {len(dataset_splits['test'])}")
    print(f"Validation samples: {len(dataset_splits['valid'])}")
    
    return dataset_splits

def format_with_disclaimer(instruction, response):
    """Format training examples with medical disclaimer"""
    disclaimer = "Important: This is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
    
    formatted_text = f"### Instruction: {instruction}\n\n### Response: {response}\n\n{disclaimer}"
    return formatted_text

# Test the functions
if __name__ == "__main__":
    data = load_and_preprocess_data()
    sample_instruction = "What are the symptoms of diabetes?"
    sample_response = "Common symptoms include increased thirst, frequent urination, and fatigue."
    print("\nSample formatted text:")
    print(format_with_disclaimer(sample_instruction, sample_response))
