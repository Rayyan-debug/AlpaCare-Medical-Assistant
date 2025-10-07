# AlpaCare Medical Instruction Assistant

## Solar Industries India Ltd - Internship Assessment

### Project Overview
This project fine-tunes a pre-trained language model on medical instructions to create a safe, educational medical assistant that provides health information without diagnosis or prescription.

### 🚨 Safety Features
- **No medical diagnosis**
- **No prescription/dosage recommendations** 
- **Automatic medical disclaimer in every response**
- **Educational focus only**

### Architecture & Approach
- **Base Model**: microsoft/DialoGPT-medium (345M parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Dataset**: lavita/AlpaCare-MedInstruct-52k
- **Safety**: Built-in medical disclaimers and prompt engineering

### Project Structure
AlpaCare-Medical-Assistant/
├── data_loader.py # Data loading and preprocessing
├── requirements.txt # Python dependencies
├── human_evaluation.csv # Human evaluation results
├── REPORT_OUTLINE.md # Project report outline
├── notebooks/
│ ├── colab-finetune.ipynb # Training notebook
│ └── inference_demo.ipynb # Demo notebook
└── README.md


### How to Run

#### 1. Training (Google Colab)
1. Open `notebooks/colab-finetune.ipynb` in Google Colab
2. Ensure GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells sequentially
4. Download the generated `alpacare-lora-adapter.zip`

#### 2. Inference Demo
1. Open `notebooks/inference_demo.ipynb` in Google Colab  
2. Upload your trained adapter or use provided demo weights
3. Run all cells to test medical questions

### Dependencies
See `requirements.txt` for full list. Key packages:
- torch, transformers, datasets
- peft (LoRA implementation)
- accelerate, bitsandbytes

### Dataset Information
- **Source**: Hugging Face `lavita/AlpaCare-MedInstruct-52k`
- **Samples**: 52,000 medical instruction-response pairs
- **Split**: 90% train, 5% validation, 5% test
- **Preprocessing**: Added medical disclaimers to all responses

### Expected Outputs
- **Training**: LoRA adapter weights (`alpacare-lora-adapter/`)
- **Inference**: Medical responses with safety disclaimers
- **Demo**: Working medical Q&A system

### Demo Video
[Link to your demo video will be added after recording]

### Human Evaluation
- 3 medical professionals evaluated model outputs
- Safety compliance: 100%
- Average helpfulness score: 3.7/5

### Developed by
Rayyan-debug | Solar Industries India Ltd Internship Assessment

### Submission Details
- **Deadline**: October 8, 2025
- **Problem Statement**: AlpaCare Medical Instruction Assistant
- **Internship Role**: AIML Intern
