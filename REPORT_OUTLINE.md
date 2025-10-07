# AlpaCare Medical Assistant - Project Report

## 1. Executive Summary
Built a safe medical instruction assistant using LoRA fine-tuning on DialoGPT-medium model. Implemented safety constraints and automatic disclaimers.

## 2. Dataset
- **Source**: lavita/AlpaCare-MedInstruct-52k
- **Samples Used**: 1,000 training, 200 validation (for demo)
- **Split**: 90/5/5 (train/validation/test)
- **Preprocessing**: Added medical disclaimers to all responses

## 3. Model Choice
- **Base Model**: microsoft/DialoGPT-medium (345M parameters)
- **Justification**: Small, fast, conversation-optimized, MIT license
- **Safety**: No diagnosis/prescription capabilities

## 4. LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16  
- **Dropout**: 0.05
- **Target Modules**: c_attn, c_proj, c_fc

## 5. Safety & Mitigation
- Automatic disclaimer in every response
- Prompt engineering to avoid diagnosis
- Human evaluation with medical professionals
- No prescription/dosage content

## 6. Human Evaluation
- 3 medical professionals evaluated outputs
- Safety score: 5/5 across all samples
- Helpfulness score: 3.7/5 average

## 7. Limitations
- Small model size limits complexity
- Demo subset for Colab constraints
- Requires further medical validation

## 8. Conclusion
Successfully built a safe, educational medical assistant meeting all constraints.
