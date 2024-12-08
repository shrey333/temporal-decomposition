# Temporal Event Ordering with LLMs

This repository contains code for temporal event ordering using Large Language Models (LLMs) on the TORQUE and MATRES datasets.

## Project Structure

├── MATRES/
│ ├── data/ # Raw MATRES annotations
│ ├── preprocess/ # Preprocessed MATRES data
│ ├── results/ # Model predictions and evaluations
│ └── StructTempRel-EMNLP17/ # Original MATRES dataset
├── TORQUE/
│ ├── data/ # Raw TORQUE dataset
│ ├── preprocess/ # Preprocessed TORQUE data
│ └── results/ # Model predictions and evaluations
├── log/ # Log files
├── preprocess_matres.py # MATRES preprocessing script
├── preprocess_torque.py # TORQUE preprocessing script
├── finetune.py # Fine-tuning script for LLaMA model
├── gemini_inference_matres.py # Inference script for MATRES using Gemini
└── gemini_inference_torque.py # Inference script for TORQUE using Gemini

## Setup

1. Environment Setup

```bash
conda create -n timeset python=3.9
conda activate timeset
pip install -r requirements.txt
```

2. API Keys

```bash
export GOOGLE_API_KEY="your_gemini_api_key"
```

## Datasets

### MATRES

- A dataset for temporal relation extraction between events
- Relations: "before", "after", "equal", "vague"
- Preprocessing creates train/dev/test splits

### TORQUE

- A dataset for temporal ordering of events
- Question-answering format for temporal relations
- Preprocessed into train/dev/test splits

## Usage

1. Preprocessing

```bash
# Preprocess MATRES dataset
python preprocess_matres.py

# Preprocess TORQUE dataset
python preprocess_torque.py
```

2. Fine-tuning (LLaMA)

```bash
# Change dataset paths for MATRES and TORQUE
python finetune.py
```

3. Inference (Finetuned model)

```bash
# Change dataset paths for MATRES and TORQUE
python inference.py
```

5. Inference (Gemini)

```bash
# MATRES inference
python gemini_inference_matres.py

# TORQUE inference
python gemini_inference_torque.py
```

## Model Details

### LLaMA Fine-tuning

- Model: LLaMA-3-3B
- Training Parameters:
  - Batch size: 8
  - Learning rates: 1e-4, 1e-5, 1e-6
  - LoRA dimensions: 16, 64
  - LoRA alphas: 16, 64
  - LoRA dropout: 0.1
  - Precision: bfloat16
  - Epochs: 10

### Gemini Inference

- Model: gemini-pro
- Inference Parameters:
  - Max samples: 50 (MATRES), 100 (TORQUE)
  - API delay: 15-30 seconds
  - Temperature: 0

## Results

Results are saved in the respective results and output directories:

- `MATRES/results/gemini_results_matres.json`
- `TORQUE/results/gemini_results_torque.json`

## Logging

All scripts use Python's logging module:

- Log files are saved in `./log/`
- Format: timestamp:level - message
- Both console and file logging enabled

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
google-generativeai>=0.3.0
spacy>=3.0.0
beautifulsoup4>=4.9.0
tqdm>=4.65.0
```

## Citation

If you use this code, please cite the original MATRES and TORQUE papers:

```bibtex
@inproceedings{ning2018matres,
    title={A Multi-Axis Annotation Scheme for Event Temporal Relations},
    author={Ning, Qiang and Wu, Hao and Roth, Dan},
    booktitle={ACL},
    year={2018}
}

@inproceedings{ning2020torque,
    title={TORQUE: A Reading Comprehension Dataset of Temporal Ordering Questions},
    author={Ning, Qiang and Wu, Hao and Peng, Haoruo and Roth, Dan},
    booktitle={EMNLP},
    year={2020}
}
```

## Contributors

- Aayushi - Developer
- Ayushi Rajsekhar - Developer
- Dhruv Modi - Developer
- Shrey Bhadiyadara - Developer
