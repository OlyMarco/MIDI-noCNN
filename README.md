# MIDI-noCNN: MidiFormer without CNN Components

A simplified version of MidiFormer for MIDI music processing and generation tasks, with CNN components removed for streamlined training and inference.

## Overview

This project provides a MidiFormer implementation without CNN components, supporting multiple tasks including:
- **Task 1**: Masked Language Modeling (MLM) and Causal Language Modeling (CLM)
- **Task 2**: Emotion and Style Classification (sequence-level tasks)
- **Task 3**: Four downstream tasks (melody, velocity, composer, emotion)

## Project Structure

```
MIDI-noCNN/
├── data/CP/                    # Training data
│   ├── emopia_train.npy       # Training data example
│   ├── emopia_train_ans.npy   # Training labels example
│   └── tmp/                   # Temporary data folders
│       ├── melody/            # Melody task data
│       └── velocity/          # Velocity task data
├── MidiFormer/CP/             # Core model implementation
│   ├── model.py              # MidiFormer model
│   ├── finetune_model.py     # Fine-tuning models
│   ├── trainer.py            # Training logic
│   └── result/               # Results and models
│       └── pretrain/         # Pre-trained models
└── result/                   # Training logs and outputs
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download pre-trained models from [HuggingFace](https://huggingface.co/TemmiePratt/MIDI-noCNN) to `MidiFormer/CP/result/pretrain/`:
   - `cnn-a`: Best pre-trained base model (for Tasks 2 & 3)
   - `cnn-a-LM`: Model with CLM and MLM heads (for Task 1)

## Data Format

The training data should follow the format of `data/CP/emopia_train.npy` and `data/CP/emopia_train_ans.npy`. 

**Required data structure:**
- `segments`: Main sequence data
- `nltm`: 12x12 zero matrix (same batch size as segments)
- `pctm`: 12x12 zero matrix (same batch size as segments)

Note: NLTM and PCTM are included for compatibility but won't participate in training as related code in `MidiFormer/CP/model.py` is commented out.

## Usage

### Task 1: Language Modeling (MLM & CLM)

#### Masked Language Modeling (MLM)
```bash
# Basic MLM inference
python my_inference.py --task mlm

# MLM with custom parameters
python my_inference.py --task mlm --mask_percent 0.2 --model_path your_model.ckpt
```

#### Causal Language Modeling (CLM)
```bash
# Generate using dataset sample as prompt
python my_inference.py --task clm --clm_mode dataset --generate_length 30

# Custom generation parameters
python my_inference.py --task clm --clm_mode dataset --generate_length 50 --temperature 0.8 --top_p 0.9

# Generate starting with custom input
python my_inference.py --task clm --clm_mode custom --custom_input "Bar:New,Position:1/16,Pitch:60,Duration:4" --generate_length 20
```

### Task 2: Emotion and Style Classification

1. **Data Preparation**: Format your emotion/style dataset following the structure of `data/CP/emopia_train.npy` and `data/CP/emopia_train_ans.npy`

2. **Modify Data Loading**: Update `load_data(dataset, task)` function and dataset module logic

3. **Fine-tuning**: Use the `SequenceClassification` class from `MidiFormer/CP/finetune_model.py`

4. **Class Configuration**: Set the number of classes according to your dataset:
```python
if args.task == 'emotion':
    args.class_num = 4  # Adjust based on your emotion classes
elif args.task == 'style':
    args.class_num = X  # Adjust based on your style classes
```

### Task 3: Four Downstream Tasks

To fine-tune on melody, velocity, composer, and emotion tasks:

1. **Enable NLTM/PCTM**: Uncomment all code related to `# if pctm is not None and nltm is not None and self.training:` in `MidiFormer/CP/model.py`

2. **Set Learning Rates**:
   - Token-level tasks (melody, velocity): `lr = 1e-5`
   - Sequence-level tasks (composer, emotion): `lr = 2e-5`

3. **Class Numbers**:
```python
if args.task == 'melody':
    args.class_num = 4
elif args.task == 'velocity':
    args.class_num = 7
elif args.task == 'composer':
    args.class_num = 8
elif args.task == 'emotion':
    args.class_num = 4
```

4. **Data Management**: For melody and velocity tasks, ensure proper data folder management during fine-tuning (move datasets in/out of `data/CP/tmp/melody/` and `data/CP/tmp/velocity/`)

## Training Configuration

- **Multi-GPU Support**: Modify CUDA device settings for multi-GPU training
- **Monitoring**: Use TensorBoard to monitor training progress:
```bash
cd result
tensorboard --logdir .
```
- **Model Saving**: The script saves the best validation model within 20 epochs

## Notes

- The model retains CNN architecture but doesn't use it during inference or fine-tuning
- NLTM and PCTM logic can be completely removed if needed (this is already a simplified version)
- For melody and velocity tasks, datasets use the same POP dataset but generate different task-specific data
- If Task 2 performance is poor, consider transfer pre-training on emotion/style datasets using the `cnn-a` model

## Model Details

- **Base Model**: MidiFormer without CNN components
- **Pre-trained Models**: Available on HuggingFace
- **Supported Tasks**: MLM, CLM, sequence classification, token classification
- **Data Format**: Custom numpy arrays with segments, NLTM, and PCTM components