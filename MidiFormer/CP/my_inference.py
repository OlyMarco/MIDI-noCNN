import argparse
import numpy as np
import random
import pickle
import os
import torch
from torch.utils.data import DataLoader
from transformers import RoFormerConfig

from model import MidiFormer
from modelLM import MidiFormerLM
from midi_dataset import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='MLM/CLM Inference Script')
    
    # Basic parameters
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--datasets', type=str, nargs='+', default=['pop909'])
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--model_path', type=str, default='result/pretrain/cnn-a-LM/model_best.ckpt')
    
    # Task selection
    parser.add_argument('--task', type=str, choices=['mlm', 'clm'], default='mlm',
                       help='Choose task: mlm (masked language model) or clm (causal language model)')
    
    # CLM specific parameters
    parser.add_argument('--clm_mode', type=str, choices=['dataset', 'custom'], default='dataset',
                       help='CLM mode: dataset (use sample from dataset) or custom (custom input)')
    parser.add_argument('--generate_length', type=int, default=50,
                       help='Number of tokens to generate for CLM')
    parser.add_argument('--custom_input', type=str, default=None,
                       help='Custom input for CLM generation (format: "Bar:New,Position:1/16,Pitch:60,Duration:4")')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling in CLM generation')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Top-k sampling (0 means disabled)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling')
    
    return parser.parse_args()


def load_data_single_sample(datasets):
    """Load single sample for inference"""
    root = '../../data/CP'
    
    # Load only the first sample from the first dataset
    dataset = datasets[0]
    
    if dataset in {'pop909', 'composer', 'emopia'}:
        data_dict_train = np.load(os.path.join(root, f'{dataset}_train.npy'), allow_pickle=True).item()
        segments = data_dict_train['segments'][:1]  # Take only the first sample
        pctm = data_dict_train['pctm'][:1]
        nltm = data_dict_train['nltm'][:1]
    elif dataset == 'pop1k7' or dataset == 'ASAP':
        data_dict = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True).item()
        segments = data_dict['segments'][:1]  # Take only the first sample
        pctm = data_dict['pctm'][:1]
        nltm = data_dict['nltm'][:1]
    
    print(f'Loaded single sample from {dataset}: segments {segments.shape}, pctm {pctm.shape}, nltm {nltm.shape}')
    
    return (segments, pctm, nltm)


def parse_custom_input(custom_input_str, e2w):
    """Parse custom input string to token indices"""
    if not custom_input_str:
        return None
    
    try:
        # Parse format: "Bar:New,Position:1/16,Pitch:60,Duration:4"
        parts = custom_input_str.split(',')
        token_indices = []
        
        token_types = ['Bar', 'Position', 'Pitch', 'Duration']
        for i, token_type in enumerate(token_types):
            found = False
            for part in parts:
                if part.strip().startswith(f'{token_type}:'):
                    value = part.split(':')[1].strip()
                    
                    if token_type == 'Pitch' and value.isdigit():
                        # For Pitch, if input is a number, convert to "Pitch XX" format
                        pitch_key = f'Pitch {value}'
                    elif token_type == 'Duration' and value.isdigit():
                        # For Duration, if input is a number, convert to "Duration XX" format
                        pitch_key = f'Duration {value}'
                    elif token_type == 'Position' and ('/' in value or value.isdigit()):
                        # For Position, handle fraction format or numbers
                        if '/' not in value:
                            value = f'{value}/16'
                        pitch_key = f'Position {value}'
                    elif token_type == 'Bar':
                        # For Bar, add "Bar " prefix
                        pitch_key = f'Bar {value}'
                    else:
                        pitch_key = value
                    
                    if pitch_key in e2w[token_type]:
                        token_indices.append(e2w[token_type][pitch_key])
                        found = True
                        break
            
            if not found:
                # If this type is not found, use default values
                if token_type == 'Bar':
                    token_indices.append(e2w[token_type]['Bar New'])
                elif token_type == 'Position':
                    token_indices.append(e2w[token_type]['Position 1/16'])
                elif token_type == 'Pitch':
                    token_indices.append(e2w[token_type]['Pitch 60'])  # C4
                elif token_type == 'Duration':
                    token_indices.append(e2w[token_type]['Duration 4'])
        
        return np.array(token_indices, dtype=np.int64)
        
    except Exception as e:
        print(f"Error parsing custom input: {e}")
        print("Using default input: C4 note")
        return np.array([
            e2w['Bar']['Bar New'],
            e2w['Position']['Position 1/16'], 
            e2w['Pitch']['Pitch 60'],  # C4
            e2w['Duration']['Duration 4']
        ], dtype=np.int64)


def get_mask_indices(max_seq_len, mask_percent):
    """Generate mask indices"""
    Lseq = [i for i in range(max_seq_len)]
    mask_ind = random.sample(Lseq, round(max_seq_len * mask_percent))
    mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
    left = list(set(mask_ind)-set(mask80))
    rand10 = random.sample(left, round(len(mask_ind)*0.1))
    cur10 = list(set(left)-set(rand10))
    return mask80, rand10, cur10


def apply_masking(input_ids, mask80, rand10, cur10, midi_former, device):
    """Apply masking to input sequence"""
    masked_input = input_ids.clone()
    loss_mask = torch.zeros(1, input_ids.shape[1])  # batch_size=1
    
    # Apply 80% real masking
    for i in mask80:
        mask_word = torch.tensor(midi_former.mask_word_np).to(device)
        masked_input[0][i] = mask_word
        loss_mask[0][i] = 1
    
    # Apply 10% random words
    for i in rand10:
        rand_word = torch.tensor(midi_former.get_rand_tok()).to(device)
        masked_input[0][i] = rand_word
        loss_mask[0][i] = 1
    
    # Apply 10% keep current words
    for i in cur10:
        loss_mask[0][i] = 1
    
    return masked_input, loss_mask.to(device)


def token_to_readable(token_seq, w2e):
    """Convert token sequence to readable format"""
    readable = []
    token_types = ['Bar', 'Position', 'Pitch', 'Duration']  # Define token types in order
    
    for i, token in enumerate(token_seq):
        token_type = token_types[i]
        if token_type in w2e and token in w2e[token_type]:
            readable.append(f"{token_type}:{w2e[token_type][token]}")
        else:
            readable.append(f"{token_type}:UNK({token})")
    return readable


def compare_sequences(original, masked, predicted, mask_positions, w2e):
    """Compare original, masked and predicted sequences"""
    print("\n" + "="*100)
    print("Sequence Comparison (showing only masked positions)")
    print("="*100)
    
    correct_predictions = 0
    total_predictions = len(mask_positions)
    
    for pos in mask_positions:
        orig_readable = token_to_readable(original[pos], w2e)
        masked_readable = token_to_readable(masked[pos], w2e)
        pred_readable = token_to_readable(predicted[pos], w2e)
        
        # Check prediction accuracy for each token type
        position_correct = 0
        for i in range(4):  # 4 token types
            if original[pos][i] == predicted[pos][i]:
                position_correct += 1
        
        accuracy = position_correct / 4
        if accuracy == 1.0:
            correct_predictions += 1
            
        print(f"\nPosition {pos:3d} (Accuracy: {accuracy:.2f}):")
        print(f"  Original:  {orig_readable}")
        print(f"  Masked:    {masked_readable}")
        print(f"  Predicted: {pred_readable}")
        
        # Mark prediction results for each token
        status = []
        for i in range(4):
            if original[pos][i] == predicted[pos][i]:
                status.append("✓")
            else:
                status.append("✗")
        print(f"  Status:    {status}")
    
    overall_accuracy = correct_predictions / total_predictions
    print(f"\nOverall Accuracy: {correct_predictions}/{total_predictions} = {overall_accuracy:.4f}")
    return overall_accuracy


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter logits for top-k and top-p sampling"""
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate_sequence(model, input_ids, midi_former, device, generate_length, temperature=1.0, top_k=0, top_p=0.9, w2e=None):
    """Generate sequence using CLM"""
    model.eval()
    
    # Create initial input
    current_sequence = input_ids.clone()  # (1, current_len, 4)
    
    print("\n" + "="*80)
    print("CLM Sequence Generation Process")
    print("="*80)
    
    # Display initial input
    if current_sequence.shape[1] > 0:
        print(f"\nInitial Input (Length: {current_sequence.shape[1]}):")
        for i in range(current_sequence.shape[1]):
            readable = token_to_readable(current_sequence[0, i].cpu().numpy(), w2e)
            print(f"  Position {i:2d}: {readable}")
    
    generated_tokens = []
    
    for step in range(generate_length):
        # Create attention mask
        attn_mask = (current_sequence[:, :, 0] != midi_former.bar_pad_word).float().to(device)
        
        with torch.no_grad():
            # Forward pass
            predictions = model.forward(x=current_sequence, attn=attn_mask, mode="clm")
            
            # Get predictions for the last position
            next_token_logits = []
            for i, etype in enumerate(midi_former.e2w):
                logits = predictions[i][0, -1, :]  # Logits of the last position
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                
                # Sampling
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_token_logits.append(next_token)
            
            # Combine next token
            next_token = torch.stack(next_token_logits, dim=-1)  # (1, 4)
            generated_tokens.append(next_token.cpu().numpy()[0])
            
            # Update current sequence
            current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
            
            # Display generated token
            readable = token_to_readable(next_token.cpu().numpy()[0], w2e)
            print(f"  Step {step+1:2d}: {readable}")
    
    return current_sequence, generated_tokens


def run_mlm_inference(model, data_batch, midi_former, device, mask_percent, max_seq_len, w2e):
    """Run MLM inference"""
    print("\n" + "="*60)
    print("Executing MLM (Masked Language Model) Inference")
    print("="*60)
    
    segments = data_batch['segments'].to(device)  # (1, seq_len, 4)
    pctm_batch = data_batch['pctm'].to(device)   # (1, 12, 12)
    nltm_batch = data_batch['nltm'].to(device)   # (1, 12, 12)
    
    original_seq = segments.clone()
    
    print(f"Original sequence shape: {original_seq.shape}")
    
    # Generate masks
    mask80, rand10, cur10 = get_mask_indices(max_seq_len, mask_percent)
    all_mask_positions = mask80 + rand10 + cur10
    
    print(f"Number of masked positions: {len(all_mask_positions)} ({mask_percent*100:.1f}%)")
    print(f"  - Real masks (80%): {len(mask80)}")
    print(f"  - Random replacement (10%): {len(rand10)}")
    print(f"  - Keep original (10%): {len(cur10)}")
    
    # Apply masking
    masked_input, loss_mask = apply_masking(original_seq, mask80, rand10, cur10, midi_former, device)
    
    # Create attention mask
    attn_mask = (masked_input[:, :, 0] != midi_former.bar_pad_word).float().to(device)
    
    # Inference
    with torch.no_grad():
        print("\nExecuting MLM inference...")
        predictions = model.forward(x=masked_input, attn=attn_mask, mode="mlm", 
                                  pctm=pctm_batch, nltm=nltm_batch)
        
        # Get prediction results
        predicted_tokens = []
        for i, etype in enumerate(midi_former.e2w):
            pred = torch.argmax(predictions[i], dim=-1)  # (batch, seq_len)
            predicted_tokens.append(pred)
        
        # Convert to numpy and reorganize
        predicted_seq = torch.stack(predicted_tokens, dim=-1)  # (batch, seq_len, 4)
    
    # Convert to CPU numpy for comparison
    original_np = original_seq[0].cpu().numpy()  # (seq_len, 4)
    masked_np = masked_input[0].cpu().numpy()    # (seq_len, 4)
    predicted_np = predicted_seq[0].cpu().numpy()  # (seq_len, 4)
    
    # Compare results
    accuracy = compare_sequences(original_np, masked_np, predicted_np, 
                               all_mask_positions, w2e)
    
    # Calculate accuracy for each token type
    print("\n" + "="*50)
    print("Token Type Accuracy Statistics:")
    print("="*50)
    
    token_types = ['Bar', 'Position', 'Pitch', 'Duration']
    for i, token_type in enumerate(token_types):
        correct = 0
        total = 0
        for pos in all_mask_positions:
            total += 1
            if original_np[pos][i] == predicted_np[pos][i]:
                correct += 1
        
        token_accuracy = correct / total if total > 0 else 0
        print(f"{token_type:15s}: {correct:3d}/{total:3d} = {token_accuracy:.4f}")
    
    print(f"\nOverall Position Accuracy: {accuracy:.4f}")


def run_clm_inference(model, midi_former, device, args, w2e, e2w, data_batch=None):
    """Run CLM inference"""
    print("\n" + "="*60)
    print("Executing CLM (Causal Language Model) Inference")
    print("="*60)
    
    if args.clm_mode == 'dataset':
        # Use dataset sample
        segments = data_batch['segments'].to(device)  # (1, seq_len, 4)
        
        # Use only the first 10 tokens as prompt
        prompt_length = min(10, segments.shape[1])
        input_sequence = segments[:, :prompt_length, :]
        
        print(f"Using dataset sample as prompt (Length: {prompt_length})")
        
    else:
        # Use custom input
        custom_tokens = parse_custom_input(args.custom_input, e2w)
        if custom_tokens is not None:
            input_sequence = torch.tensor(custom_tokens).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 4)
        else:
            # Default to C4 note
            default_tokens = np.array([
                e2w['Bar']['Bar New'],
                e2w['Position']['Position 1/16'],
                e2w['Pitch']['Pitch 60'],  # C4
                e2w['Duration']['Duration 4']
            ])
            input_sequence = torch.tensor(default_tokens).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 4)
        
        print(f"Using custom input as prompt")
    
    # Generate sequence
    generated_sequence, generated_tokens = generate_sequence(
        model, input_sequence, midi_former, device, args.generate_length,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, w2e=w2e
    )
    
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"Total length: {generated_sequence.shape[1]} (prompt: {input_sequence.shape[1]}, generated: {len(generated_tokens)})")
    
    return generated_sequence, generated_tokens


def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.task.upper()}")
    
    # Load dictionary
    print("\nLoading dictionary...")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    print("Dictionary size:", {k: len(v) for k, v in e2w.items()})
    
    # Build model
    print("\nBuilding model...")
    configuration = RoFormerConfig(
        max_position_embeddings=args.max_seq_len, 
        vocab_size=2, 
        d_model=768,
        position_embedding_type='absolute'
    )
    
    midi_former = MidiFormer(formerConfig=configuration, e2w=e2w, w2e=w2e, use_fif=True)
    model = MidiFormerLM(midi_former).to(device)
    
    # Load pretrained weights
    print(f"\nLoading pretrained model: {args.model_path}")
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrapped model state dict
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully!")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Trying non-strict loading...")
            model.load_state_dict(state_dict, strict=False)
            print("Non-strict loading completed!")
    else:
        print(f"Warning: Model file does not exist {args.model_path}")
        print("Using randomly initialized model for demonstration...")
    
    # Set to evaluation mode
    model.eval()
    
    # Execute different inference based on task type
    if args.task == 'mlm':
        # MLM task requires loading dataset sample
        print("\nLoading test data...")
        single_sample = load_data_single_sample(args.datasets)
        test_dataset = MidiDataset(X=single_sample)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(test_loader))
        
        run_mlm_inference(model, data_batch, midi_former, device, args.mask_percent, args.max_seq_len, w2e)
        
    else:  # CLM task
        data_batch = None
        if args.clm_mode == 'dataset':
            print("\nLoading test data...")
            single_sample = load_data_single_sample(args.datasets)
            test_dataset = MidiDataset(X=single_sample)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            data_batch = next(iter(test_loader))
        
        run_clm_inference(model, midi_former, device, args, w2e, e2w, data_batch)


if __name__ == '__main__':
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()


"""
MLM Inference:
# Basic MLM inference
python my_inference.py --task mlm

# MLM inference with custom parameters
python my_inference.py --task mlm --mask_percent 0.2 --model_path your_model.ckpt


CLM Inference (Dataset Sample):
# Generate using dataset sample as prompt
python my_inference.py --task clm --clm_mode dataset --generate_length 30

# Custom generation parameters
python my_inference.py --task clm --clm_mode dataset --generate_length 50 --temperature 0.8 --top_p 0.9


CLM Inference (Custom Input):
# Generate starting with C4 note
python my_inference.py --task clm --clm_mode custom --custom_input "Bar:New,Position:1/16,Pitch:60,Duration:4" --generate_length 20

# Use different notes and parameters
python my_inference.py --task clm --clm_mode custom --custom_input "Bar:New,Position:1/16,Pitch:72,Duration:8" --generate_length 30 --temperature 1.2 --top_k 50
"""