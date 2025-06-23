import argparse
import numpy as np
import random
import pickle
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
from transformers import RoFormerConfig
from model import MidiFormer
from trainer import FormerTrainer
from midi_dataset import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### Path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--name', type=str, default='MidiFormer')

    ### Pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['pop909', 'composer', 'pop1k7', 'ASAP', 'emopia'])

    ### Parameter setting ###
    parser.add_argument('--pos_type', type=str, default='absolute')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### CUDA ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")

    parser.add_argument("--use_mlm", action="store_true", default=False, help="whether to use mlm when training")  # default: False
    parser.add_argument("--use_clm", action="store_true", default=False, help="whether to use clm when training")  # default: False
    parser.add_argument("--use_fif", action="store_true", default=False, help="whether to use fif when training")  # default: False

    args = parser.parse_args()

    return args


def load_data(datasets):
    to_concat_segments = []
    to_concat_pctm = []
    to_concat_nltm = []
    root = '../../data/CP'

    for dataset in datasets:
        if dataset in {'pop909', 'composer', 'emopia'}:
            # Load data
            data_dict_train = np.load(os.path.join(root, f'{dataset}_train.npy'), allow_pickle=True).item()
            data_dict_valid = np.load(os.path.join(root, f'{dataset}_valid.npy'), allow_pickle=True).item()
            data_dict_test = np.load(os.path.join(root, f'{dataset}_test.npy'), allow_pickle=True).item()
            
            # Extract segments and matrices
            segments = np.concatenate((data_dict_train['segments'], data_dict_valid['segments'], data_dict_test['segments']), axis=0)
            pctm = np.concatenate((data_dict_train['pctm'], data_dict_valid['pctm'], data_dict_test['pctm']), axis=0)
            nltm = np.concatenate((data_dict_train['nltm'], data_dict_valid['nltm'], data_dict_test['nltm']), axis=0)
            
        elif dataset == 'pop1k7' or dataset == 'ASAP':
            data_dict = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True).item()
            segments = data_dict['segments']
            pctm = data_dict['pctm']
            nltm = data_dict['nltm']

        print(f'   {dataset}: segments {segments.shape}, pctm {pctm.shape}, nltm {nltm.shape}')
        to_concat_segments.append(segments)
        to_concat_pctm.append(pctm)
        to_concat_nltm.append(nltm)

    # Merge all datasets
    all_segments = np.vstack(to_concat_segments)
    all_pctm = np.vstack(to_concat_pctm)
    all_nltm = np.vstack(to_concat_nltm)
    print('   > all training data:', all_segments.shape)
    
    # Create index array and shuffle to maintain correspondence of three data types
    indices = np.arange(len(all_segments))
    np.random.shuffle(indices)
    
    all_segments = all_segments[indices]
    all_pctm = all_pctm[indices]
    all_nltm = all_nltm[indices]
    
    # Split training and validation sets
    split = int(len(all_segments)*0.85)
    X_train = (all_segments[:split], all_pctm[:split], all_nltm[:split])
    X_val = (all_segments[split:], all_pctm[split:], all_nltm[split:])
    
    return X_train, X_val


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset", args.datasets)
    X_train, X_val = load_data(args.datasets)

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))

    print("\nBuilding Former model")
    configuration = RoFormerConfig(max_position_embeddings=args.max_seq_len, vocab_size=2, d_model=args.hs,
                                   position_embedding_type=args.pos_type)
    # 0: MLM
    # 1: CLM
    midi_former = MidiFormer(formerConfig=configuration, e2w=e2w, w2e=w2e, use_fif=args.use_fif)

    print("\n Model:")
    print(midi_former)

    print("\nCreating Former Trainer")
    trainer = FormerTrainer(midi_former, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len,
                              args.mask_percent, args.cpu, args.use_mlm, args.use_clm, args.cuda_devices)

    print("\nTraining Start")
    save_dir = 'result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))


    from torch.utils.tensorboard import SummaryWriter
    # Initialize TensorBoard SummaryWriter
    tb_log_dir = os.path.join(save_dir, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"   TensorBoard logs will be saved to {tb_log_dir}")

    best_acc, best_epoch = 0, 0
    bad_cnt = 0
    restart_count = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
            
        # Training and validation steps remain unchanged
        if args.use_mlm:
            if args.use_clm:
                train_loss, train_mlm_acc, train_clm_acc = trainer.train()
                valid_loss, valid_mlm_acc, valid_clm_acc = trainer.valid()
                
                # Record MLM and CLM metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)
                
                # Record MLM accuracy for each token type
                for i, acc in enumerate(train_mlm_acc):
                    writer.add_scalar(f'Accuracy/train_mlm_{list(e2w.keys())[i]}', acc, epoch)
                for i, acc in enumerate(valid_mlm_acc):
                    writer.add_scalar(f'Accuracy/valid_mlm_{list(e2w.keys())[i]}', acc, epoch)
                
                # Record CLM accuracy for each token type
                for i, acc in enumerate(train_clm_acc):
                    writer.add_scalar(f'Accuracy/train_clm_{list(e2w.keys())[i]}', acc, epoch)
                for i, acc in enumerate(valid_clm_acc):
                    writer.add_scalar(f'Accuracy/valid_clm_{list(e2w.keys())[i]}', acc, epoch)
            else:
                train_loss, train_mlm_acc = trainer.train()
                valid_loss, valid_mlm_acc = trainer.valid()
                
                # Record MLM metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)
                
                # Record MLM accuracy for each token type
                for i, acc in enumerate(train_mlm_acc):
                    writer.add_scalar(f'Accuracy/train_mlm_{list(e2w.keys())[i]}', acc, epoch)
                for i, acc in enumerate(valid_mlm_acc):
                    writer.add_scalar(f'Accuracy/valid_mlm_{list(e2w.keys())[i]}', acc, epoch)
        else:
            if args.use_clm:
                train_loss, train_clm_acc = trainer.train()
                valid_loss, valid_clm_acc = trainer.valid()
                
                # Record CLM metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)
                
                # Record CLM accuracy for each token type
                for i, acc in enumerate(train_clm_acc):
                    writer.add_scalar(f'Accuracy/train_clm_{list(e2w.keys())[i]}', acc, epoch)
                for i, acc in enumerate(valid_clm_acc):
                    writer.add_scalar(f'Accuracy/valid_clm_{list(e2w.keys())[i]}', acc, epoch)
            else:
                train_loss = trainer.train()
                valid_loss = trainer.valid()
                
                # Only record loss
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)

        # Calculate and record weighted average accuracy
        if args.use_mlm:
            weighted_score = [x * y for (x, y) in zip(valid_mlm_acc, midi_former.n_tokens)]
            avg_acc = sum(weighted_score) / sum(midi_former.n_tokens)
            writer.add_scalar('Accuracy/weighted_avg_mlm', avg_acc, epoch)
        else:
            weighted_score = [x * y for (x, y) in zip(valid_clm_acc, midi_former.n_tokens)]
            avg_acc = sum(weighted_score) / sum(midi_former.n_tokens)
            writer.add_scalar('Accuracy/weighted_avg_clm', avg_acc, epoch)


        # Record best model information
        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)

        if is_best:
            # Save the best model
            torch.save(trainer.model.state_dict(), filename)
            bad_cnt = 0
            best_epoch = epoch
            writer.add_scalar('Best/epoch', best_epoch, epoch)
            writer.add_scalar('Best/accuracy', best_acc, epoch)
        else:
            bad_cnt += 1
            writer.add_scalar('Training/bad_count', bad_cnt, epoch)

        # Print logs and save model code remains unchanged
        if args.use_mlm:
            if args.use_clm:        
                print('epoch: {}/{} | Train Loss: {} | Train acc: {}, {} | Valid Loss: {} | Valid acc: {}, {}'.format(
                    epoch + 1, args.epochs, train_loss, train_mlm_acc, train_clm_acc, valid_loss, valid_mlm_acc, valid_clm_acc))
            else:
                print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
                    epoch + 1, args.epochs, train_loss, train_mlm_acc, valid_loss, valid_mlm_acc))
        else:
            if args.use_clm:        
                print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
                    epoch + 1, args.epochs, train_loss, train_clm_acc, valid_loss, valid_clm_acc))
            else:
                print('epoch: {}/{} | Train Loss: {} | Valid Loss: {}'.format(
                    epoch + 1, args.epochs, train_loss, valid_loss))

        if args.use_mlm:
            trainer.save_checkpoint(epoch, best_acc, valid_mlm_acc,
                                    valid_loss, train_loss, is_best, filename)
        else:
            trainer.save_checkpoint(epoch, best_acc, valid_clm_acc,
                                    valid_loss, train_loss, is_best, filename)

        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            if args.use_mlm:
                if args.use_clm:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, {}, valid_loss={}, valid_acc={},{}\n'.format(
                        epoch + 1, train_loss, train_mlm_acc, train_clm_acc, valid_loss, valid_mlm_acc, valid_clm_acc))
                else:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                        epoch + 1, train_loss, train_mlm_acc, valid_loss, valid_mlm_acc))
            else:
                if args.use_clm:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                        epoch + 1, train_loss, train_clm_acc, valid_loss, valid_clm_acc))
                else:
                    outfile.write('Epoch {}: train_loss={}, valid_loss={}\n'.format(
                        epoch + 1, train_loss, valid_loss))         
            outfile.write(f'bad_cnt: {bad_cnt}\n')

    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining completed. TensorBoard logs saved to {tb_log_dir}")
    print(f"Run 'tensorboard --logdir={tb_log_dir}' to view training metrics")     


if __name__ == '__main__':
    main()

# python3 main.py --name=cnn-a-aaa --pos_type absolute --batch_size 12 --use_clm --use_mlm --use_fif --cuda_devices 0