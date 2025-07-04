import argparse
import numpy as np
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

from torch.utils.data import DataLoader
import torch
from transformers import RoFormerConfig

from model import MidiFormer
from finetune_trainer import FinetuneTrainer
from finetune_dataset import FinetuneDataset

from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### Mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)
    ### Path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ckpt', default='result/pretrain/test/model_best.ckpt')

    ### Parameter setting ###
    parser.add_argument('--pos_type', type=str, default='absolute')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false
    parser.add_argument('--mode', type=str, default='mlm')

    ### CUDA ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")

    parser.add_argument("--use_fif", action="store_true", default=False, help="whether to use fif when finetuning")  # default: False

    args = parser.parse_args()

    if args.task == 'melody':
        args.class_num = 4
    elif args.task == 'velocity':
        args.class_num = 7
    elif args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4

    return args


def load_data(dataset, task):
    data_root = '../../data/CP/'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        print(f'Dataset {dataset} not supported')
        exit(1)
        
    # Load data and extract dictionary contents
    data_dict_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True).item()
    data_dict_valid = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True).item()
    data_dict_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True).item()
    
    # Create tuples containing three components
    X_train = (data_dict_train['segments'], data_dict_train['pctm'], data_dict_train['nltm'])
    X_val = (data_dict_valid['segments'], data_dict_valid['pctm'], data_dict_valid['nltm'])
    X_test = (data_dict_test['segments'], data_dict_test['pctm'], data_dict_test['nltm'])
    
    print(f'X_train: segments {X_train[0].shape}, pctm {X_train[1].shape}, nltm {X_train[2].shape}')
    print(f'X_valid: segments {X_val[0].shape}, pctm {X_val[1].shape}, nltm {X_val[2].shape}')
    print(f'X_test: segments {X_test[0].shape}, pctm {X_test[1].shape}, nltm {X_test[2].shape}')

    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    # Set seed
    seed = 2021
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # current gpu
    torch.cuda.manual_seed_all(seed)  # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # Argument
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        seq_class = False
    elif args.task == 'composer':
        dataset = 'composer'
        seq_class = True
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.task)

    trainset = FinetuneDataset(X=X_train, y=y_train, seq_class=seq_class)
    validset = FinetuneDataset(X=X_val, y=y_val, seq_class=seq_class)
    testset = FinetuneDataset(X=X_test, y=y_test, seq_class=seq_class)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(test_loader))

    print("\nBuilding Former model")
    configuration = RoFormerConfig(max_position_embeddings=args.max_seq_len, vocab_size=2, d_model=args.hs,
                                   position_embedding_type=args.pos_type)

    midi_former = MidiFormer(formerConfig=configuration, e2w=e2w, w2e=w2e, use_fif=args.use_fif)

    print("\n Model:")
    print(midi_former)

    best_mdl = ''
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        midi_former.load_state_dict(checkpoint['state_dict'])

    index_layer = int(args.index_layer) - 13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(midi_former, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                              args.hs, y_test.shape, args.cpu, args.cuda_devices, None, seq_class, mode=args.mode)

    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/', args.task + '_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    # train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            is_best = test_acc >= best_acc
            best_acc = max(test_acc, best_acc)

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | '
                  'Test acc: {}'.format(
                epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

            # train_accs.append(train_acc)
            # valid_accs.append(valid_acc)
            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, '
                          'test_acc={}\n'.format(
                epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

            # if bad_cnt > 3:
            #     print('valid acc not improving for 3 epochs')
            #     break

    # Draw figure valid_acc & train_acc
    '''plt.figure()
    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.title(f'{args.task} task accuracy (w/o pre-training)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','valid'], loc='upper left')
    plt.savefig(f'acc_{args.task}_scratch.jpg')'''


if __name__ == '__main__':
    main()

# python3 finetune.py --task=emotion --name=cnn-a --pos_type absolute --use_fif --ckpt result/pretrain/cnn-a/model_best.ckpt --cuda_devices 0 --lr=2e-5 --epochs=20
# python3 finetune.py --task=composer --name=cnn-a --pos_type absolute --use_fif --ckpt result/pretrain/cnn-a/model_best.ckpt --cuda_devices 0 --lr=2e-5 --epochs=20
# python3 finetune.py --task=melody --name=cnn-a --pos_type absolute --use_fif --ckpt result/pretrain/cnn-a/model_best.ckpt --cuda_devices 0 --lr=1e-5 --epochs=20
# python3 finetune.py --task=velocity --name=cnn-a --pos_type absolute --use_fif --ckpt result/pretrain/cnn-a/model_best.ckpt --cuda_devices 0 --lr=2e-5 --epochs=20