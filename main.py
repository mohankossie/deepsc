# This script trains a DeepSC model for sequence-to-sequence tasks using Pytorch.
# It includes functionality for training, validation, and BLUE evaluation.
# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi, SeqtoText
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from tqdm import tqdm
from performance import evaluate_bleu

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

# BLEU evaluation flags
parser.add_argument('--evaluate-bleu', action='store_true', help="Run BLEU evaluation instead of training")
parser.add_argument('--bleu-snrs', nargs='+', type=int, default=[0, 5, 10], help="SNRs to test BLEU on")
parser.add_argument('--export-csv', action='store_true', help="Export BLEU scores to CSV")
parser.add_argument('--csv-path', type=str, default="bleu_results.csv", help="CSV path for BLEU results")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, pad_idx, criterion):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    total_loss = 0
    pbar = tqdm(test_iterator, desc=f"[Validation Epoch {epoch+1}]")
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)
            total_loss += loss
            pbar.set_postfix(loss=loss)
    return total_loss / len(test_iterator)

def train(epoch, args, net, pad_idx, optimizer, criterion):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator, desc=f"[Training Epoch {epoch+1}]")
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
    for sents in pbar:
        sents = sents.to(device)
        loss = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel)
        pbar.set_postfix(loss=loss)

if __name__ == '__main__':
    setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = os.path.join('data', args.vocab_file)
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    initNetParams(deepsc)

    # BLEU evaluation mode
    if args.evaluate_bleu:
        # Load latest checkpoint
        checkpoint_dir = args.checkpoint_path
        if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        latest_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        deepsc.load_state_dict(torch.load(checkpoint_path, map_location=device))
        deepsc.eval()

        # Prepare test loader
        test_dataset = EurDataset(split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_data)

        # Run BLEU evaluation
        bleu_results = evaluate_bleu(
            models=[deepsc],
            test_loader=test_loader,
            pad_idx=pad_idx,
            start_idx=token_to_idx["<START>"],
            end_idx=token_to_idx["<END>"],
            snrs=args.bleu_snrs,
            max_len=args.MAX_LENGTH,
            vocab=token_to_idx,
            channel_type=args.channel,
            export_csv=args.export_csv,
            csv_path=args.csv_path
        )
        exit(0)

    # Training loop
    record_loss = float('inf')
    for epoch in range(args.epochs):
        start = time.time()

        train(epoch, args, deepsc, pad_idx, optimizer, criterion)
        avg_loss = validate(epoch, args, deepsc, pad_idx, criterion)

        print(f"Epoch {epoch+1}: Validation Loss = {avg_loss:.4f}, Time = {time.time() - start:.2f}s")

        if avg_loss < record_loss:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            torch.save(deepsc.state_dict(), os.path.join(
                args.checkpoint_path, f'checkpoint_{str(epoch + 1).zfill(2)}.pth'))
            record_loss = avg_loss
