# !usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Huiqiang Xie, updated by Moha Nkossie
@File: performance.py
@Time: 2021/4/1
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def performance(args, SNR, net, token_to_idx):
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, end_idx)

    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    test_eur = EurDataset('test')
    test_loader = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                             pin_memory=True, collate_fn=collate_data)

    net.eval()
    all_bleu_scores = []

    with torch.no_grad():
        for snr in tqdm(SNR, desc="Evaluating SNR"):
            noise_std = SNR_to_noise(snr)

            references = []
            hypotheses = []

            for batch in test_loader:
                batch = batch.to(device)

                decoded_output = greedy_decode(net, batch, noise_std, args.MAX_LENGTH,
                                               pad_idx, start_idx, end_idx, args.channel)

                decoded_texts = list(map(StoT.sequence_to_text, decoded_output.cpu().numpy()))
                target_texts = list(map(StoT.sequence_to_text, batch.cpu().numpy()))

                hypotheses.extend(decoded_texts)
                references.extend(target_texts)

            # BLEU Score Evaluation (1-gram)
            bleu = bleu_score_1gram.compute_blue_score(hypotheses, references)
            all_bleu_scores.append(bleu)

    return np.array(all_bleu_scores)


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    args.vocab_file = os.path.join('data', args.vocab_file)
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)

    model = DeepSC(args.num_layers, num_vocab, num_vocab,
                   num_vocab, num_vocab, args.d_model, args.num_heads,
                   args.dff, 0.1).to(device)

    # Load latest checkpoint
    model_paths = [
        (os.path.join(args.checkpoint_path, fn), int(fn.split('_')[-1].split('.')[0]))
        for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')
    ]
    model_paths.sort(key=lambda x: x[1])
    latest_checkpoint_path, _ = model_paths[-1]

    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {latest_checkpoint_path}")

    bleu_scores = performance(args, SNR, model, token_to_idx)
    print("BLEU-1 Scores per SNR:", bleu_scores)
