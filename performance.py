# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie (modified)
@File: performance.py
@Time: 2021/4/1 11:48
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
import matplotlib.pyplot as plt

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

def performance(args, SNR, net):
    bleu_scores_1 = BleuScore(1, 0, 0, 0)
    bleu_scores_2 = BleuScore(0.5, 0.5, 0, 0)
    bleu_scores_3 = BleuScore(1/3, 1/3, 1/3, 0)
    bleu_scores_4 = BleuScore(0.25, 0.25, 0.25, 0.25)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    bleu_scores_all = {1: [], 2: [], 3: [], 4: []}

    net.eval()
    with torch.no_grad():
        for snr in tqdm(SNR, desc="Evaluating SNR"):
            noise_std = SNR_to_noise(snr)
            all_preds = []
            all_targets = []

            for sents in test_iterator:
                sents = sents.to(device)
                target = sents

                out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                     start_idx, end_idx, args.channel)
                pred = out.cpu().numpy().tolist()
                tgt = target.cpu().numpy().tolist()

                all_preds += list(map(StoT.sequence_to_text, pred))
                all_targets += list(map(StoT.sequence_to_text, tgt))

            bleu_scores_all[1].append(bleu_scores_1.compute_blue_score(all_preds, all_targets))
            bleu_scores_all[2].append(bleu_scores_2.compute_blue_score(all_preds, all_targets))
            bleu_scores_all[3].append(bleu_scores_3.compute_blue_score(all_preds, all_targets))
            bleu_scores_all[4].append(bleu_scores_4.compute_blue_score(all_preds, all_targets))

    return bleu_scores_all


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    args.vocab_file = os.path.join('data', args.vocab_file)
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if fn.endswith('.pth'):
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print(f"Model loaded from {model_path}")

    bleu_scores_all = performance(args, SNR, deepsc)

    # print("\nBLEU Scores vs SNR:")
    # for n in range(1, 5):
    #     # print(f"BLEU-{n}:", [round(s, 4) for s in bleu_scores_all[n]])
    #     print(f"BLEU-{n}:", [round(np.mean(s), 4) for s in bleu_scores_all[n]])
    

    #BLEU-N Plotting
    print("\nBLEU Scores vs SNR:")
    for n in range(1, 5):
        scores_per_snr = [np.mean(s) for s in bleu_scores_all[n]]
        print(f"BLEU-{n}:", [round(score, 4) for score in scores_per_snr])
        plt.plot(SNR, scores_per_snr, label=f"BLEU-{n}")

    plt.title("BLEU-N Scores vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU Score")
    plt.xticks(SNR)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bleu_vs_snr.png")
    plt.show()