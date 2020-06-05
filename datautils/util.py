import os
import json
import numpy as np
import torch
from .instance import Instance
from .vocab import Vocab
import re
# 文本蕴含: premise -> hypothesis
# label: neutral  contradiction  entailment
# dataset: https://nlp.stanford.edu/projects/snli/


def eng_split(s):
    def insert_space(punct):
        return ' ' + punct.group() + ' '
    s = re.sub(r'[.?,!:";]+', insert_space, s)
    return [w for w in s.strip().split(' ') if w != '']


def load_dataset(path):
    assert os.path.exists(path), '文件不存在！'
    insts = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            inst = json.loads(line.strip())
            gold_lbl = inst['gold_label']
            if gold_lbl in ["entailment", "neutral", "contradiction"]:
                s1 = inst['sentence1'].lower()
                s2 = inst['sentence2'].lower()
                insts.append(Instance(eng_split(s1),
                                      eng_split(s2),
                                      gold_lbl))
    return insts


def create_vocab(path):
    assert os.path.exists(path), '文件不存在！'
    word_vocab = Vocab(min_count=3)
    lbl_vocab = Vocab(pad=None, unk=None)

    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            inst = json.loads(line.strip())
            gold_lbl = inst['gold_label']
            if gold_lbl in ["entailment", "neutral", "contradiction"]:
                s1 = inst['sentence1'].lower()
                s2 = inst['sentence2'].lower()
                word_vocab.add(eng_split(s1))
                word_vocab.add(eng_split(s2))
                lbl_vocab.add(gold_lbl)
    return word_vocab, lbl_vocab


def build_pretrain_embedding(embed_path):
    assert os.path.exists(embed_path), 'embedding path does not exist!'
    extwd_vocab = Vocab()
    vec_size = 0
    with open(embed_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if len(tokens) > 10:
                wd = tokens[0]
                if vec_size == 0:
                    vec_size = len(tokens[1:])
                extwd_vocab.add(wd)

    wd_count = len(extwd_vocab)
    unk_idx = extwd_vocab.unk_idx
    embed_weights = np.zeros((wd_count, vec_size), dtype=np.float32)
    with open(embed_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if len(tokens) > 10:
                idx = extwd_vocab.inst2idx(tokens[0])
                vec = np.asarray(tokens[1:], dtype=np.float32)
                embed_weights[idx] = vec
                embed_weights[unk_idx] += vec
    embed_weights[unk_idx] /= wd_count
    embed_weights /= np.std(embed_weights)
    return embed_weights, extwd_vocab


def batch_iter(dataset: list, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = (len(dataset) + batch_size - 1) // batch_size
    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)
        yield batch_data


def batch_variable(batch_data, wd_vocab, extwd_vocab, lbl_vocab):
    sent1_max_len, sent2_max_len = 0, 0
    for inst in batch_data:
        if len(inst.sent1) > sent1_max_len:
            sent1_max_len = len(inst.sent1)
        if len(inst.sent2) > sent2_max_len:
            sent2_max_len = len(inst.sent2)

    batch_size = len(batch_data)
    sent1_idxs = torch.zeros((batch_size, sent1_max_len), dtype=torch.long)
    extsent1_idxs = torch.zeros((batch_size, sent1_max_len), dtype=torch.long)

    sent2_idxs = torch.zeros((batch_size, sent2_max_len), dtype=torch.long)
    extsent2_idxs = torch.zeros((batch_size, sent2_max_len), dtype=torch.long)
    gold_lbl = torch.zeros((batch_size, ), dtype=torch.long)

    for i, inst in enumerate(batch_data):
        sent1_idxs[i, :len(inst.sent1)] = torch.tensor(wd_vocab.inst2idx(inst.sent1))
        sent2_idxs[i, :len(inst.sent2)] = torch.tensor(wd_vocab.inst2idx(inst.sent2))

        extsent1_idxs[i, :len(inst.sent1)] = torch.tensor(extwd_vocab.inst2idx(inst.sent1))
        extsent2_idxs[i, :len(inst.sent2)] = torch.tensor(extwd_vocab.inst2idx(inst.sent2))

        gold_lbl[i] = torch.tensor(lbl_vocab.inst2idx(inst.label))

    return sent1_idxs, sent2_idxs, extsent1_idxs, extsent2_idxs, gold_lbl
