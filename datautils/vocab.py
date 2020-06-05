from collections import Counter
from functools import wraps
import pickle


# 检查词表是否被创建
def _check_build_vocab(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._inst2idx is None:
            self.build_vocab()
        return func(self, *args, **kwargs)
    return _wrapper


class Vocab(object):
    def __init__(self, min_count=None, pad='<pad>', unk='<unk>'):
        self.min_count = min_count
        self._idx2inst = None
        self._inst2idx = None
        self._inst_count = Counter()

        self.PAD = pad
        self.UNK = unk

    def add(self, inst):
        if isinstance(inst, list):
            self._inst_count.update(inst)
        else:
            self._inst_count[inst] += 1

    def build_vocab(self):
        if self._inst2idx is None:
            self._inst2idx = dict()
            if self.PAD is not None:
                self._inst2idx[self.PAD] = len(self._inst2idx)
            if (self.UNK is not None) and (self.PAD != self.UNK):
                self._inst2idx[self.UNK] = len(self._inst2idx)

        min_count = 1 if self.min_count is None else self.min_count
        for inst, count in self._inst_count.items():
            if count >= min_count and inst not in self._inst2idx:
                self._inst2idx[inst] = len(self._inst2idx)

        self._idx2inst = dict((idx, inst) for inst, idx in self._inst2idx.items())
        # del self._inst_count
        return self

    @_check_build_vocab
    def inst2idx(self, inst):
        if isinstance(inst, list):
            return [self._inst2idx.get(i, self.unk_idx) for i in inst]
        else:
            return self._inst2idx.get(inst, self.unk_idx)

    @_check_build_vocab
    def idx2inst(self, idx):
        if isinstance(idx, list):
            return [self._idx2inst.get(i, self.UNK) for i in idx]
        else:
            return self._idx2inst.get(idx, self.UNK)

    @_check_build_vocab
    def save(self, path):
        core_content = {'inst2idx': self._inst2idx,
                        'idx2inst': self._idx2inst,
                        'pad': self.PAD,
                        'unk': self.UNK}

        with open(path, 'wb') as fw:
            pickle.dump(core_content, fw)

    @_check_build_vocab
    def load(self, path):
        with open(path, 'rb') as fin:
            core_content = pickle.load(fin)
        self.PAD = core_content['pad']
        self.UNK = core_content['unk']
        self._inst2idx = core_content['inst2idx']
        self._idx2inst = core_content['idx2inst']

    @property
    @_check_build_vocab
    def pad_idx(self):
        if self.PAD is None:
            return None
        return self._inst2idx[self.PAD]

    @property
    @_check_build_vocab
    def unk_idx(self):
        if self.UNK is None:
            return None
        return self._inst2idx[self.UNK]

    @_check_build_vocab
    def __len__(self):
        return len(self._inst2idx)

    @_check_build_vocab
    def __iter__(self):
        for inst, idx in self._inst2idx.items():
            yield inst, idx

    @_check_build_vocab
    def __contains__(self, item):
        return item in self._inst2idx

