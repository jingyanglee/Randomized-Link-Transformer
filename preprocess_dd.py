from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import config
import torch
import re
from io import open
import ast
from torchnlp.word_to_vector import FastText
vectors = FastText()
from torchtext.data.utils import get_tokenizer
en_tokenizer = get_tokenizer('spacy', language='en')
from torch.utils.data import Dataset, DataLoader
import os
import sys

UNK_idx = 0
PAD_idx = 1  # Used for padding short sentences # Start-of-sentence token
EOS_idx = 2  # End-of-sentence token
SOS_idx = 3

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "PAD", 2: "EOS", 3: "SOS"}
        self.n_words = 4  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.vocab = vocab
        self.num_total_seqs = len(data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["input_txt"] = self.data[index]['context']
        item["target_txt"] = self.data[index]['label']
        item["input_batch"], item["input_mask"] = self.preprocess(self.data[index]['context'])
        item["posterior_batch"], item["posterior_mask"] = self.preprocess(self.data[index]['label'], posterior=True)
        item["target_batch"] = self.preprocess(item["target_txt"], anw=True)
        return item

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, arr, anw=False, meta=None, posterior=False):
        """Converts words to ids."""
        if (anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        arr.split(' ')] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            if meta ==None:
                X_dial = [config.CLS1_idx]
                X_mask = [config.CLS1_idx]
            else:
                X_dial = [config.CLS1_idx, meta] if posterior else [config.CLS_idx, meta]
                X_mask = [config.CLS1_idx, meta] if posterior else [config.CLS_idx, meta]
            if (config.model == "seq2seq" or config.model == "cvae"):
                X_dial = []
                X_mask = []
            if type(arr) == list:
                for i, sentence in enumerate(arr):
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                            sentence.split(' ')]
                    spk = config.USR_idx if i % 2 == 0 else config.SYS_idx
                    if posterior: spk = config.SYS_idx
                    X_mask += [spk for _ in range(len(sentence.split(' ')))]
            else:
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           arr.split(' ')]
                spk = config.USR_idx
                if posterior: spk = config.SYS_idx
                X_mask += [spk for _ in range(len(arr.split(' ')))]
            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)

    def process_input(self, input_txt):
        seq = []
        oovs = []
        for word in ' '.join(input_txt).split(' '):
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            else:
                seq.append(config.UNK_idx)
            # else:
            #     if word not in oovs:
            #         oovs.append(word)
            #     seq.append(self.vocab.n_words + oovs.index(word))
        seq.append(config.PAD_idx)
        seq = torch.LongTensor(seq)
        return seq, oovs

    def process_target(self, target_txt, oovs):
        # seq = [self.word2index[word] if word in self.word2index and self.word2index[word] < self.output_vocab_size else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
        seq = []
        for word in target_txt.strip().split():
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            elif word in oovs:
                seq.append(self.vocab.n_words + oovs.index(word))
            else:
                seq.append(UNK_idx)
        seq.append(EOS_idx)
        seq = torch.LongTensor(seq)
        return seq


def loadLines(file):
    with open(file, 'r') as datafile:
        lines =datafile.readlines()
    data =[]
    for line in lines:
        data.append(ast.literal_eval(line))
    return data

def create_data(data, vocab):
    dialog_data = []
    for d in data:
        #print(d)
        context = []
        for i in range(len(d['dialogue'])):
            text = re.sub(r'[^\w\s]','', d['dialogue'][i]['text'])
            vocab.index_words(d['dialogue'][i]['text'])
            if i == 0:
                #context = context + d['dialogue'][i]['text']
                context.append(d['dialogue'][i]['text'])
                continue
            elif i%2 == 0:
                context.append(d['dialogue'][i]['text'])
                continue
            else:
                con = context.copy()
                dialog = {'context':con, 'label':d['dialogue'][i]['text']}
                dialog_data.append(dialog)
                context.append(d['dialogue'][i]['text'])

    return dialog_data
        #context =
        #dialog = {}
        #print(d)

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["input_batch"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    input_batch, input_lengths = merge(item_info['input_batch'])
    posterior_batch, posterior_lengths = merge(item_info['posterior_batch'])
    input_mask, input_mask_lengths = merge(item_info['input_mask'])
    posterior_mask, posterior_mask_lengths = merge(item_info['posterior_mask'])
    target_batch, target_lengths = merge(item_info['target_batch'])

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        posterior_batch = posterior_batch.cuda()
        posterior_mask = posterior_mask.cuda()
        input_mask = input_mask.cuda()
        target_batch = target_batch.cuda()

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["input_mask"] = input_mask
    d["posterior_batch"] = posterior_batch
    d["posterior_lengths"] = torch.LongTensor(posterior_lengths)
    d["posterior_mask"] = posterior_mask
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    if 'input_ext_vocab_batch' in item_info:
        input_ext_vocab_batch, _ = merge(item_info['input_ext_vocab_batch'])
        target_ext_vocab_batch, _ = merge(item_info['target_ext_vocab_batch'])
        #print( input_batch.size())
        input_ext_vocab_batch = input_ext_vocab_batch
        target_ext_vocab_batch = target_ext_vocab_batch
        if config.USE_CUDA:
            input_ext_vocab_batch = input_ext_vocab_batch.cuda()
            target_ext_vocab_batch = target_ext_vocab_batch.cuda()
        d["input_ext_vocab_batch"] = input_ext_vocab_batch
        d["target_ext_vocab_batch"] = target_ext_vocab_batch
        if "article_oovs" in item_info:
            d["article_oovs"] = item_info["article_oovs"]
            d["max_art_oovs"] = max(len(art_oovs) for art_oovs in item_info["article_oovs"])
    d["input_txt"] = item_info['input_txt']
    d["target_txt"] = item_info['target_txt']
    d['program_label'] =  None

    return d

vocab = Lang()
raw_train_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/train.json"))
raw_valid_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/valid.json"))
raw_test_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/test.json"))
train_data = create_data(raw_train_data, vocab)
valid_data = create_data(raw_valid_data, vocab)
test_data = create_data(raw_test_data, vocab)

train_dataset = Dataset(train_data, vocab)

train_dataloader = DataLoader(dataset=train_dataset ,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=collate_fn)
