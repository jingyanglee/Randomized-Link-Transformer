
import argparse
import jsonlines
import collections
import numpy as np
from tqdm import tqdm
import nltk
nltk.download('punkt')
from itertools import chain
from math import log, e
from lexical_diversity import lex_div as ld

def eval_distinct_k(candidates, k):
    """The total number of k-grams divided by the total number of tokens
         over all the candidates.
      """
    kgrams = set()
    total = 0
    if isinstance(candidates, str):
        if len(candidates) >= k:
            for i in range(0, len(candidates) - k + 1):
                kgrams.add(tuple(candidates[i:i + k]))
            total += len(candidates)
    else:
        for cand in candidates:
            if len(cand) < k:
                continue
            for i in range(0, len(cand) - k + 1):
                kgrams.add(tuple(cand[i:i + k]))
            total += len(cand)
    if total == 0:
        return 0
    else:
        return len(kgrams) / total

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                     left_pad_symbol=None, right_pad_symbol=None):
        """
        Returns a padded sequence of items before ngram extraction.
            #>>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            ['<s>', 1, 2, 3, 4, 5, '</s>']
            #>>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
            ['<s>', 1, 2, 3, 4, 5]
            #>>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
            [1, 2, 3, 4, 5, '</s>']
        :param sequence: the source data to be padded
        :type sequence: sequence or iter
        :param n: the degree of the ngrams
        :type n: int
        :param pad_left: whether the ngrams should be left-padded
        :type pad_left: bool
        :param pad_right: whether the ngrams should be right-padded
        :type pad_right: bool
        :param left_pad_symbol: the symbol to use for left padding (default is None)
        :type left_pad_symbol: any
        :param right_pad_symbol: the symbol to use for right padding (default is None)
        :type right_pad_symbol: any
        :rtype: sequence or iter
        """
        sequence = iter(sequence)
        if pad_left:
            sequence = chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        from nltk.util import ngrams
        list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
       list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        try:
            history.append(next(sequence))
        except StopIteration:
            return
        #history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def eval_distinct_k_v2(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    #sentence = nltk.word_tokenize(sentence)
    sentence= ld.flemmatize(sentence)
    for sen in sentence:
        if sen == '' or sen == 's':
            sentence.remove(sen)
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

def eval_entropy_k(candidates, k):
    """Entropy method which takes into account word frequency."""
    kgram_counter = collections.Counter()
    for cand in candidates:
        for i in range(0, len(cand) - k + 1):
            kgram_counter.update([tuple(cand[i:i + k])])

    counts = kgram_counter.values()
    s = sum(counts)
    if s == 0:
        # all of the candidates are shorter than k
        return 0
    return (-1.0 / s) * sum(f * np.log(f / s) for f in counts)

def entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    labels = nltk.word_tokenize(labels)
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

def eval_len(candidates):
    lengths = []
    for cand in candidates:
        lengths.append(len(cand.split(" ")))
    ave = sum(lengths) / len(lengths)
    return ave

def mtld(res):
    flt = ld.flemmatize(res)
    for sen in flt:
        if sen == '' or sen == 's':
            flt.remove(sen)
    mtld = ld.mtld(flt)
    return mtld

def hdd(res):
    flt = ld.flemmatize(res)
    for sen in flt:
        if sen == '' or sen == 's':
            flt.remove(sen)
    hdd = ld.hdd(flt)
    return hdd

def ttr(res):
    flt = ld.flemmatize(res)
    for sen in flt:
        if sen == '' or sen == 's':
            flt.remove(sen)
    ttr = ld.ttr(flt)
    return ttr

def mattr(res):
    flt = ld.flemmatize(res)
    for sen in flt:
        if sen == '' or sen == 's':
            flt.remove(sen)
    mattr = ld.mattr(flt)
    return mattr


def main(args):
    lengths = []
    corpus = ''
    with jsonlines.open('generated.jsonl') as reader:
        for idx, row in enumerate(tqdm(reader)):
            lengths.append(len(row["res"].split()))
            corpus = corpus + ' ' + row['res']

    average_distinct_1 = eval_distinct_k_v2(corpus, 1)
    average_distinct_2 = eval_distinct_k_v2(corpus, 2)
    average_distinct_3 = eval_distinct_k_v2(corpus, 3)
    average_hdd = hdd(corpus)
    average_mtld = mtld(corpus)
    average_mattr = mattr(corpus)
    average_ttr = ttr(corpus)

    print(f"Corpus distinct 1: {average_distinct_1}")
    print(f"Corpus distinct 2: {average_distinct_2}")
    print(f"Corpus distinct 3: {average_distinct_3}")
    print(f"Corpus TTR: {average_ttr}")
    print(f"Corpus MATTR: {average_mattr}")
    print(f"Corpus MTLD: {average_mtld}")
    print(f"Corpus HDD: {average_hdd}")


def cli_main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
