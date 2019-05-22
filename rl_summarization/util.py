from collections import defaultdict
import numpy as np
import os
from nltk.tokenize import wordpunct_tokenize


def read_corpus(file_path: str, source: str):
    """
    Чтение текстов либо заголовков (каждый сохранен в отдельном файле)
    :param file_path:
    :param source:
    :return:
    """
    data = []
    if not file_path.endswith('/'):
        file_path += '/'
    for file in os.listdir(file_path):
        sent = []
        with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                sent.extend(wordpunct_tokenize(line))
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        buckets[len(pair[0])].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle:
            np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
