import numpy as np
import sys
import json
import torch
import random
import collections
from torchtext.vocab import Vocab, Vectors


def _load_json(path, labeled=True):
    label = {}
    with open(path, 'r') as f:
        data = []
        for line in f:
            row = json.loads(line)
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            data.append({
                'label': int(row['label']),
                'text': row['text']
                })

        print(label)
        return data


def _read_words(data):
    words = []
    for example in data:
        words += example['text']
    return words


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    # process the text
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    text = np.ones([len(data), max_text_len], dtype=np.int64)\
        * vocab.stoi['<pad>']

    for i in range(len(data)):
        text[i, :len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi
                else vocab.stoi['<unk>'] for x in data[i]['text']]

    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    return {'text': text, 'text_len': text_len, 'label': doc_label, 'raw': raw}


def load_dataset(args):
    print("Loading data...")

    if args.data_dir == '':
        train_data = _load_json('../data/fold/' + args.task + '_fold_' +
                                str(args.fold) + '_train.jsonl')
        dev_data = _load_json('../data/fold/' + args.task + '_fold_' +
                              str(args.fold) + '_dev.jsonl')
        test_data = _load_json('../data/fold/' + args.task + '_fold_' +
                               str(args.fold) + '_test.jsonl')

        vectors = Vectors(args.word_vector,
                          cache='./.vector_cache')
        vocab = Vocab(collections.Counter(
            _read_words(train_data)), vectors=vectors)

    else:
        train_data = _load_json(args.data_dir + '/fold/' + args.task +
                                '_fold_' + str(args.fold) + '_train.jsonl')
        dev_data = _load_json(args.data_dir + '/fold/' + args.task +
                              '_fold_' + str(args.fold) + '_dev.jsonl')
        test_data = _load_json(args.data_dir + '/fold/' + args.task +
                               '_fold_' + str(args.fold) + '_test.jsonl')

        if args.data_size != 0:
            # use a subset of the training data
            pos, neg = [], []
            for example in train_data:
                if example['label'] == 1:
                    pos.append(example)
                else:
                    neg.append(example)

            train_data = []
            random.shuffle(pos)
            random.shuffle(neg)
            pos_ratio = len(pos) / (len(pos) + len(neg))
            for i in range(min(len(pos), int(args.data_size * pos_ratio))):
                train_data.append(pos[i])
            tmp = len(train_data)
            for i in range(min(len(neg), args.data_size - tmp)):
                train_data.append(neg[i])

        vectors = Vectors(args.word_vector,
                          cache=args.data_dir + '/vector_cache')
        vocab = Vocab(collections.Counter(
            _read_words(train_data)), vectors=vectors)

    wv_size = vocab.vectors.size()
    print('Total num. of words: %d\nWord vector dimension: %d' %
          (wv_size[0], wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    print(('Num. of out-of-vocabulary words'
           '(they are initialized to zero vector): %d') % num_oov)

    print('#train: %d, #val: %d, #test: %d'
          % (len(train_data), len(dev_data), len(test_data)))

    sys.stdout.flush()

    train_data = _data_to_nparray(train_data, vocab, args)
    dev_data = _data_to_nparray(dev_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    return train_data, dev_data, test_data, vocab


def data_loader(origin_data, batch_size, num_epochs=1):
    """
    Generates a batch iterator for a dataset.
    """
    data = {}
    for key, value in origin_data.items():
        data[key] = np.copy(value)

    data_size = len(data['text_len'])
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # shuffle the dataset at the begging of each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for key, value in data.items():
            data[key] = value[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            max_text_len = max(data['text_len'][start_index:end_index])

            yield (data['text'][start_index:end_index, :max_text_len],
                   data['text_len'][start_index:end_index],
                   data['label'][start_index:end_index],
                   data['raw'][start_index:end_index])
