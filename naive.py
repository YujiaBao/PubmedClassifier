import argparse
import itertools as it
import json
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import sys
import multiprocessing
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import pickle
import os
import random


parser = argparse.ArgumentParser(description='Pubmed Abstract Classifier')
# data loading
parser.add_argument('--data_dir', type=str, default='',
                    help='data dir (leave it blank if run on local machine)')
parser.add_argument('--task', type=str, default='penetrance',
                    help='task id, either penetrance or incidence')


def _compute_score(y_pred, y_true, num_classes):
    '''
    Compute the accuracy, f1, recall and precision
    '''
    if num_classes == 2:
        average = "binary"
    else:
        average = "weighted"

    acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = metrics.f1_score(y_pred=y_pred, y_true=y_true, average=average,
                          pos_label=1)
    recall = metrics.recall_score(
            y_pred=y_pred, y_true=y_true, average=average)
    precision = metrics.precision_score(
            y_pred=y_pred, y_true=y_true, average=average)

    return acc, f1, recall, precision


def _data2list(data):
    text, label = [], []
    for example in data:
        label.append(example['label'])
        text.append(' '.join(example['text']))

    return text, label


def train_and_evaluate(writer, train_data, dev_data, test_data):
    train_text, train_label = _data2list(train_data)
    dev_text, dev_label = _data2list(dev_data)
    test_text, test_label = _data2list(test_data)

    hyper_parms = {
            'ngram_range': [(1, 2), (1, 3), (1, 4)],
            'sublinear_tf': [True, False],
            'alpha': [0.001, 0.01],
            }
    all_hyper_parms = it.product(*(hyper_parms[k] for k in hyper_parms))
    all_hyper_parms_dict = [
            dict(zip(hyper_parms, arg)) for arg in all_hyper_parms]

    best_dev = 0
    best_args = None

    queue = multiprocessing.Queue()
    for cur_args in all_hyper_parms_dict:
        multiprocessing.Process(
                target=_worker,
                args=(queue, cur_args, train_text, train_label, dev_text,
                    dev_label)).start()

    results = []
    for _ in all_hyper_parms_dict:
        cur_args, acc, f1 = queue.get()

        res_str = ''
        for key, value in cur_args.items():
            res_str = res_str + key + ': ' + str(value) + '. '

        if f1 > best_dev:
            best_dev = f1
            best_args = cur_args
            res_str += colored(('Dev acc: %.4f, f1: %.4f' % (acc, f1)), 'red')
        else:
            res_str += ('Dev acc: %.4f, f1: %.4f' % (acc, f1))
        print(res_str)

    text_clf = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=best_args['ngram_range'],
                                 sublinear_tf=best_args['sublinear_tf'])),
        ('clf', MultinomialNB(alpha=best_args['alpha']))
        ])

    train_text = train_text + dev_text
    train_label = train_label + dev_label

    text_clf.fit(train_text, train_label)
    test_pred = text_clf.predict(test_text)
    acc, f1, recall, precision = _compute_score(
            y_pred=test_pred, y_true=test_label, num_classes=2)

    print(text_clf.classes_)
    scores = text_clf.predict_proba(test_text)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(test_label, scores, pos_label=1)

    mean_fpr = np.linspace(0, 1, 100)
    tpr = interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0
    roc = metrics.auc(mean_fpr, tpr)

    print("End of training")
    print("Best dev f1: %.4f" % best_dev)
    print("Test acc: %.4f, f1: %.4f, recall: %.4f, precision: %.4f, roc: %.4f" % (
        acc, f1, recall, precision, roc))

    return acc, f1, roc, tpr


def _worker(queue, cur_args, train_text, train_label, dev_text, dev_label):
    text_clf = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=cur_args['ngram_range'],
                                 sublinear_tf=cur_args['sublinear_tf'])),
        ('clf', MultinomialNB(alpha=cur_args['alpha']))
        ])

    text_clf.fit(train_text, train_label)
    dev_pred = text_clf.predict(dev_text)
    acc, f1, recall, precision = _compute_score(
            y_pred=dev_pred, y_true=dev_label, num_classes=2)

    queue.put((cur_args, acc, f1))


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


def load_dataset(args, fold):
    '''
        Load the train/val/test data from ../data/
    '''
    print("Loading data...")

    train_data = _load_json('../data/fold/' + args.task + '_fold_' + str(fold) + '_train.jsonl')
    dev_data = _load_json('../data/fold/' + args.task + '_fold_' + str(fold) + '_dev.jsonl')
    test_data = _load_json('../data/fold/' + args.task + '_fold_' + str(fold) + '_test.jsonl')

    print('#train: %d, #val: %d, #test: %d'
          % (len(train_data), len(dev_data), len(test_data)))

    sys.stdout.flush()

    return train_data, dev_data, test_data


def split_train_data(train_data, ds):
    if ds == 0:
        return train_data

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

    for i in range(min(len(pos), int(ds * pos_ratio))):
        train_data.append(pos[i])

    tmp = len(train_data)
    for i in range(min(len(neg), ds - tmp)):
        train_data.append(neg[i])

    return train_data


def save_jsonl(obj, name):
    '''
        Write the list of json object.
    '''
    if os.path.isfile(name):
        os.system("rm {}".format(name))

    with open(name, 'w') as f:
        for d in obj:
            json.dump(d, f)
            f.write(os.linesep)


if __name__ == '__main__':
    args = parser.parse_args()

    data_size = [0, 50, 100, 200, 400, 800, 1600]

    results = []

    for ds in data_size:
        for fold in range(10):
            print("===Fold %d===" % fold)
            train_data, dev_data, test_data = load_dataset(args, fold)
            train_data = split_train_data(train_data, ds)

            acc, f1, roc, tpr = train_and_evaluate(
                    sys.stdout, train_data, dev_data, test_data)

            results.append({
                    'data_size': ds,
                    'fold': fold,
                    'task': args.task,
                    'acc': acc,
                    'f1': f1,
                    'roc': roc,
                    'tpr': tpr.tolist(),
                })

    save_jsonl(results, 'nb.jsonl')

    # print(acc_list)
    # print(f1_list)

    # print(st.t.interval(
    #     0.95, len(acc_list)-1, loc=np.mean(acc_list), scale=st.sem(acc_list)))
    # print(st.t.interval(
    #     0.95, len(f1_list)-1, loc=np.mean(f1_list), scale=st.sem(f1_list)))

    # print(roc_list)
    # print(st.t.interval(
    #     0.95, len(roc_list)-1, loc=np.mean(roc_list), scale=st.sem(roc_list)))

    # acc = np.mean(np.array(acc_list))
    # start, end = st.t.interval(0.95, len(acc_list)-1, loc=np.mean(acc_list),
    #         scale=st.sem(acc_list))
    # print('ACC: {:.4f} ({:.4f}, {:.4f})'.format(acc, start, end))

    # f1 = np.mean(np.array(f1_list))
    # start, end = st.t.interval(0.95, len(f1_list)-1, loc=np.mean(f1_list),
    #         scale=st.sem(f1_list))
    # print('f1: {:.4f} ({:.4f}, {:.4f})'.format(f1, start, end))

    # roc = np.mean(np.array(roc_list))
    # start, end = st.t.interval(0.95, len(roc_list)-1, loc=np.mean(roc_list),
    #         scale=st.sem(roc_list))
    # print('ROC: {:.4f} ({:.4f}, {:.4f})'.format(roc, start, end))

    # with open('roc_' + args.task + '.pkl', 'wb') as f:
    #     pickle.dump(tpr_list, f, pickle.HIGHEST_PROTOCOL)
