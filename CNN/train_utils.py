import os
import sys
import copy
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import data_utils as data_utils
import datetime
import time
import numpy as np
from termcolor import colored
import math
from tqdm import tqdm
from scipy import interp


def _to_tensor(x_list, cuda):
    if type(x_list) is tuple:
        x_list = list(x_list)

    if type(x_list) is not list:
        x_list = [x_list]

    res_list = []
    for x in x_list:
        y = torch.from_numpy(x)
        if cuda != -1:
            y = y.cuda(cuda)
        res_list.append(y)

    if len(res_list) == 1:
        return res_list[0]
    else:
        return tuple(res_list)


def _to_numpy(x_list):
    if type(x_list) is not list:
        x_list = [x_list]

    res_list = []
    for x in x_list:
        res_list.append(x.data.cpu().numpy())

    if len(res_list) == 1:
        return res_list[0]
    else:
        return tuple(res_list)


def _to_number(x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.cpu().item()
        else:
            return x.item()
    else:
        return x


def _compute_score(y_pred, y_true, num_classes=2):
    '''
    Compute the accuracy, f1, recall and precision
    '''
    if num_classes == 2:
        average = "binary"
    else:
        average = "weighted"

    acc = metrics.accuracy_score(
            y_pred=y_pred, y_true=y_true)
    f1 = metrics.f1_score(
            y_pred=y_pred, y_true=y_true, average=average)
    recall = metrics.recall_score(
            y_pred=y_pred, y_true=y_true, average=average)
    precision = metrics.precision_score(
            y_pred=y_pred, y_true=y_true, average=average)

    return acc, f1, recall, precision


def train(train_data, dev_data, model, args):
    best = 100
    sub_cycle = 0
    best_model = None

    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.patience, factor=0.1, verbose=True)

    for ep in range(args.num_epochs):
        start = time.time()

        batches = data_utils.data_loader(train_data, args.batch_size, 1)

        if args.data_dir == '':
            for batch in tqdm(
                    batches,
                    total=math.ceil(len(train_data['label'])/args.batch_size)):
                train_batch(model, batch, optimizer, args)
        else:
            for batch in batches:
                train_batch(model, batch, optimizer, args)

        end = time.time()
        print("{}, Epoch {:3d}, Time Cost: {} seconds".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            ep, end-start))

        if ep % 1 == 0:
            print("Train:", end=" ")
            evaluate(train_data, model, args)

        print("Dev  :", end=" ")
        cur_loss, _, _, _, _, _ = evaluate(dev_data, model, args)

        scheduler.step(cur_loss)

        if cur_loss < best:
            best = cur_loss
            best_model = copy.deepcopy(model)
            sub_cycle = 0
        else:
            sub_cycle += 1

        if sub_cycle == args.patience*2:
            break

        sys.stdout.flush()

    print("End of training. Restore the best weights")
    model = copy.deepcopy(best_model)

    print("Best development performance during training")
    loss, acc, recall, precision, f1, _ = evaluate(dev_data, model, args)

    if args.save:
        # get time stamp for snapshot path
        timestamp = str(int(time.time() * 1e7))
        out_dir = os.path.abspath(
                os.path.join(os.path.curdir, "runs", timestamp))

        print("Saving the model to {}\n".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("Save the best model")
        torch.save(model, os.path.join(out_dir, "best"))
        print("Best model is saved to {:s}".format(
            os.path.join(out_dir, "best")))

    return model, (loss, acc, recall, precision, f1)


def train_batch(model, batch, optimizer, args, verbose=False):
    '''
        Train the generator and classifier jointly on one batch of examples
    '''
    model.train()
    optimizer.zero_grad()

    # ------------------------------------------------------------------------
    # Preparing the data
    # ------------------------------------------------------------------------
    text, text_len, label, _ = batch
    text, label = _to_tensor([text, label], args.cuda)

    # Run the model
    out = model(text)
    loss = F.cross_entropy(out, label)

    loss.backward()
    optimizer.step()


def evaluate_batch(model, batch, args):
    model.eval()

    # Preparing the data
    # ------------------------------------------------------------------------
    text, text_len, label, raw = batch

    text, label = _to_tensor([text, label], args.cuda)

    # Run the model, use hard attention
    out = model(text)

    loss = F.cross_entropy(out, label)
    pred_label = np.argmax(_to_numpy(out), axis=1)

    return _to_numpy(label), pred_label, _to_number(loss), _to_numpy(out[:, 1])


def evaluate(test_data, model, args, roc=False):
    total_true = np.array([], dtype=int)
    total_pred = np.array([], dtype=int)
    total_out = np.array([], dtype=int)
    total_loss = []

    batches = data_utils.data_loader(test_data, args.batch_size, 1)
    for batch in batches:
        true, pred, loss, out = evaluate_batch(model, batch, args)

        total_true = np.concatenate((total_true, true))
        total_pred = np.concatenate((total_pred, pred))
        total_out = np.concatenate((total_out, out))
        total_loss.append(loss)

    loss_total = sum(total_loss)/len(total_loss)

    acc, f1, recall, precision = _compute_score(
        y_pred=total_pred, y_true=total_true)

    tpr = None
    if roc:
        fpr, tpr, thresholds = metrics.roc_curve(
                total_true, total_out, pos_label=1)

        mean_fpr = np.linspace(0, 1, 100)
        tpr = interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0

    print("{}, {:s} {:.6f}, "
          "{:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            colored("loss", "red"),
            loss_total,
            colored(" acc", "blue"),
            acc,
            colored("recall", "blue"),
            recall,
            colored("precision", "blue"),
            precision,
            colored("f1", "blue"),
            f1))

    return loss_total, acc, recall, precision, f1, tpr
