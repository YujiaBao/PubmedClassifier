import argparse
import os
import train_utils as train_utils
import data_utils as data_utils
import model_utils as model_utils
import torch
import pickle

parser = argparse.ArgumentParser(description='Pubmed Abstract Classifier')

# learning
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout zero probability [default: 0.1]')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Maximum num. of epochs for train [default: 100]')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size for training [default: 50]')
parser.add_argument('--patience', type=int, default=10,
                    help=('Reduce learning rate (by a half) when dev loss stop'
                          'improving during the last "patience" evaluations. '
                          '[default: 10]'))

# model configuration
parser.add_argument('--num_filters', type=int, default=50,
                    help="Num of filters per filter size [default: 50]")
parser.add_argument('--filter_sizes', type=str, default="3,4,5",
                    help="Filter sizes [default: 3,4,5]")
parser.add_argument('--hidden_dim',   type=int, default=50,
                    help=('Dim. of the hidden layer (between conv and output) '
                          '[default: 100]'))
parser.add_argument('--fine_tune_wv', type=str, default='False',
                    help='Set this on to fine tune word vectors.')
parser.add_argument('--seed', type=int, default=1,
                    help='seed')

# data loading
parser.add_argument('--data_dir', type=str, default='',
                    help='data dir (leave it blank if run on local machine)')
parser.add_argument('--data_size', type=int, default=0,
                    help='whether to use a subset of the training data')
parser.add_argument('--task', type=str, default='penetrance',
                    help='task id, either penetrance or incidence')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Num of classes [default: 2]')
parser.add_argument('--word_vector', type=str, default='wiki.en.vec',
                    help=('Name of pretrained word embeddings.'
                          'Options: charngram.100d fasttext.en.300d '
                          'fasttext.simple.300d glove.42B.300d '
                          'glove.840B.300d glove.twitter.27B.25d '
                          'glove.twitter.27B.50d glove.twitter.27B.100d '
                          'glove.twitter.27B.200d glove.6B.50d glove.6B.100d '
                          'glove.6B.200d glove.6B.300d '
                          '[Default: fasttext.simple.300d]'))

parser.add_argument('--fold', type=int, default=0)

# option
parser.add_argument('--cuda', type=int, default=-1,
                    help='run on gpu')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model snapshot after training')
parser.add_argument('--snapshot', type=str, default=None,
                    help='path for loading model snapshot [default: None]')
parser.add_argument('--result_path', type=str, default=None,
                    help=('Path to store a pickle file of the resulting '
                          'performance [default: None]'))


if __name__ == '__main__':
    args = parser.parse_args()
    args.filter_sizes = [int(K) for K in args.filter_sizes.split(',')]
    torch.manual_seed(args.seed)

    # update args and print
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    train_data, dev_data, test_data, vocab = data_utils.load_dataset(args)
    print('Vocabular size: ', end='')
    print(vocab.vectors.shape)

    # ------------------------------------------------------------------------
    # Get Model (either create a new one or load it from snapshot)
    # ------------------------------------------------------------------------
    if args.snapshot is None:
        model = model_utils.get_model(vocab, args)
    else:
        # load saved model
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except Exception as e:
            print(e)
            exit(1)

    print("Load complete")

    # Train the model on train_data, use dev_data for early stopping
    model, dev_res = train_utils.train(train_data, dev_data, model, args)

    # Evaluate the trained model
    print("Evaluate on train set")
    train_res = train_utils.evaluate(train_data, model, args)

    print("Evaluate on test set")
    test_res = train_utils.evaluate(test_data, model, args, roc=True)

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind('/')]
        if not os.path.exists(directory):
            os.makedirs(directory)

        result = {
                'train_loss':      train_res[0],
                'train_acc':       train_res[1],
                'train_recall':    train_res[2],
                'train_precision': train_res[3],
                'train_f1':        train_res[4],
                'dev_loss':        dev_res[0],
                'dev_acc':         dev_res[1],
                'dev_recall':      dev_res[2],
                'dev_precision':   dev_res[3],
                'dev_f1':          dev_res[4],
                'test_loss':       test_res[0],
                'test_acc':        test_res[1],
                'test_recall':     test_res[2],
                'test_precision':  test_res[3],
                'test_f1':         test_res[4],
                'tpr':             test_res[5],
                }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
