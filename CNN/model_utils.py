import torch
import torch.nn.functional as F
import torch.nn as nn


def get_model(vocab, args):
    print("\nBuilding model...")

    model = CNN(vocab, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model


class CNN(nn.Module):
    def __init__(self, vocab, args):
        super(CNN, self).__init__()
        self.args = args

        # Word embedding
        vocab_size, embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors
        self.embedding_layer.weight.requires_grad = False

        if args.fine_tune_wv == 'True':
            self.tune_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
            self.tune_embedding_layer.weight.data = vocab.vectors
            embedding_dim *= 2

        # Convolution
        self.convs = nn.ModuleList([nn.Conv1d(
                    in_channels=embedding_dim, out_channels=args.num_filters,
                    kernel_size=K) for K in args.filter_sizes])

        self.num_filters_total = args.num_filters * len(args.filter_sizes)

        if args.hidden_dim != 0:
            self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_filters_total, args.hidden_dim),
                    nn.Dropout(args.dropout),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim, args.num_classes))
        else:
            self.seq = nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.ReLU(),
                    nn.Linear(self.num_filters_total, args.num_classes))

        self.dropout = nn.Dropout(args.dropout)

    def _conv_max_pool(self, x, conv_filter):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3)

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        # Apply the 1d conv. Resulting dimension is
        # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        x = [conv(x) for conv in conv_filter]

        # max pool over time. Resulting dimension is
        # [batch_size, num_filters] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]

        # concatenate along all filters. Resulting dimension is
        # batch_size, num_filters_total
        x = torch.cat(x, 1)

        return x

    def forward(self, text):
        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        x = self.embedding_layer(text).float()
        if self.args.fine_tune_wv == 'True':
            x = torch.cat([x, self.tune_embedding_layer(text).float()], dim=2)

        x = self.dropout(x)

        # apply 1d conv + max pool, result:  batch_size, num_filters_total
        x = self._conv_max_pool(x, self.convs)

        # apply MLP
        x = self.seq(x)

        return x
