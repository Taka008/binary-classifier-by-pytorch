import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 device,
                 emb_size=256,
                 bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_layer_num = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.emb_size = emb_size

        self.embedding = nn.Embedding(self.input_size, self.emb_size, padding_idx=1)
        self.bi_lstm = RNNWrapper(rnn=nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
                                              num_layers=self.encoder_layer_num, bidirectional=self.bidirectional,
                                              batch_first=True, dropout=self.dropout_rate))
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, x, mask):
        x = self.embedding(x)
        lstm_out = self.bi_lstm(x, mask)
        out = lstm_out.sum(dim=1)
        h = self.tanh(self.linear1(out))
        y = self.linear2(h)
        return y


class RNNWrapper(nn.Module):
    def __init__(self, rnn: nn.Module):
        super(RNNWrapper, self).__init__()
        self.rnn = rnn

    def forward(self, x, mask):
        lengths = mask.sum(dim=1)
        sorted_lengths, sorted_indices = lengths.sort(0, descending=True)
        sorted_input = x[sorted_indices]
        _, unsorted_indices = sorted_indices.sort(0)

        # masking
        # ignore padding
        packed = pack_padded_sequence(sorted_input, lengths=sorted_lengths, batch_first=True)
        output, _ = self.rnn(packed)
        # padding
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)
        unsorted_input = unpacked[unsorted_indices]

        return unsorted_input
