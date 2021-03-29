from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from preprocess import get_dataset
from utils.transformer import *
import argparse


# Predefined tokens


# Extract the languages' attributes
# input_lang, output_lang = trainset.langs()

# The trainloader for parallel processing

# iterate through training

# Create testing data object
# valset = get_dataset(types="val",batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=False,drop_last=True)


#####################
# Encoder / Decoder #
#####################

class EncoderRNN(nn.Module):
    """
    The encoder generates a single output vector that embodies the input sequence meaning.
    The general procedure is as follows:
        1. In each step, a word will be fed to a network and it generates
         an output and a hidden state.
        2. For the next step, the hidden step and the next word will
         be fed to the same network (W) for updating the weights.
        3. In the end, the last output will be the representative of the input sentence (called the "context vector").
    """

    def __init__(self, hidden_size, input_size, batch_size, num_layers=1, bidirectional=False):
        """
        * For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        :param batch_size: The batch_size for mini-batch optimization.
        :param num_layers: Number of RNN layers. Default: 1
        :param bidirectional: If the encoder is a bi-directional LSTM. Default: False
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        # The input should be transformed to a vector that can be fed to the network.
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)

        # The LSTM layer for the input

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input, hidden, bidirectional=False):

        # Make the data in the correct format as the RNN input.
        embedded = self.embedding(input).view(1, 1, -1)
        rnn_input = embedded
        # The following descriptions of shapes and tensors are extracted from the official Pytorch documentation:
        # output-shape: (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
        # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state
        # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state
        output, (h_n, c_n) = self.lstm(rnn_input, hidden)
        return output, (h_n, c_n)

    def initHidden(self, device):
        encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                         torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
        return encoder_state


class DecoderRNN(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    Decoding is as follows:
    1. At each step, an input token and a hidden state is fed to the decoder.
        * The initial input token is the <SOS>.
        * The first hidden state is the context vector generated by the encoder (the encoder's
    last hidden state).
    2. The first output, shout be the first sentence of the output and so on.
    3. The output token generation ends with <EOS> being generated or the predefined max_length of the output sentence.
    """

    def __init__(self, hidden_size, output_size, batch_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, (h_n, c_n)

    def initHidden(self, device):
        """
        The spesific type of the hidden layer for the RNN type that is used (LSTM).
        :return: All zero hidden state.
        """
        return [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]


class Linear(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    In case that their dimension is not matched, a linear layer should be used to transformed the context vector
    to a suitable input (shape-wise) for the decoder cell state (including the memory(Cn) and hidden(hn) states).
    The shape mismatch is True in the following conditions:
    1. The hidden sizes of encoder and decoder are the same BUT we have a bidirectional LSTM as the Encoder.
    2. The hidden sizes of encoder and decoder are NOT same.
    3. ETC?
    """

    def __init__(self, bidirectional, hidden_size_encoder, hidden_size_decoder):
        super(Linear, self).__init__()
        self.bidirectional = bidirectional
        num_directions = int(bidirectional) + 1
        self.linear_connection_op = nn.Linear(
            num_directions * hidden_size_encoder, hidden_size_decoder)
        self.connection_possibility_status = num_directions * \
            hidden_size_encoder == hidden_size_decoder

    def forward(self, input):

        if self.connection_possibility_status:
            return input
        else:
            return self.linear_connection_op(input)