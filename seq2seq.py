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
         For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)


    def forward(self, input, hidden,bidirectional=False):

        
        
            # Make the data in the correct format as the RNN input.
        embedded = self.embedding(input).view(1, 1, -1)
        rnn_input = embedded
            # The following descriptions of shapes and tensors are extracted from the official Pytorch documentation:
            # output-shape: (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
            # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state
            # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state
        output, (h_n, c_n) = self.lstm(rnn_input, hidden)
        return output, (h_n, c_n)

    def initHidden(self,device):
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
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, (h_n, c_n)

    def initHidden(self,device):
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
        self.linear_connection_op = nn.Linear(num_directions * hidden_size_encoder, hidden_size_decoder)
        self.connection_possibility_status = num_directions * hidden_size_encoder == hidden_size_decoder

    def forward(self, input):

        if self.connection_possibility_status:
            return input
        else:
            return self.linear_connection_op(input)



# for i,data in enumerate(trainset,1):
#     input_data=data[0]
#     print(input_data.shape)
#     input_data,input_mask=reformat_tensor_mask(input_data)
#     output_data=data[1]
#     output_data,output_mask=reformat_tensor_mask(output_data)
#     print(output_data.shape,input_data.shape)
#     encoder1 = EncoderRNN(args.hidden_size_encoder, len(source_vocab), args.batch_size, num_layers=args.num_layer_encoder, bidirectional=args.bidirectional).to(device)
#     # init_states=encoder1.initHidden()
#     encoder_hiddens_last=[]
#     bridge=Linear(args.bidirectional,args.hidden_size_encoder,args.hidden_size_decoder)
#     decoder=DecoderRNN(args.hidden_size_decoder,len(target_vocab),args.batch_size)

#     for step_idx in range(args.batch_size):
#         # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
#         # the new input sequence as a continuation of the previous sequence.
#         encoder_hidden = encoder1.initHidden()
#         # print(encoder_hidden)
#         input_tensor_step=input_data[:,step_idx][input_data[:,step_idx]!=0]
#         input_length=input_tensor_step.size(0)
#         # print(input_tensor_step.shape,input_length)
#         encoder_outputs = torch.zeros(args.batch_size,20, encoder1.hidden_size, device=device)
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder1(
#                 input_tensor_step[ei], encoder_hidden)
#             encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]
#         hn, cn = encoder_hidden
#         encoder_hn_last_layer = hn[-1].view(1,1,-1)
#         encoder_cn_last_layer = cn[-1].view(1,1,-1)
#         encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]
#         encoder_hidden = [bridge(item) for item in encoder_hidden]
#         encoder_hiddens_last.append(encoder_hidden)
#     decoder_input = torch.tensor([SOS_token], device=device)
#     decoder_hiddens = encoder_hiddens_last
#     for step_idx in range(args.batch_size):
#             # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
#             # the new input sequence as a continuation of the previous sequence

#             target_tensor_step = output_data[:, step_idx][output_data[:, step_idx] != 0]
#             target_length = target_tensor_step.size(0)
#             decoder_hidden = decoder_hiddens[step_idx]

#             # Without teacher forcing: use its own predictions as the next input
#             for di in range(target_length):
#                 # decoder_output, decoder_hidden, decoder_attention = decoder(
#                 #     decoder_input, decoder_hidden, encoder_outputs)
#                 decoder_output, decoder_hidden = decoder(
#                     decoder_input, decoder_hidden)
                
#                 topv, topi = decoder_output.topk(1)
#                 print("top value and its corresponding index: {} {} and word {}".format(topv,topi,di))
#                 decoder_input = topi.squeeze().detach()
#                 print("word number {} has decoder output as {}".format(di,decoder_input))


#     break
