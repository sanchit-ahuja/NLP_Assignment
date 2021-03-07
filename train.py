######################
# Training the Model #
######################
from torch import optim
import torch
import torch.nn as nn
from utils.transformer import *
from torch import device
def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer,device, criterion, max_length,batch_size=32,bidirectional=False,teacher_forcing=True):
    # To train, each element of the input sentence will be fed to the encoder.
    # At the decoding phase``<SOS>`` will be fed as the first input to the decoder
    # and the last hidden (state,cell) of the encoder will play the role of the first hidden (cell,state) of the decoder.

    # optimizer steps
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    bridge_optimizer.zero_grad()

    # Define a list for the last hidden layer
    encoder_hiddens_last = []
    loss = 0

    #################
    #### DECODER ####
    #################
    for step_idx in range(batch_size):
        # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
        # the new input sequence as a continuation of the previous sequence.
        encoder_hidden = encoder.initHidden(device)
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
        input_length = input_tensor_step.size(0)

        # Switch to bidirectional mode
        

        
        encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor_step[ei], encoder_hidden)
            encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

            # only return the hidden and cell states for the last layer and pass it to the decoder
        hn, cn = encoder_hidden
        encoder_hn_last_layer = hn[-1].view(1,1,-1)
        encoder_cn_last_layer = cn[-1].view(1,1,-1)
        encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]

        # A linear layer to establish the connection between the encoder/decoder layers.
        encoder_hidden = [bridge(item) for item in encoder_hidden]
        encoder_hiddens_last.append(encoder_hidden)

    #################
    #### DECODER ####
    #################
    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hiddens = encoder_hiddens_last

    # teacher_forcing uses the real target outputs as the next input
    # rather than using the decoder's prediction.
    if teacher_forcing:

        for step_idx in range(batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                decoder_input = target_tensor_step[di]  # Teacher forcing

        loss = loss / batch_size

    else:
        for step_idx in range(batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                if decoder_input.item() == EOS_token:
                    break
        loss = loss /batch_size

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




def trainIters(trainloader,encoder, decoder, bridge,device,bidirectional=False,teacher_forcing=True,num_epochs=600,batch_size=32,max_length=20, print_every=1000, plot_every=100, learning_rate=0.1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    bridge_optimizer = optim.SGD(bridge.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_iters_per_epoch = int(len(trainloader) /batch_size)
    for i in range(num_epochs):
        print("epoch {} begins".format(i))

        for iteration, data in enumerate(trainloader, 1):
            print("training batch {} in epoch {}".format(iteration,i))

            # Get a batch
            training_pair = data
            print("training pair shape {} {}".format(training_pair[0].shape,training_pair[1].shape))

            # Input
            input_tensor = training_pair[0]
            input_tensor, mask_input = reformat_tensor_mask(input_tensor)

            # Target
            target_tensor = training_pair[1]
            target_tensor, mask_target = reformat_tensor_mask(target_tensor)

            if device == torch.device("cuda"):
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            loss = train(input_tensor, target_tensor, mask_input, mask_target, encoder,
                         decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer,device, criterion,max_length,batch_size)
            print_loss_total += loss
            plot_loss_total += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / n_iters_per_epoch),
                                             iteration, iteration / n_iters_per_epoch * 100, print_loss_avg))

            if iteration % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


        print('####### Finished epoch %d of %d ########' % (i+1,num_epochs))