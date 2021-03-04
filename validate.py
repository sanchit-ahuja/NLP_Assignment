##############
# Evaluation #
##############
#
# In evaluation, we simply feed the sequence and observe the output.
# The generation will be over once the "EOS" has been generated.
import torch
from utils.transformer import *
SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
def evaluate(encoder, decoder, bridge, input_tensor,device,index2word_hin, max_length=20,bidirectional=False):


    # Required for tensor matching.
    # Remove to see the results for educational purposes.
    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(device)

        if bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']

            for ei in range(input_length):
                (encoder_hidden_forward, encoder_hidden_backward) = encoder(
                    (input_tensor[ei],input_tensor[input_length - 1 - ei]), (encoder_hidden_forward,encoder_hidden_backward))

            # Extract the hidden and cell states
            hn_forward, cn_forward = encoder_hidden_forward
            hn_backward, cn_backward = encoder_hidden_backward

            # Concatenate the hidden and cell states for forward and backward paths.
            encoder_hn = torch.cat((hn_forward, hn_backward), 2)
            encoder_cn = torch.cat((cn_forward, cn_backward), 2)


            # Only return the hidden and cell states for the last layer and pass it to the decoder
            encoder_hn_last_layer = encoder_hn[-1].view(1, 1, -1)
            encoder_cn_last_layer = encoder_cn[-1].view(1,1,-1)

            # The list of states
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        else:
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)

            # only return the hidden and cell states for the last layer and pass it to the decoder
            hn, cn = encoder_hidden
            encoder_hn_last_layer = hn[-1].view(1,1,-1)
            encoder_cn_last_layer = cn[-1].view(1,1,-1)
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(index2word_hin[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words


######################################################################
def evaluateRandomly(encoder, decoder, bridge,device,testset,idx2word_en,idx2word_hin, n=10):
    j=0
    for i,data in enumerate(testset,1):
        j=j+1
        pair = data
        input_tensor, mask_input = reformat_tensor_mask(pair[0].view(1,1,-1))
        print(input_tensor.shape)
        input_tensor = input_tensor[input_tensor != 0]
        output_tensor, mask_output = reformat_tensor_mask(pair[1].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        input_sentence = ' '.join(SentenceFromTensor_(idx2word_en, input_tensor))
        output_sentence = ' '.join(SentenceFromTensor_(idx2word_hin, output_tensor))
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        input_tensor=input_tensor.to(device)
        output_words = evaluate(encoder, decoder, bridge, input_tensor,device,idx2word_hin)
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')
        if(j==n):
          break

from preprocess import get_dataset
device = torch.device("cpu")
testset,idx2word_en,idx2word_hin = get_dataset(batch_size=1,types="val",shuffle=False,num_workers=1,pin_memory=False,drop_last=False)
encoder=torch.load("encoder.pt")
encoder=encoder.to(device)

decoder=torch.load("decoder.pt")
decoder=decoder.to(device)
bridge=torch.load("bridge.pt")
bridge=bridge.to(device)
evaluateRandomly(encoder,decoder,bridge,device,testset,idx2word_en,idx2word_hin)
  