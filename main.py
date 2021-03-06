######################################################################
# Training and Evaluating
# =======================
from seq2seq import EncoderRNN,Linear,DecoderRNN
from train import trainIters
# from validate import evaluateRandomly
from preprocess import get_dataset
import argparse
import torch
import os
SOS_token = 1   # SOS_token: start of sentence
EOS_token = 2   # EOS_token: end of sentence

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")

# Parser
parser = argparse.ArgumentParser(description='Creating Classifier')

######################
# Optimization Flags #
######################

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs_per_lr_drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

##################
# Training Flags #
##################
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs for training for training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_epoch', default=600, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--save_folder', default=os.path.expanduser('~/weights'), help='Location to save checkpoint models')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

###############
# Model Flags #
###############

parser.add_argument('--auto_encoder', default=True, type=str2bool, help='Use auto-encoder model')
parser.add_argument('--MAX_LENGTH', default=20, type=int, help='Maximum length of sentence')
parser.add_argument('--bidirectional', default=False, type=str2bool, help='bidirectional LSRM')
parser.add_argument('--hidden_size_decoder', default=256, type=int, help='Decoder Hidden Size')
parser.add_argument('--num_layer_decoder', default=1, type=int, help='Number of LSTM layers for decoder')
parser.add_argument('--hidden_size_encoder', default=256, type=int, help='Eecoder Hidden Size')
parser.add_argument('--num_layer_encoder', default=1, type=int, help='Number of LSTM layers for encoder')
parser.add_argument('--teacher_forcing', default=False, type=str2bool, help='If using the teacher frocing in decoder')

# Add all arguments to parser
args = parser.parse_args()

##############
# Cuda Flags #
##############
if args.cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

###############################
# Creating the dataset object #
###############################
# Create training data object
trainset,source_vocab,target_vocab = get_dataset(types="train",batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=False,drop_last=True)
print(len(target_vocab),len(source_vocab))
encoder1 = EncoderRNN(args.hidden_size_encoder,len(source_vocab)+2, args.batch_size, num_layers=args.num_layer_encoder, bidirectional=args.bidirectional).to(device)
bridge = Linear(args.bidirectional, args.hidden_size_encoder, args.hidden_size_decoder).to(device)
decoder1 = DecoderRNN(args.hidden_size_decoder,len(target_vocab)+2, args.batch_size, num_layers=args.num_layer_decoder).to(device)

trainIters(trainset,encoder1, decoder1, bridge,num_epochs=10,batch_size=args.batch_size,print_every=10,device=device)
torch.save(encoder1,"encoder.pt")
torch.save(decoder1,"decoder.pt")
torch.save(bridge,"bridge.pt")



######################################################################
# evaluateRandomly(encoder1, decoder1, bridge)