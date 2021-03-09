from typing import List, Dict
import unidecode
import string
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# This function is preprocessing a single sentence from the database

#Defining the global Tokens.
SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
MAX_LENGTH=20

def preprocess(sentence: str, hindi=False) -> str:
    # remove tabs and newlines
    sentence = ' '.join(sentence.split())
    ## Write your code below
    #convert the sentence into lower cases
    sentence = #Remove this comment before writing code here.
    # remove accented chars such as in cafe
    sentence = unidecode.unidecode(sentence)
    # remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    # remove hindi digits
    if hindi:
        sentence = re.sub("[२३०८१५७९४६]", "", sentence)
    ##Write your code below
    #remove trailing and leading extra white spaces
    sentence = #
    ##Write your code below
    #Join the senteces after removing all the extra spaces using split()
    sentence = #
    ##Write your code below
    #append the prepend the SOS token and append the EOS token to the sentence with spaces.
    sentence = #
    return sentence

#helper function 1. Returns a list of all the unique words in our corpora.

def get_vocab(lang: pd.Series) -> List:
    all_words = set()
    for sent in lang:
        for word in sent.split():
            if word not in all_words:
                all_words.add(word)

    return sorted(list(all_words))

#Helper 2: Creates a dictionary with token-> index mapping. Used in encoding.
def token_idx(words: List) -> Dict:
    word2idx = {'SOS': 1, 'EOS': 2, 'PAD': 0}
    counter = 3
    for word in words:
        if word not in word2idx:
            word2idx[word] = counter
            counter += 1
    return word2idx

#Helper 3: Creates a dictionary for index to word mapping. Used in decoding
def idx_token(wor2idx: Dict) -> Dict:
    idx2word = {1: 'SOS', 2: 'EOS', 'PAD': 0}
    for word, idx in wor2idx.items():
        if idx not in idx2word:
            idx2word[idx] = word

    return idx2word

#Helper 4: Pads sequences to a particular length so that all the sequences are of same length in a batch.
def pad_sequences(x, max_len=20):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded

#Converts a particular sentence to its corresponding numeric tensor using word2idx dictionary.
def convert_to_tensor(word2idx: Dict, sentences: pd.Series):
    tensor = [[word2idx[s] for s in eng.split()]
              for eng in sentences.values.tolist()]
    tensor = [pad_sequences(x) for x in tensor]
    return tensor

#Class of type Dataset. This must contain __len__() and __getitem() functions as a part of their hooks.
class Data(Dataset):
    def __init__(self, input_sent, target_sent):
        self.input_sent = input_sent
        self.target_sent = target_sent

    def __len__(self):
        return len(self.input_sent)

    def __getitem__(self, index):
        x = self.input_sent[index]
        y = self.target_sent[index]
        return x, y

#Main function being called when we need to retrieve inout batch, output batch and DataLoader objects.
def get_dataset(batch_size=2, types="train", shuffle=True, num_workers=1, pin_memory=False, drop_last=True):
    #Read the file 
    lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
    #Get lines only with source as "ted"
    lines = lines[lines['source'] == 'ted']  # Remove other sources
    ## Write your code below
    #Remove duplicate lines
    #
    #Random Sample of 25000 sentences
    lines = lines.sample(n=25000, random_state=42)
    ##Write your code below
    #Call preprocess functions on all english sentences
    #

    #Call preprocess functions on all hindi sentences
    #

    #Retrieve length of all english sentences and store it in the lines dataframe under colums "length_english_sentence"
    #

    #Retrieve length of all english sentences and store it in the lines dataframe under colums "length_english_sentence"
    #

    #Remove all the sentences with length less than max_length
    #
    #

    #Get List of english words
    #       
    
    #Get List of Hindi Words
    #

    #Get word2idx_eng for english
    #
    
    #Get word2idx_hin for hindi
    #

    #get idx2word_eng for english
    #

    #get idx2word_hin for hindi
    #
    
    #Convert the sentences to tensors using dictionaries created above
    #English tensor in input_tensor
    #Hindi tensor in output_tensor

    #Calling the split function and doing a 80-20 split on the input and target tensors.
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42)

    #Calling the Dataloaders and passing the Dataset type objects created using Data() class.
    if types == "train":
        train_dataset = Data(input_tensor_train, target_tensor_train)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), english_words, hindi_words
    elif types == "val":
        val_dataset = Data(input_tensor_val, target_tensor_val)
        return DataLoader(val_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), idx2word_eng, idx2word_hin
    else:
        raise ValueError("types must be in ['train','val']")


if __name__ == "__main__":
    get_dataset()
