from typing import List, Dict
import unidecode
import string
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# This function is preprocessing a single sentence from the database

SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0


def preprocess(sentence: str, hindi=False) -> str:
    # remove tabs and newlines
    sentence = ' '.join(sentence.split())
    # sentence to lower
    sentence = sentence.lower()
    # remove accented chars such as in cafe
    # sentence = unidecode.unidecode(sentence)
    # remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    # remove whitespaces
    if hindi:
        sentence = re.sub("[२३०८१५७९४६]", "", sentence)
    sentence = sentence.strip()
    # removing extra spaces
    sentence = ' '.join(sentence.split())
    sentence = "SOS" + " " + sentence + " " + "EOS"
    return sentence


def get_vocab(lang: pd.Series) -> List:
    all_words = set()
    for sent in lang:
        for word in sent.split():
            if word not in all_words:
                all_words.add(word)

    return sorted(list(all_words))


def token_idx(words: List) -> Dict:
    word2idx = {'SOS': 1, 'EOS': 2, 'PAD':0}
    counter = 3
    for word in words:
        if word not in word2idx:
            word2idx[word] = counter
            counter += 1
    return word2idx


def idx_token(wor2idx: Dict) -> Dict:
    idx2word = {1: 'SOS', 2: 'EOS', 'PAD':0}
    for word, idx in wor2idx.items():
        if idx not in idx2word:
            idx2word[idx] = word
    
    return idx2word

def pad_sequences(x, max_len = 20):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

class Data(Dataset):
    def __init__(self, input_sent, target_sent):
        self.input_sent = input_sent
        self.target_sent = target_sent
    
    def __len__(self):
        return len(self.input_sent)
    
    def __getitem__(self, index):
        x = self.input_sent[index]
        y = self.target_sent[index]
        return x,y
    


if __name__ == '__main__':
    lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
    lines = lines[lines['source'] == 'ted']  # Remove other sources
    # print(lines.head(20))
    lines.drop_duplicates(inplace=True)
    lines = lines.sample(n=25000, random_state=42)
    lines['english_sentence'] = lines['english_sentence'].apply(
        lambda x: preprocess(x))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(
        lambda x: preprocess(x, True))
    lines['length_eng_sentence'] = lines['english_sentence'].apply(
        lambda x: len(x.split(" ")))
    lines['length_hin_sentence'] = lines['hindi_sentence'].apply(
        lambda x: len(x.split(" ")))
    lines = lines[lines['length_eng_sentence'] <= 20]
    lines = lines[lines['length_hin_sentence'] <= 20]
    english_words  = get_vocab(lines['english_sentence'])
    hindi_words = get_vocab(lines['hindi_sentence'])
    word2idx_eng = token_idx(english_words)
    word2idx_hin = token_idx(hindi_words)
    input_tensor = [[word2idx_eng[s] for s in eng.split()] for eng in lines['english_sentence'].values.tolist()]
    target_tensor = [[word2idx_hin[s] for s in hin.split()] for hin in lines['hindi_sentence'].values.tolist()]
    input_tensor = [pad_sequences(x) for x in input_tensor]
    target_tensor = [pad_sequences(x) for x in target_tensor]
    # print(target_tensor[:10][2])
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2, random_state = 42)
    # print(len(input_tensor_train))
    train_dataset = Data(input_tensor_train, target_tensor_train)
    val_dataset = Data(input_tensor_val, target_tensor_val)
    dataset = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    print(next(iter(dataset)))