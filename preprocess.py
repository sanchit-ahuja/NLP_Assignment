from typing import List, Dict
import unidecode
import string
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import configparser
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read('dev.config')
# config.read(os.path.join(dir_path,"dev.config"))
config=config["values"]
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
    sentence = unidecode.unidecode(sentence)
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
    word2idx = {'SOS': 1, 'EOS': 2, 'PAD': 0}
    counter = 3
    for word in words:
        if word not in word2idx:
            word2idx[word] = counter
            counter += 1
    return word2idx


def idx_token(wor2idx: Dict) -> Dict:
    idx2word = {1: 'SOS', 2: 'EOS', 0: 'PAD'}
    for word, idx in wor2idx.items():
        if idx not in idx2word:
            idx2word[idx] = word

    return idx2word


def pad_sequences(sent_tensor: List[List[int]], max_len: int=20) -> np.ndarray:
    padded = np.zeros((max_len), dtype=np.int64)
    if len(sent_tensor) > max_len:
        padded[:] = sent_tensor[:max_len]
    else:
        padded[:len(sent_tensor)] = sent_tensor
    return padded


def convert_to_tensor(word2idx: Dict, sentences: pd.Series):
    tensor = [[word2idx[s] for s in eng.split()]
              for eng in sentences.values.tolist()]
    tensor = [pad_sequences(x) for x in tensor]
    return tensor


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

def save_dict(di_, filename_) -> None:
    import pickle
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def get_dataset(batch_size=2, types="train", shuffle=True, num_workers=1, pin_memory=False, drop_last=True):
    lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
    lines = lines[lines['source'] == 'ted']  # Remove other sources
    # print(lines.head(20))
    lines.drop_duplicates(inplace=True)
    lines = lines.sample(n=int(config["samples"]), random_state=42)
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

    english_words = get_vocab(lines['english_sentence'])
    hindi_words = get_vocab(lines['hindi_sentence'])
    word2idx_eng = token_idx(english_words)
    word2idx_hin = token_idx(hindi_words)
    idx2word_eng = idx_token(word2idx_eng)
    idx2word_hin = idx_token(word2idx_hin)

    input_tensor = convert_to_tensor(word2idx_eng, lines['english_sentence'])
    target_tensor = convert_to_tensor(word2idx_hin, lines['hindi_sentence'])

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42)

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
