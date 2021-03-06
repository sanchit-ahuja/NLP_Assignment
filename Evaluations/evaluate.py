import os
import pickle
import sys
from collections import OrderedDict
from typing import Dict

import pandas as pd
from numpy.lib.npyio import save

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
ENGLISH_WORD_TEST_CASES = ['EOS', 'SOS', 'a', 'aaaahhh', 'aaron', 'ab', 'abacha', 'abandoned', 'abc', 'abducting', 'abdul', 'abe', 'abhishek', 'abilities', 'ability', 'abject', 'able', 'abnormality', 'abode', 'abolished', 'abololo', 'about', 'about”', 'above', 'abraham', 'abrahams', 'abroad', 'absence', 'absent', 'absolute', 'absolutely', 'absoluteness', 'absolution', 'absorb', 'absorbing', 'abstract', 'absurd', 'abu', 'abundance', 'abundant']

WORD_IDX_DICT_TEST_CASES = {'SOS': 1, 'EOS': 2, 'PAD': 0, 'a': 3, 'aaaahhh': 4, 'aaron': 5, 'ab': 6, 'abacha': 7, 'abandoned': 8, 'abc': 9, 'abducting': 10, 'abdul': 11, 'abe': 12, 'abhishek': 13, 'abilities': 14, 'ability': 15, 'abject': 16, 'able': 17, 'abnormality': 18, 'abode': 19}

def testing_dict(test_dict: Dict, correct_dict: Dict):
    test_dict = OrderedDict((test_dict.items()))
    correct_dict = OrderedDict((correct_dict.items()))
    return test_dict == correct_dict

def load_dict(filename):
    with open(filename,'rb') as f:
        ret_dic = pickle.load(f)
    return ret_dic

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def test_token_index():
    from preprocess import token_idx
    test_dic_token_idx = token_idx(ENGLISH_WORD_TEST_CASES)
    actual_dic_token_idx = load_dict('word_idx_token.pkl')
    assert testing_dict(test_dic_token_idx,actual_dic_token_idx) == True

def test_index_token():
    from preprocess import idx_token
    test_dic_word_idx = idx_token(WORD_IDX_DICT_TEST_CASES)
    actual_dic_word_idx = load_dict('idx_word_token.pkl')
    assert testing_dict(test_dic_word_idx,actual_dic_word_idx)

def test_convert_to_tensor():
    from preprocess import convert_to_tensor
    sent = pd.read_pickle('pd_lines_eng.pkl')
    word_idx = load_dict('all_words_token_idx.pkl')
    tens = convert_to_tensor(word_idx,sent)
    tens_correct = load_dict('tensor_val_correct.pkl')
    for arr, arr1 in zip(tens, tens_correct):
        assert arr.all() == arr1.all()

def test_dataset_object():
    from preprocess import Data
    input_tensor_train = load_dict('input_tensor_train.pkl')
    target_tensor_train = load_dict('target_tensor_train.pkl')
    dataset = Data(input_tensor_train, target_tensor_train)
    item_10_correct = load_dict('item_10.pkl')
    item_21_correct = load_dict('item_21.pkl')
    item_45_correct = load_dict('item_45.pkl')
    item_100_correct = load_dict('item_100.pkl')

    assert (dataset.__getitem__(10)[0] == item_10_correct[0]).all()
    assert (dataset.__getitem__(21)[0] == item_21_correct[0]).all()
    assert (dataset.__getitem__(45)[0] == item_45_correct[0]).all()
    assert (dataset.__getitem__(100)[0] == item_100_correct[0]).all()

    assert (dataset.__getitem__(10)[1] == item_10_correct[1]).all()
    assert (dataset.__getitem__(21)[1] == item_21_correct[1]).all()
    assert (dataset.__getitem__(45)[1] == item_45_correct[1]).all()
    assert (dataset.__getitem__(100)[1] == item_100_correct[1]).all()


if __name__ == "__main__":
    test_dataset_object()