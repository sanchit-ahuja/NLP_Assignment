import pickle
from typing import Dict

import pandas as pd
from torch.utils import data
from torch.utils.data import dataloader

ENGLISH_WORD_TEST_CASES = ['EOS', 'SOS', 'a', 'aaaahhh', 'aaron', 'ab', 'abacha', 'abandoned', 'abc', 'abducting', 'abdul', 'abe', 'abhishek', 'abilities', 'ability', 'abject', 'able', 'abnormality', 'abode', 'abolished',
                           'abololo', 'about', 'about”', 'above', 'abraham', 'abrahams', 'abroad', 'absence', 'absent', 'absolute', 'absolutely', 'absoluteness', 'absolution', 'absorb', 'absorbing', 'abstract', 'absurd', 'abu', 'abundance', 'abundant']

HINDI_SENTENCES = ['हम अभी तक नहीं जानते हैं कि उसके माता-पिता कौन हैं, वह कौन है,', 'कोई कुंजीपटल नहीं,', 'लेकिन एक कलाकार होने के साथ', 'और यह खास गुब्बारा,', 'और जितना आपको लगता है, यह उतना कठिन नहीं है.अपने सभी नवाचारों में जलवायु समाधान को एकीकृत करें,', 'और जेफ़ हान द्वारा प्रदर्शन देखा होगा',
                   'यह बच्चा पूरी तरह से बिजली से चलता है.', 'फिर भी बच्चों को नियम बनाने का बिलकुल नहीं, या बहुत कम, मौका मिलता है', 'मै अपने द्वारा दिए गए उम्र के बढ़्ने के विवर्ण की कुछ व्याख्या करूंगा.', 'फैलने लगता है, फिर ये ठंडा होकर उस अवस्था तक पहुँच जाता है, जब ये पारदर्शी बन जाता है,']

ENGLISH_SENTENCES = ["We still don't know who her parents are, who she is.", 'no keyboard,', 'But as far as being a performer,', 'And this particular balloon,', "and it's not as hard as you think. Integrate climate solutions into all of your innovations,",
                     'and saw the demo by Jeff Han', 'This baby is fully electric.', 'kids have no, or very little, say in making the rules,', "I'm going to add a little bit to my description of aging.", 'expands and cools until it gets to the point where it becomes transparent,']

WORD_IDX_DICT_TEST_CASES = {'SOS': 1, 'EOS': 2, 'PAD': 0, 'a': 3, 'aaaahhh': 4, 'aaron': 5, 'ab': 6, 'abacha': 7, 'abandoned': 8, 'abc': 9,
                            'abducting': 10, 'abdul': 11, 'abe': 12, 'abhishek': 13, 'abilities': 14, 'ability': 15, 'abject': 16, 'able': 17, 'abnormality': 18, 'abode': 19}


def testing_dict(test_dict: Dict, correct_dict: Dict, token_idx=True):
    test_dict = sorted(test_dict.items(),
                       key=lambda item: item[1] if token_idx else item[0])
    correct_dict = sorted(correct_dict.items(),
                          key=lambda item: item[1] if token_idx else item[0])
    return test_dict == correct_dict


def load_dict(filename):
    with open(filename, 'rb') as f:
        ret_dic = pickle.load(f)
    return ret_dic


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def test_token_index():
    from preprocess import token_idx
    test_dic_token_idx = token_idx(ENGLISH_WORD_TEST_CASES)
    actual_dic_token_idx = load_dict('word_idx_token.pkl')
    assert testing_dict(test_dic_token_idx, actual_dic_token_idx) == True


def test_index_token():
    from preprocess import idx_token
    test_dic_word_idx = idx_token(WORD_IDX_DICT_TEST_CASES)
    actual_dic_word_idx = load_dict('idx_word_token.pkl')
    # print(test_dic_word_idx)
    assert testing_dict(test_dic_word_idx, actual_dic_word_idx, False)


def test_preprocess():
    from preprocess import preprocess
    correct_sent_eng = load_dict('correct_preprocess_eng.pkl')
    correct_sent_hindi = load_dict('correct_preprocess_hindi.pkl')
    correct_sent_hindi_unide = load_dict('unide_correct_preprocess_hindi.pkl')
    test_sent_eng = []
    test_sent_hindi = []

    for sent in ENGLISH_SENTENCES:
        test_sent_eng.append(preprocess(sent))

    for sent in HINDI_SENTENCES:
        test_sent_hindi.append(preprocess(sent, True))

    test_sent_eng = sorted(test_sent_eng)
    test_sent_hindi = sorted(test_sent_hindi)
    assert test_sent_eng == correct_sent_eng
    assert test_sent_hindi == correct_sent_hindi or test_sent_hindi == correct_sent_hindi_unide


def test_get_vocab():
    sent = pd.read_pickle('pd_lines_eng.pkl')
    from preprocess import get_vocab
    vocab = get_vocab(sent)
    vocab = sorted(list(vocab))
    vocab_correct = load_dict('vocab_file.pkl')
    vocab_correct = sorted(list(vocab_correct))
    assert vocab == vocab_correct


# def test_dataset_object():
#     from preprocess import Data
#     input_tensor_train = load_dict('input_tensor_train.pkl')
#     target_tensor_train = load_dict('target_tensor_train.pkl')
#     dataset = Data(input_tensor_train, target_tensor_train)
#     item_10_correct = load_dict('item_10.pkl')
#     item_21_correct = load_dict('item_21.pkl')
#     item_45_correct = load_dict('item_45.pkl')
#     item_100_correct = load_dict('item_100.pkl')

#     assert (dataset.__getitem__(10)[0] == item_10_correct[0]).all()
#     assert (dataset.__getitem__(21)[0] == item_21_correct[0]).all()
#     assert (dataset.__getitem__(45)[0] == item_45_correct[0]).all()
#     assert (dataset.__getitem__(100)[0] == item_100_correct[0]).all()

#     assert (dataset.__getitem__(10)[1] == item_10_correct[1]).all()
#     assert (dataset.__getitem__(21)[1] == item_21_correct[1]).all()
#     assert (dataset.__getitem__(45)[1] == item_45_correct[1]).all()
#     assert (dataset.__getitem__(100)[1] == item_100_correct[1]).all()



def test_get_dataset_func():
    from preprocess import get_dataset
    dataset,_,_ = get_dataset(batch_size=32, shuffle=False, num_workers=0)


    item_10_correct = load_dict('item_10.pkl')
    item_4_correct = load_dict('item_4.pkl')
    item_15_correct = load_dict('item_15.pkl')
    item_1_correct = load_dict('item_1.pkl')

    assert (dataset.dataset.__getitem__(10)[0] == item_10_correct[0]).all()
    assert (dataset.dataset.__getitem__(4)[0] == item_4_correct[0]).all()
    assert (dataset.dataset.__getitem__(15)[0] == item_15_correct[0]).all()
    assert (dataset.dataset.__getitem__(1)[0] == item_1_correct[0]).all()

    assert (dataset.dataset.__getitem__(10)[1] == item_10_correct[1]).all()
    assert (dataset.dataset.__getitem__(4)[1] == item_4_correct[1]).all()
    assert (dataset.dataset.__getitem__(15)[1] == item_15_correct[1]).all()
    assert (dataset.dataset.__getitem__(1)[1] == item_1_correct[1]).all()
    

if __name__ == "__main__":
    test_get_vocab()
    test_index_token()
    test_token_index()
    test_get_dataset_func()
    # from preprocess import get_dataset
    # a,b,c = get_dataset(batch_size = 16,shuffle=False, num_workers=0)
    # print(a.dataset.__getitem__(12)[0])
    # print(dataloader.items())
    # print(dir(dataloader))

