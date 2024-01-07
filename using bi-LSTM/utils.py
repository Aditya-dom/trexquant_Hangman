import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch


def get_char_mapping():
    char_mapping = {'_': 27}
    ct = 1
    for i in list(string.ascii_lowercase):
        char_mapping[i] = ct
        ct = ct + 1
    return char_mapping

def create_intermediate_data(df):
    x = pd.DataFrame(df.split('\n'))
    x[1] = x[0].apply(lambda p: len(p))
    x['vowels_present'] = x[0].apply(lambda p: set(p).intersection({'a', 'e', 'i', 'o', 'u'}))
    x['vowels_count'] = x['vowels_present'].apply(lambda p: len(p))
    x['unique_char_count'] = x[0].apply(lambda p: len(set(p)))
    x_ = x[~((x['unique_char_count'].isin([0, 1, 2])) | (x[1] <= 3)) & (x.vowels_count != 0)]

def read_data():
    with open("./Data/words_250000_train.txt", "r") as f:
        df = f.read()
    return df

def loop_for_permutation(unique_letters, word, all_perm, i):
    random_letters = random.sample(unique_letters, i+1)
    new_permuted_word = word
    for letter in random_letters:
        new_permuted_word = new_permuted_word.replace(letter, "_")
        all_perm.append(new_permuted_word)

def permute_all(word, vowel_permutation_loop=False):
    unique_letters = list(set(word))
    all_perm = []
    if vowel_permutation_loop:
        for i in range(len(unique_letters) - 1):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm
    else:
        for i in range(len(unique_letters) - 2):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm

def permute_consonents(word):
    len_word = len(word)
    vowel_word = "".join([i if i in ["a", "e", "i", "o", "u"] else "_" for i in list(word)])
    vowel_idxs = []
    for i in range(len(vowel_word)):
        if vowel_word[i] == "_":
            continue
        else:
            vowel_idxs.append(i)  
    abridged_vowel_word = vowel_word.replace("_", "")
    all_permute_consonents = permute_all(abridged_vowel_word, vowel_permutation_loop=True)
    permuted_consonents = []
    for permuted_word in all_permute_consonents:
        a = ["_"] * len(word)
        vowel_no = 0
        for vowel in permuted_word:
            a[vowel_idxs[vowel_no]] = vowel
            vowel_no += 1
        permuted_consonents.append("".join(a))
    return permuted_consonents

def create_masked_dictionary(df_aug):
    masked_dictionary = {}
    counter = 0
    for word in df_aug[0]:
        all_masked_words_for_word = []
        all_masked_words_for_word = all_masked_words_for_word + permute_all(word)
        all_masked_words_for_word = all_masked_words_for_word +  permute_consonents(word)
        all_masked_words_for_word = list(set(all_masked_words_for_word))
        masked_dictionary[word] = all_masked_words_for_word
        if counter % 10000 == 0:
            print(f"Iteration {counter} completed")
        counter = counter + 1

def get_vowel_prob(df_vowel, vowel):
    try:
        return df_vowel[0].apply(lambda p: vowel in p).value_counts(normalize=True).loc[True]
    except:
        return 0

def get_vowel_prior(df_aug):
    prior_json = {}
    for word_len in range(df_aug[1].max()):
        prior_json[word_len + 1] = []
        df_vowel = df_aug[df_aug[1] == word_len]
        for vowel in ['a', 'e', 'i', 'o', 'u']:
            prior_json[word_len + 1].append(get_vowel_prob(df_vowel, vowel))
        prior_json[word_len + 1] = pd.DataFrame([pd.Series(['a', 'e', 'i', 'o', 'u']), pd.Series(prior_json[word_len + 1])]).T.sort_values(by=1, ascending=False)
    return prior_json    

def save_vowel_prior(vowel_prior):
    pickle.dump(vowel_prior, open("prior_probabilities.pkl", "wb"))    

def encode_output(word):
    char_mapping = get_char_mapping()
    output_vector = [0] * 26
    for letter in word:
        output_vector[char_mapping[letter] - 1] = 1
#     return torch.tensor([output_vector])
    return output_vector

def encode_input(word):
    char_mapping = get_char_mapping()
    given_word_len = len(word)
    embedding_len = 35
    word_vector = [0] * embedding_len
    ct = 0
    for letter_no in range(embedding_len - given_word_len, embedding_len):
        word_vector[letter_no] = char_mapping[word[ct]]
        ct += 1
    return word_vector

def encode_words(masked_dictionary): 
    target_data = []
    input_data = []
    counter = 0
    for output_word, input_words in masked_dictionary.items():
        output_vector = encode_output(output_word)
        for input_word in input_words:
            target_data.append(output_vector)
            input_data.append(encode_input(input_word))
        if counter % 10000 == 0:
            print(f"Iteration {counter} completed")
        counter += 1
    return input_data, target_data

def save_input_output_data(input_data, target_data):
    with open(r'input_features.txt', 'w') as fp:
        for item in input_data:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    with open(r'target_features.txt', 'w') as fp:
        for item in target_data:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

def convert_to_tensor(input_data, target_data):
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    return input_tensor, target_tensor

def get_datasets():
    df = read_data()
    x_ = create_intermediate_data(df)
    df_aug = x_.copy()
    masked_dictionary = create_masked_dictionary(df_aug)
    vowel_prior = get_vowel_prior(df_aug)
    save_vowel_prior(vowel_prior)
    input_data, target_data = encode_words(masked_dictionary)
    save_input_output_data(input_data, target_data)
    input_tensor, target_tensor = convert_to_tensor(input_data, target_data)
    



