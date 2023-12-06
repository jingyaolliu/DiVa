# utils for data preprocess
import os
import json
import argparse
import re
from collections import Counter
import math
import numpy as np

def save_json(file_path, data):
    """
    Saves data to a JSON file.

    Args:
        file_path (str): The file path to save the data to.
        data (any): The data to be saved.

    Returns:
        None
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def remove_special_tokens_from_source_raw_text(data_dict, use_source):
    """
    Removes special characters from the tokenized text and generates a new list of text without special characters. 
    Stores the new list in a new key 'candidates' in each data item.

    Args:
        data_dict (dict): A dictionary of data items.
        use_source (str): The key in each data item to retrieve the tokenized text.

    Returns:
        data_dict (dict): The input data dictionary with the new list of text without special characters added.
    """
    # Loop through each data item in the dictionary
    for data_item_key in data_dict.keys():
        tokenized_words = data_dict[data_item_key][use_source]
        candidates = []
        for one_word in tokenized_words:
            match_result = re.match("^[\u4e00-\u9fa5a-zA-Z0-9]+$", one_word)
            if match_result is not None:
                candidates.append(one_word)
        data_dict[data_item_key]['candidates'] = candidates
    return data_dict

def re_extract_candidate_words(data_dict, stop_words, word_counts_dict, use_source, col_name='final_score_dict_sort'):
    """
    Extracts candidate words from the tokenized text in each data item, and stores them in a new key 'col_name' in each data item.

    Args:
        data_dict (dict): A dictionary of data items.
        stop_words (dict): A dictionary of stop words.
        word_counts_dict (dict): A dictionary of word counts.
        use_source (str): The key in each data item to retrieve the tokenized text.
        col_name (str): The key in each data item to store the extracted candidate words.

    Returns:
        data_dict (dict): The input data dictionary with the extracted candidate words added.
    """
    stop_words_set = set(list(stop_words.keys()))
    for data_item_key in data_dict.keys():
        candidate_words = []
        for one_word in data_dict[data_item_key]['candidates']:
            # If the word appears more than 5 times and its length is greater than 1, consider it as a candidate word.
            if word_counts_dict[one_word] > 5 and len(one_word) > 1:
                candidate_words.append(one_word)
        # Remove stop words from the candidate word list
        candidate_words = list(set(candidate_words) - stop_words_set)
        data_dict[data_item_key]['candidates']=candidate_words
        data_dict[data_item_key]['distinct_candidats']=list(set(candidate_words))
        data_dict[data_item_key][col_name] = {}
        for one_candidate_word in data_dict[data_item_key]['distinct_candidats']:
            data_dict[data_item_key][col_name][one_candidate_word] = [0] * 15
    # Return the updated data dictionary
    return data_dict

def cal_df(data_lst, col_name=''):
    """
    Calculates the document frequency (df) of each word in the data.

    Args:
        data_lst (list): A list of data dictionaries.
        col_name (str): The key in each data dictionary where the candidate words are stored.

    Returns:
        df_dict (dict): A dictionary of document frequencies.
    """
    df_info = []
    for data_dict in data_lst:
        for data_item in data_dict.values():
            df_info += list(set(data_item[col_name]))
    # Count the document frequency for each word in the list of all candidate words
    df_dict = Counter(df_info)
    return dict(df_dict)

def cal_tf(w_lst, candidate_w_lst):
    """
    Calculates the term frequency (tf) of each candidate word in the comments.

    Args:
        w_lst (list): A list of segmented comments with special characters and stop words removed.
        candidate_w_lst (list): A list of candidate words.

    Returns:
        tf_score (dict): A dictionary of term frequencies for each candidate word.
        sum (int): The sum of all word counts in the comments.
    """
    tf_score = {}
    # Calculate the term frequency (tf) for each candidate word
    for candidate in candidate_w_lst:
        tmp_w_count = w_lst.count(candidate)
        tf_score[candidate] = tmp_w_count
    # Get the sum of all word counts in the comments
    tf_count = len(w_lst)
    return tf_score, tf_count

import math

def w_freq_based_baseline(data_dict, df, N, stop_words, col_name):
    """
    Calculates the tf-idf score for each candidate word in the data.

    Args:
        data_dict (dict): A dictionary of data items.
        df (dict): A dictionary of document frequencies.
        N (int): The number of documents.
        stop_words (dict): A dictionary of stop words.
        col_name (str): The key in each data item where the candidate words are stored.

    Returns:
        data_dict (dict): The updated data dictionary with the tf-idf scores added.
    """
    for data_item_key in data_dict:
        data_item = data_dict[data_item_key]
        # Get the list of all candidate words and the list of distinct candidate words
        candidate_lst = data_item['candidates']
        distinct_candidate_lst = data_item['distinct_candidates']
        # Calculate the term frequency (tf) for each distinct candidate word
        tf, tf_count = cal_tf(candidate_lst, distinct_candidate_lst)
        # Calculate the tf-idf score for each distinct candidate word
        for candidate in distinct_candidate_lst:
            df_val = df[candidate]
            tn_val = tf[candidate]
            tf_val = tn_val / tf_count
            idf_val = math.log(N / (df_val + 1))
            # Assign the tf-idf score to the candidate word in the data dictionary
            data_dict[data_item_key]['final_score_dict_sort'][candidate][0] = tn_val
            data_dict[data_item_key]['final_score_dict_sort'][candidate][1] = tf_val
            data_dict[data_item_key]['final_score_dict_sort'][candidate][2] = df_val
            data_dict[data_item_key]['final_score_dict_sort'][candidate][10] = tf_val * idf_val 
    return data_dict
            
        
        
    