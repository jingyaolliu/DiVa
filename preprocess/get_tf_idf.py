# calculate tf-idf for each tokenized words .
import os
from utils import *

def init_score_sort_dict(data_dict, use_source='final_score_dict_sort'):
    """
    Initializes a sorting dictionary for the given data dictionary.

    Args:
        data_dict (dict): A dictionary containing data to be sorted.
        use_source (str): The key in each data item to store the sorting dictionary (default is 'final_score_dict_sort').

    Returns:
        data_dict (dict): The input data dictionary with each data item containing an empty sorting dictionary.
    """
    for data_item in data_dict.values():
        data_item[use_source] = {}
    return data_dict

def main():
    # Set the source for segmented text
    segmented_text_source = 'tokenized_words'

    # Load stop words
    data_dir = "../data"
    stopwords_dir = "stopwords"
    hit_stop_words = open(os.path.join(data_dir, stopwords_dir, "hit_stopwords.txt"), "r", encoding='utf-8').readlines()
    cn_stop_words = open(os.path.join(data_dir, stopwords_dir, "cn_stopwords.txt"), "r", encoding='utf-8').readlines()
    baidu_stop_words = open(os.path.join(data_dir, stopwords_dir, "baidu_stopwords.txt"), "r", encoding='utf-8').readlines()
    all_stop_words = hit_stop_words + cn_stop_words + baidu_stop_words

    # Load song information files
    train_dict_origin = json.load(open(os.path.join(data_dir, "train.json"), "r", encoding='utf-8'))
    test_dict_origin = json.load(open(os.path.join(data_dir, "test1.json"), "r", encoding='utf-8'))

    # Initialize final_score_dict_sort in the JSON files
    test_dict = init_score_sort_dict(test_dict_origin)
    train_dict = init_score_sort_dict(train_dict_origin)
    
    N = len(train_dict) + len(test_dict)

    # Load word count dictionary
    word_counts_dict = json.load(open(os.path.join(data_dir, "word_count.json"), "r", encoding='utf-8'))
    
    stop_words = {}
    use_hit_stop_words = True
    use_cn_stop_words = True
    use_baidu_stop_words = True
    if use_hit_stop_words and use_cn_stop_words and use_baidu_stop_words:
        print("Using all stop words.")
        # Merge the stop words from different sources into one dictionary
        for one_word in all_stop_words:
            one_word_tmp = one_word.strip()
            if one_word_tmp not in stop_words:
                stop_words[one_word_tmp] = 1
    print("Stop words generation finished.")

    print("Cleaning source raw Chinese text.")
    # Clean the raw text data in segmented_text_source by removing special tokens
    train_dict = remove_special_tokens_from_source_raw_text(train_dict, use_source=segmented_text_source)
    test_dict = remove_special_tokens_from_source_raw_text(test_dict, use_source=segmented_text_source)

    print("Extracting candidate words from segmented text.")
    # Extract candidate words from segmented_text_source and store them in a new key in each data item
    train_dict = re_extract_candidate_words(train_dict, stop_words, word_counts_dict, use_source=segmented_text_source)
    test_dict = re_extract_candidate_words(test_dict, stop_words, word_counts_dict, use_source=segmented_text_source)
    
    df_dict = cal_df([train_dict, test_dict], 'candidates')
    
    print("Calculating tf-idf scores.")
    # Calculate the tf-idf scores for each candidate word in the train and test data
    train_dict = w_freq_based_baseline(train_dict, df_dict, N, stop_words, segmented_text_source)
    test_dict = w_freq_based_baseline(test_dict, df_dict, N, stop_words, segmented_text_source)
    
    # Update train_dict and test_dict to files
    with open(os.path.join(data_dir, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train_dict, f, ensure_ascii=False)
    with open(os.path.join(data_dir, "test1.json"), "w", encoding='utf-8') as f:
        json.dump(test_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    main()