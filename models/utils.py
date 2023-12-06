import json, os
import math
import random
import re, csv
import pandas as pd
import torch
import time, sys, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init
from pynvml import *


CODE_START_TIME = time.strftime("%Y_%m_%d_day_%Hh_%Mm_%Ss", time.localtime())
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
# ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH='/data/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split'
ROUND_SIZE=5


torch.manual_seed(100)
random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda0 = torch.device('cuda:0')
if torch.cuda.device_count() >= 2:
    cuda1 = torch.device('cuda:1')
else:
    cuda1 = torch.device('cuda:0')


def get_recent_time():
    return time.strftime("%Y_%m_%d_day_%Hh_%Mm_%Ss", time.localtime())

def convert_list_to_dict(data_list):
    data_dict = {}
    for i in range(len(data_list)):
        song_name = data_list[i]['song_name']
        song_id = re.search('(.*?) song_title', song_name).group(1).strip()
        song_id = song_id.replace(": ", "_")
        data_dict[song_id] = data_list[i]
    return data_dict


def convert_dict_to_list(song_comments_detail_dict):
    return [song_comments_detail_dict[one_key] for one_key in song_comments_detail_dict]


def get_song_id(one_data):
    song_name = one_data['song_name']
    song_id = re.search('(.*?) song_title', song_name).group(1).strip()
    song_id = song_id.replace(": ", "_")
    return song_id

def get_song_id_given_song_name(song_name):
    song_id = re.search('(.*?) song_title', song_name).group(1).strip()
    song_id = song_id.replace(": ", "_")
    return song_id

def normalize_diversity_score(data_list):
    for i in range(len(data_list)):
        final_score_dict_sort = data_list[i]['final_score_dict_sort']
        diversity_score_list = [final_score_dict_sort[one_key][1]*final_score_dict_sort[one_key][2] for one_key in final_score_dict_sort]
        max_score = max(diversity_score_list)
        min_score = min(diversity_score_list)
        for one_key in final_score_dict_sort:
            diversity_score = final_score_dict_sort[one_key][1]*final_score_dict_sort[one_key][2]
            diversity_score_normalized = (diversity_score - min_score) / (max_score - min_score)
            if len(data_list[i]['final_score_dict_sort'][one_key]) >= 5:
                data_list[i]['final_score_dict_sort'][one_key][4] = diversity_score_normalized
            else:
                data_list[i]['final_score_dict_sort'][one_key].append(diversity_score_normalized)
                assert len(data_list[i]['final_score_dict_sort'][one_key]) >= 5, print(data_list[i]['final_score_dict_sort'][one_key])
            data_list[i]['final_score_dict_sort'][one_key][0] = final_score_dict_sort[one_key][3] + \
                                                                final_score_dict_sort[one_key][4]
        data_list[i]['final_score_dict_sort'] = {k: v for k, v in sorted(data_list[i]['final_score_dict_sort'].items(),
                                 key=lambda item: item[1][0],
                                 reverse=True)}
    return data_list

def normalize_tf_idf_score(data_list):
    idf_score = {}
    for i in range(len(data_list)):
        tf_freq = {}
        final_score_dict_sort = data_list[i]['final_score_dict_sort']
        total_word_num = 0
        for comment_id in data_list[i]['song_comments_detail_final']:
            for one_sent_id in data_list[i]['song_comments_detail_final'][comment_id]:
                if one_sent_id != 'likecnt_weight':
                    for one_word in data_list[i]['song_comments_detail_final'][comment_id][one_sent_id]['song_view_segmented']:
                        if one_word not in tf_freq:
                            tf_freq[one_word] = 1
                        else:
                            tf_freq[one_word] += 1
                        total_word_num += 1
        tf_idf_score_list = [(tf_freq[one_key] / float(total_word_num)) * final_score_dict_sort[one_key][2] for one_key in
                                final_score_dict_sort]
        max_score = max(tf_idf_score_list)
        min_score = min(tf_idf_score_list)
        for one_key in final_score_dict_sort:
            tf = tf_freq[one_key] / float(total_word_num)
            tf_idf_score = tf*final_score_dict_sort[one_key][2]
            tf_idf_score_normalized = (tf_idf_score - min_score) / (max_score - min_score)
            if len(data_list[i]['final_score_dict_sort'][one_key]) >= 6:
                data_list[i]['final_score_dict_sort'][one_key][5] = tf_idf_score_normalized
            else:
                data_list[i]['final_score_dict_sort'][one_key].append(tf_idf_score_normalized)
            #data_list[i]['final_score_dict_sort'][one_key].append(tf_idf_score_normalized)
            data_list[i]['final_score_dict_sort'][one_key][0] = final_score_dict_sort[one_key][3] + final_score_dict_sort[one_key][4]
        data_list[i]['final_score_dict_sort'] = {k: v for k, v in sorted(data_list[i]['final_score_dict_sort'].items(),
                                 key=lambda item: item[1][0],
                                 reverse=True)}
    return data_list

def get_all_words(data_segmented_list):
    total_doc_num = len(data_segmented_list)
    words_dict = {}
    for i in range(len(data_segmented_list)):
        print(str(i) + "/" + str(total_doc_num))
        song_name = data_segmented_list[i]["song_name"]
        comments_dict = data_segmented_list[i]['song_comments_detail_final']
        for song_id in comments_dict:
            for sent_id in comments_dict[song_id]:
                for one_word in comments_dict[song_id][sent_id]['song_view_segmented']:
                    if one_word not in words_dict:
                        words_dict[one_word] = 1
    labels_all_list = extract_labels(data_segmented_list)
    for one_label in labels_all_list:
        if one_label not in words_dict:
            words_dict[one_label] = 1
    words_list = list(words_dict.keys())
    return words_list

def extract_labels(train_data_segmented_list):
    labels_all_list = []
    for i in range(len(train_data_segmented_list)):
        labels_list = train_data_segmented_list[i]["song_labels"]
        for one_label in labels_list:
            if one_label not in labels_all_list:
                labels_all_list.append(one_label)
    return labels_all_list

def simplify_data_set(data_list):
    for i in range(len(data_list)):
        data_list[i].pop('song_comments_detail')
        data_list[i].pop('song_comments_detail_2')
    return data_list

def extract_annotation_content(file_path, song_id, one_dict):
    ref_labels = []
    not_labels = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        for row in spamreader:
            if count == 0:
                # skip first line (head)
                count += 1
                continue
            try:
                label, is_ref_label = row[0].split(",")
            except ValueError:
                count += 1
                continue
            if int(is_ref_label) == 1:
                ref_labels.append(label)
            else:
                not_labels.append(label)
            count += 1
    one_dict[song_id] = {"ref_labels": ref_labels, "not_labels": not_labels}
    return one_dict

def extract_annotation_content_excel(file_path, song_id, one_dict, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    ref_labels = []
    not_labels = []
    for i in range(len(df["候选词"])):
        if df["是否为这首歌的标签"][i] == 1:
            ref_labels.append(df["候选词"][i])
        else:
            not_labels.append(df["候选词"][i])
    one_dict[song_id] = {"ref_labels": ref_labels, "not_labels": not_labels}

def get_final_score_dict(file_name):
    data_list = json.load(open(file_name, "r", encoding='utf-8'))
    song_silver_label_details_dict = {}
    final_score_dict_sort_dict = {}
    for i in range(len(data_list)):
        song_id = get_song_id(data_list[i])
        if(len(data_list[i]["song_silver_labels"])) > 0:
            song_silver_label_details_dict[song_id] = data_list[i]['song_silver_label_details']
            final_score_dict_sort_dict[song_id] = data_list[i]["final_score_dict_sort"]
    return final_score_dict_sort_dict, song_silver_label_details_dict


def compute_label_cluster_distance_vocab(vocab, labels_all_list, words_list):
    device = cuda1
    label_cluster_distance_dict = {}
    label_cluster_distance_dict_detail = {}
    labels_tensor_list = []
    words_tensor_list = []
    [labels_tensor_list.append(vocab.word_embedding[vocab.word2index[one_label]] if one_label in vocab.word2index else vocab.word_embedding[vocab.word2index['<unk>']]) for one_label in labels_all_list]
    #labels_tensor = torch.stack(labels_tensor_list)
    [words_tensor_list.append(vocab.word_embedding[vocab.word2index[one_word]
    if one_word in vocab.word2index else vocab.word2index['<unk>']]) for one_word in words_list]
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    h = nvmlDeviceGetHandleByIndex(device.index)
    info = nvmlDeviceGetMemoryInfo(h)
    gpu_memory_free = info.free / 1024 / 1024
    standard_memory = 32506.75 # V100 32506.75 MB
    word_step = int(1000 * (gpu_memory_free / standard_memory))
    label_step = int(1000 * (gpu_memory_free / standard_memory))
    for i in range(0, len(words_tensor_list), word_step):
        word_start = i
        print("compute_label_cluster_distance_vocab processing words {} / {}".format(str(word_start), str(len(words_tensor_list))))
        word_end = i + word_step if i + word_step < len(words_tensor_list) else len(words_tensor_list)
        words_tensor = torch.stack(words_tensor_list[word_start:word_end])
        # words_tensor_tmp = words_tensor.view(1, words_tensor.size()[0], words_tensor.size()[1]).to(device=device)
        words_tensor = words_tensor.view(1, words_tensor.size()[0], words_tensor.size()[1]).to(device=device)
        final_output = []
        for label_i in range(0, len(labels_tensor_list), label_step):
            label_start = label_i
            label_end = label_i + label_step if label_i + label_step < len(labels_tensor_list) else len(labels_tensor_list)
            labels_tensor = torch.stack(labels_tensor_list[label_start:label_end])
            #labels_tensor_tmp = labels_tensor.view(
            labels_tensor_tmp = labels_tensor.view(
                labels_tensor.size()[0], 1, labels_tensor.size()[1]).repeat(1, word_end-word_start, 1).to(device=device)
            output = torch.transpose(cos(words_tensor, labels_tensor_tmp), 0, 1)
            final_output.append(output)
        output = torch.cat(final_output, dim=1)
        for j in range(word_end-word_start):
            distance_list = output[j].tolist()
            one_distance_dict = {}
            for kk in range(len(distance_list)):
                one_distance_dict[labels_all_list[kk]] = distance_list[kk]
            # one_distance_dict_sort = {k: v for k, v in sorted(one_distance_dict.items(),
            #                          key=lambda item: item[1],
            #                          reverse=False)}
            #label_cluster_distance_dict_detail[words_list[i + j]] = one_distance_dict_sort
            distance_list.sort(reverse=True)
            #label_cluster_distance_dict[words_list[i+j]] = distance_list
            label_cluster_distance_dict[words_list[i + j]] = distance_list[0]
    #json.dump(label_cluster_distance_dict, open("label_cluster_distance_dict", "w", encoding='utf-8'))
    return label_cluster_distance_dict, label_cluster_distance_dict_detail

# 获取negative sample
# 首先尝试所有negative sample都要依照条件而定
def get_negative_sample(data_list,random_ration=0):
    # todo 根据条件获取negative sample，valid score<0.001 ebc_based<0.0001是negative sample
    # retuen list ('song_id'|'candidate',negative rate(ebc_iter2))
    negative_candidates=[]
    negative_sample_ration=[]
    for data in data_list:
        song_id=get_song_id(data)
        score_dict=data['final_score_dict_sort']
        for key in score_dict:
            if score_dict[key][0]<0.001 or score_dict[key][1]<0.0001:
                negative_candidates.append("|".join([song_id,key,'negative']))
                negative_sample_ration.append(score_dict[key][-1])
    return negative_candidates,negative_sample_ration




# ebc -- random  diva --  probability
def compute_negative_sample(data_list, reset_negative_sample=False, sample_type="random"):
    data_dict = convert_list_to_dict(data_list)
    for i in range(len(data_list)):
        if reset_negative_sample:
            data_list[i]["negative_samples"] = {"doc": [], "label": []}
        song_golden_labels = data_list[i]['song_pseudo_golden_labels']
        if 'song_all_silver_labels' in data_list[i]:
            song_silver_labels = data_list[i]['song_all_silver_labels']
        else:
            song_silver_labels = []
        self_song_id = get_song_id(data_list[i])
        song_positive_labels = song_golden_labels + song_silver_labels
        random_lst=list(set(data_list[i]['final_score_dict_sort'].keys())-set(song_positive_labels))
        if sample_type == "probability":
            # P_negative_sample = log(1/diversity_score)*valid_score , diversity_score is normalized
            # inspired by TF-IDF = log(D/(1+d))*tf
            small_value = 0.0000000001
            final_score_list = [[one_word_tmp,
                                 data_list[i]['final_score_dict_sort'][one_word_tmp][4] * (
                                     math.log((1 + small_value)/ (data_list[i]['final_score_dict_sort'][one_word_tmp][5] + small_value)))]
                                for one_word_tmp in data_list[i]['final_score_dict_sort']]

            probability_score_list = []
            for j in range(len(final_score_list)):
                probability_score_list.append(final_score_list[j][1])
            max_probability = float(max(probability_score_list))
            min_probability = float(min(probability_score_list))

            final_score_list_normalized = [
                [one_word, (one_probability - min_probability) / (max_probability - min_probability)]
                for one_word, one_probability in final_score_list]
       
        elif sample_type == "random":
            final_score_list_normalized = [[one_word, random.random()] for one_word in random_lst]
        else:
            raise ValueError
        random.shuffle(final_score_list_normalized)
        diversity_negative_sample = []
        for one_word, one_probability in final_score_list_normalized:
            if one_word not in song_golden_labels:
                if sample_type == "probability":
                    if random.random() < one_probability:
                        diversity_negative_sample.append(one_word)
                        if len(diversity_negative_sample) + len(data_list[i]["negative_samples"]["doc"]) >= len(song_golden_labels)+len(song_silver_labels):
                            break
                elif sample_type == "random":
                    diversity_negative_sample.append(one_word)
                    if len(diversity_negative_sample)  >= len(
                            song_golden_labels)+len(song_silver_labels):
                        break
                else:
                    raise ValueError
        # if "negative_samples" not in data_dict[self_song_id]:
        data_dict[self_song_id]["negative_samples"] = {"doc": [], "label": []}
        data_dict[self_song_id]["negative_samples"]["label"] = diversity_negative_sample

    return convert_dict_to_list(data_dict)
    # return data_dict

def compute_negative_sample_for_silver_label(data_list):
    data_dict = convert_list_to_dict(data_list)
    doc_skip_count = 0
    for i in range(len(data_list)):
        self_song_id = get_song_id(data_list[i])
        if len(data_list[i]['song_silver_label_details']) == 1:
            song_silver_labels = data_list[i]['song_silver_label_details']['0']
        else:
            current_step = len(data_list[i]['song_silver_label_details'])
            song_silver_labels = list(set(data_list[i]['song_silver_label_details'][str(current_step-1)]) -
                                      set(data_list[i]['song_silver_label_details'][str(current_step-2)]))
        '''
        #only sample negative samples for new silver labels
        for one_silver_label in song_silver_labels:
            doc_skip = False
            if one_silver_label in label_cluster_distance_dict_detail:
                faraway_copper_label = list(label_cluster_distance_dict_detail[one_silver_label].keys())[0]
                if faraway_copper_label in copper_label_to_song_id_dict:
                    sample_length = len(copper_label_to_song_id_dict[faraway_copper_label])
                    random_song_id_list = random.sample(copper_label_to_song_id_dict[faraway_copper_label], sample_length)
                    for random_song_id in random_song_id_list:
                        if random_song_id in data_dict:
                            if "negative_samples" not in data_dict[random_song_id]:
                                data_dict[random_song_id]["negative_samples"] = {"doc":[], "label":[]}
                            if one_silver_label not in data_dict[random_song_id]['song_silver_labels']:
                                if one_silver_label not in data_dict[random_song_id]["negative_samples"]["doc"]:
                                    data_dict[random_song_id]["negative_samples"]["doc"].append(one_silver_label)
                                    break
                                else:
                                    doc_skip = True
                            else:
                                doc_skip = True
                        else:
                            doc_skip = True
            if doc_skip:
                doc_skip_count += 1
        '''
        final_score_dict_sort = {k: v for k, v in
         sorted(data_list[i]['final_score_dict_sort'].items(),
                key=lambda item: item[1][4],
                reverse=True)}
        diversity_negative_sample = []
        for one_word in final_score_dict_sort:
            if one_word not in song_silver_labels:
                if final_score_dict_sort[one_word][3] > 0.95:
                    diversity_negative_sample.append(one_word)
                    if len(diversity_negative_sample) >= len(data_list[i]["song_silver_labels"]):
                        break
        if "negative_samples" not in data_dict[self_song_id]:
            data_dict[self_song_id]["negative_samples"] = {"doc": [], "label": []}
        data_dict[self_song_id]["negative_samples"]["label"] += diversity_negative_sample
    print("doc skip count is : {}".format(str(doc_skip_count)))
    return data_dict


def get_copper_label_to_song_id_dict(data_list):
    data_dict = convert_list_to_dict(data_list)
    copper_label_to_song_id_dict = {}
    for one_song_id in data_dict:
        for one_copper_label in data_dict[one_song_id]["song_labels"]:
            if one_copper_label not in copper_label_to_song_id_dict:
                copper_label_to_song_id_dict[one_copper_label] = []
            copper_label_to_song_id_dict[one_copper_label].append(one_song_id)
    return copper_label_to_song_id_dict


def simplify_data_list(data_list):
    new_list = []
    for i in range(len(data_list)):
        one_line = {}
        for one_key in data_list[i]:
            if one_key != 'song_comments_detail_final':
                one_line[one_key] = data_list[i][one_key]
        new_list.append(one_line)
    return new_list


def get_unique_silver_labels_given_data_list(data_list, step):
    silver_labels_all = []
    golden_labels_all = []

    for i in range(len(data_list)):
        golden_labels_all += data_list[i]['song_pseudo_golden_labels']

    for i in range(len(data_list)):
        if step == 0:
            silver_labels_all = []
        elif step == 1:
            if "song_silver_label_details" in data_list[i]:
                silver_labels_all += data_list[i]['song_silver_label_details'][str(step-1)]
        else:
            silver_labels_all += list(set(data_list[i]['song_silver_label_details'][str(step-1)]) -
                                      set(data_list[i]['song_silver_label_details'][str(step-2)]))
    unique_silver_labels = list(set(silver_labels_all) - set(golden_labels_all))
    return unique_silver_labels, list(set(golden_labels_all))

def sort_according_ref_labels(data_list):
    results_list = []
    new_data_list = []
    for i in range(len(data_list)):
        song_name = data_list[i]["song_name"]
        song_index = i
        ref_labels = data_list[i]["annotation_dict"]["ref_labels"]
        ref_labels_length = len(ref_labels)
        results_list.append([ref_labels_length, song_index, song_name, ref_labels])
    results_list.sort(key=lambda tup: tup[0], reverse=True)
    for i in range(len(results_list)):
        new_data_list.append(data_list[results_list[i][1]])
    return results_list, new_data_list

