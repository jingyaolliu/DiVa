import os, copy
import statistics

import torch
import json, time, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init

import utils
from data_loader import CustomDatasetBinary, CustomDatasetBinaryUpdateMatchScore
from os.path import join
import logging, random
import numpy as np
from arguments import args

import sys
sys.path.append("..")

from Evaluate.update_silver import update_silver_iter,update_silver_simple


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import Vocab, OS_NAME, LOGIN_NAME
from utils import *

from evaluation_util import get_threshold_from_validation_set, extract_annotation_set
from evaluate_iterations import evaluate_annotation_performance_top_k

from transformers import AutoModel, AutoTokenizer
from loss_func import *

best_epoch_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/DIVA-Large-Probability-Negative-Sample-Log_rf-Self_training/model_train_1000_use_layer_0_last_cat_2_run_fasttext_global_word_embedding_0_95_average_all_random_cat_2_0_001_Adam_300_15_20000_long_sent_2022_11_18_day_22h_07m_46s/train_1000_use_layer_0_last_cat_2_run_fasttext_global_word_embedding_0_95_average_all_random_cat_2_0_001_Adam_300_15_20000_long_sent_model_binary_t0_146'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

setup_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationModelBinaryXLNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(TextClassificationModelBinaryXLNet, self).__init__()
        self.model_name = "hfl/chinese-xlnet-base"
        classification_layer_input_dim = 768
        fasttext_dim = 300
        self.fc0 = nn.Linear(classification_layer_input_dim * 2, classification_layer_input_dim).to(device=device)
        self.fc1 = nn.Linear(classification_layer_input_dim * 2, classification_layer_input_dim).to(device=device)
        self.fc2 = nn.Linear(classification_layer_input_dim * 2, int(classification_layer_input_dim/2)).to(device=device)
        self.dropout=nn.Dropout(p=dropout)
        if args.model_type == "cat_2_large":
            self.fc3 = nn.Linear(classification_layer_input_dim * 3, 1).to(device=device)
        else:
            self.fc3 = nn.Linear(int(classification_layer_input_dim/2), 1).to(device=device)

        self.activation=nn.ReLU()

        self.m = nn.Sigmoid()

    def forward(self, comment_vector, candidate_words_vectors):

        comment_vector_dense_layer1 = self.activation(self.fc0(comment_vector))
        candidate_words_vectors_dense_layer1 = self.activation(self.fc1(candidate_words_vectors))
        #todo normalize after concat
        #todo concat , normalize
        dense_layer2 = self.activation(self.fc2(torch.cat([comment_vector_dense_layer1, candidate_words_vectors_dense_layer1], dim=1)))
        outputs_binary = self.fc3(dense_layer2)
        out_sigmoid = self.m(outputs_binary)
        return out_sigmoid

class TextClassificationModelBinaryGRU(nn.Module):
    def __init__(self, embed_dim=300):
        super(TextClassificationModelBinaryGRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding).float().to(device=device)
        self.lstm_sent = nn.GRU(embed_dim, embed_dim, batch_first=True).to(device)
        self.fc0 = nn.Linear(embed_dim * 2, 1).to(device=device)
        self.gelu = nn.GELU()
        self.m = nn.Sigmoid()

    def forward(self, comment_vector, candidate_words_vectors, text_len):
        packed_input = pack_padded_sequence(comment_vector, text_len.to(device), batch_first=True, enforce_sorted=False)
        sent_packed_output, (hn, cn) = self.lstm_sent(packed_input)
        sent_output, sent_len = pad_packed_sequence(sent_packed_output, batch_first=True)
        sent_out_final = torch.stack([sent_output[i][sent_len[i] - 1] for i in range(sent_output.size()[0])])

        comment_vector_dense_layer1 = self.gelu(self.fc0(comment_vector))
        candidate_words_vectors_dense_layer1 = self.gelu(self.fc1(candidate_words_vectors))
        dense_layer2 = self.gelu(self.fc2(torch.cat([comment_vector_dense_layer1, candidate_words_vectors_dense_layer1], dim=1)))
        outputs_binary = self.fc3(dense_layer2)
        out_sigmoid = self.m(outputs_binary)
        return out_sigmoid
# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, reset_optimizer=False):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    logging.info(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    if not reset_optimizer:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']

def get_model_prediction(model, comment_vector_list, ref_labels_vector_list, type="cat_2"):
    comment_vector_batch = torch.stack(comment_vector_list).to(device)
    ref_labels_vector_batch = torch.stack(ref_labels_vector_list).to(device)  # cat
    predicted_label = model(comment_vector_batch, ref_labels_vector_batch)
    return predicted_label#, song_ref_scores_list_batch, song_ref_weight_list_batch


def train(model, data_loader, epoch, optimizer, criterion, index2label, batch_size, encode_type):
    model.train()
    update_count = 0
    log_interval = 1
    batch_count = 0
    gpu_train_size = 400000 / 2 
    epoch_loss = []
    epoch_acc = []
    batch_acc = []
    batch_loss = []
    comment_vector_list = []
    ref_labels_vector_list = []
    song_ref_scores_list = []
    song_ref_weight_list = []
    for idx, (song_comments_vector, ref_labels_vector, ref_sample_type, song_id) in enumerate(data_loader):
        optimizer.zero_grad()
        comment_vector_list.append(song_comments_vector)
        ref_labels_vector_list.append(ref_labels_vector)
        if ref_sample_type == "golden":
            song_ref_scores = 1
            song_ref_weight = 1.0
        elif ref_sample_type == "negative":
            song_ref_scores = 0
            song_ref_weight = 1.0
        else: # silver
            song_ref_scores = 1
            song_ref_weight = 1.0

        song_ref_scores_list.append(song_ref_scores)
        # song_ref_weight_list.append(song_ref_weight)
        batch_count += 1
        # if batch_count >= batch_size or idx == len(data_loader) - 1:
        # if label_count >= 390000 or idx == len(data_loader) - 1:
        if batch_count >= gpu_train_size or idx == len(data_loader) - 1:
            predicted_label = get_model_prediction(model, comment_vector_list, ref_labels_vector_list, type=args.model_type)
            song_ref_scores_list_batch = torch.FloatTensor(song_ref_scores_list).to(device)
            acc = compute_acc(predicted_label.flatten(), song_ref_scores_list_batch.flatten(), threshold=0.5)

            loss = criterion(predicted_label, song_ref_scores_list_batch.view(predicted_label.size()[0], 1))
            loss.backward()
            optimizer.step()
            print('epoch {:3d} | {:5d}/{:5d} batches '
                  '| loss {:8.5f}'.format(epoch, idx, len(data_loader), loss.item()))
            # batch_loss.append(loss)
            # epoch_loss.append(loss.item())
            batch_count = 0
            # update_count += 1
            comment_vector_list = []
            ref_labels_vector_list = []
            song_ref_scores_list = []

    return loss.item(), acc


def evaluate(model, data_loader, criterion, batch_size):
    model.eval()
    total_acc, total_count = 0, 0
    batch_loss = []
    batch_count = 0
    gpu_train_size = 400000 / 2
    comment_vector_list = []
    ref_labels_vector_list = []
    song_ref_scores_list = []
    song_ref_weight_list = []
    with torch.no_grad():
        for idx, (song_comments_vector, ref_labels_vector, ref_sample_type, song_id) in enumerate(data_loader):
            #print("processing idx {}".format(str(idx)))
            # comment_vector = [song_comments_vector["layer_1"], song_comments_vector["layer_last"]]
            comment_vector_list.append(song_comments_vector)
            ref_labels_vector_list.append(ref_labels_vector)
            if ref_sample_type == "golden":
                song_ref_scores = 1
                song_ref_weight = 1.0
            elif ref_sample_type == "negative":
                song_ref_scores = 0
                song_ref_weight = 1.0
            else:  # silver
                song_ref_scores = 1
                song_ref_weight = 1.0
            song_ref_scores_list.append(song_ref_scores)
            song_ref_weight_list.append(song_ref_weight)
            batch_count += 1
            # if batch_count >= batch_size or idx == len(data_loader) - 1:
            # if label_count >= 390000 or idx == len(data_loader) - 1:
            if batch_count >= gpu_train_size or idx == len(data_loader) - 1:
                predicted_label = get_model_prediction(model, comment_vector_list, ref_labels_vector_list, type=args.model_type)

                song_ref_scores_list_batch = torch.FloatTensor(song_ref_scores_list).to(device)
                song_ref_weight_list_batch = torch.FloatTensor(song_ref_weight_list).to(device)

                # loss_tensor = torch.stack(
                #     [criterion(predicted_label.flatten()[i], song_ref_scores_list_batch.flatten()[i]) for i
                #      in range(song_ref_scores_list_batch.flatten().size()[0])])
                # loss_weighted = torch.sum(loss_tensor * song_ref_weight_list_batch) / \
                #                 song_ref_scores_list_batch.flatten().size()[0]
                # loss = criterion(predicted_label, song_ref_scores_list_batch.view(predicted_label.size()[0], 1))
                loss = criterion(predicted_label, song_ref_scores_list_batch.view(predicted_label.size()[0], 1).to(device))
                batch_loss.append(loss.item())
                acc = compute_acc(predicted_label.flatten(), song_ref_scores_list_batch.flatten(), threshold=0.5)
                total_acc += acc
                batch_count = 0
                comment_vector_list = []
                ref_labels_vector_list = []
                song_ref_scores_list = []
                song_ref_weight_list = []
                total_count += 1

        average_acc = total_acc/total_count

        #logging.INFO('loss {:8.3f} | acc {:8.3f}'.format(sum(batch_loss)/total_count, average_acc))
    return float(sum(batch_loss))/len(batch_loss), average_acc

def compute_acc(predicted_label, ref_scores, threshold):
    predicted_label_tmp = predicted_label.view(predicted_label.size()[0]).tolist()
    predicted_label_tmp2 = [1 if one_pred > threshold else 0 for one_pred in predicted_label_tmp]
    ref_scores_tmp = ref_scores.tolist()
    total_acc_sum = sum(
        [1 if (predicted_label_tmp2[i] == ref_scores_tmp[i] and ref_scores_tmp[i] == 1.0) else 0 for i in range(len(predicted_label_tmp2))])
    total_acc = total_acc_sum / (sum(predicted_label_tmp2) if sum(predicted_label_tmp2) > 0 else 1e6)
    return total_acc

def test(model, data_loader, criterion, index2word):
    model.eval()
    total_acc, total_count = 0, 0
    doc_score_dict = {}
    batch_loss = []
    #k_list = [1, 5, 10, 50, "N"]
    k_list = [1, 10, 50, 100]
    result_dict = {}
    for k in k_list:
        result_dict[k] = {"hr_k": [], "precision_k": [], "recall_k": [], "f1_k": [], "map_k": [], "ndcg_k": []}
    with torch.no_grad():
        for idx, (text, text_len, song_candidate_words_vector, ref_scores, ref_labels, song_ref_weight, song_id) in enumerate(data_loader):
            #print("processing idx {}".format(str(idx)))
            predicted_label = model(text.to(device), text_len.to(device))
            result_dict, final_score_dict_sort = compute_hr(predicted_label, ref_labels, k_list, result_dict)
            doc_score_dict[song_id] = final_score_dict_sort
            #acc = compute_acc(predicted_label, ref_scores)
            #total_acc += acc
            total_count += 1
        for one_k in result_dict:
            hr_k_acc = sum([one_song[0] for one_song in result_dict[one_k]["hr_k"]])
            hr_k_total = sum([one_song[1] for one_song in result_dict[one_k]["hr_k"]])
            hr_k_average = hr_k_acc / hr_k_total
            precision_k_average = sum(result_dict[one_k]["precision_k"]) / total_count
            recall_k_average = sum(result_dict[one_k]["recall_k"]) / total_count
            f1_k_average = sum(result_dict[one_k]["f1_k"]) / total_count
            map_k_average = np.round(np.average(np.array(result_dict[one_k]["map_k"])), 5)
            ndcg_k_average = np.round(np.average(np.array(result_dict[one_k]["ndcg_k"])), 5)
            #map_k_average = sum(result_dict[one_k]["map_k"]) / total_count
            print('K {} | hr_k {:8.3f} | precision_k {:8.3f} | recall_k {:8.3f} | f1_k {:8.3f} | map_k {:8.3f} | ndcg {:8.3f}'.format(
                one_k,
                hr_k_average,
                precision_k_average,
                recall_k_average,
                f1_k_average,
                map_k_average,
                ndcg_k_average
            ))
    json.dump(doc_score_dict, open("doc_score_dict_train_data", "w", encoding='utf-8'))

    return 0.1, 0.1

def update_match_score(model, data_loader, label_class,data_list):
    data_dict = convert_list_to_dict(data_list)
    model.eval()
    batch_count = 0
    comment_vector_list = []
    song_candidate_words_vector_list = []
    song_candidate_words_list = []
    song_id_list = []
    not_found_words_list = []
    with torch.no_grad():
        for idx, (song_comments_vector, one_candidate_word, song_candidate_word_vector, song_id) in enumerate(
                data_loader):
            # comment_vector = [song_comments_vector["layer_1"], song_comments_vector["layer_last"]]
            comment_vector_list.append(song_comments_vector)
            song_candidate_words_vector_list.append(song_candidate_word_vector)
            song_id_list.append(song_id)
            song_candidate_words_list.append(one_candidate_word)
            batch_count += 1
            if batch_count == 200000 or idx == len(data_loader.train_data) - 1:
                predicted_label = get_model_prediction(model, comment_vector_list, song_candidate_words_vector_list, type=args.model_type)
                predicted_label_flatten = predicted_label.flatten().tolist()
                assert len(song_id_list) == len(song_candidate_words_list)
                for j in range(len(song_id_list)):
                    one_song_id = song_id_list[j]
                    one_candidate_word = song_candidate_words_list[j]
                    if one_candidate_word in data_dict[one_song_id]["final_score_dict_sort"]:
                        data_dict[one_song_id]["final_score_dict_sort"][one_candidate_word][-1] = predicted_label_flatten[j]
                    else:
                        if label_class=='expert':
                            continue    
                        else:
                            not_found_words_list.append("|".join([one_song_id, one_candidate_word]))
                        pass
                if len(not_found_words_list) > 0:
                    print(len(not_found_words_list))
                batch_count = 0
                comment_vector_list = []
                song_candidate_words_vector_list = []
                song_candidate_words_list = []
                song_id_list = []

        open("not_found_words.txt", "w", encoding='utf-8').write("\n".join(not_found_words_list))

        #todo 将match score排名更新到13位中
        # for one_key in data_dict:
        #     final_score_dict = data_dict[one_key]["final_score_dict_sort"]
        #     # TODO SORT BY MATCH SCORE

    return convert_dict_to_list(data_dict)

def update_silver_label_finetune(data_list, diversity_topk, valid_threshold, match_score_threshold, step):
    added_silver_labels = {}
    for i in range(len(data_list)):
        if "song_silver_labels" not in data_list[i]:
            data_list[i]["song_silver_labels"] = []
        if "song_silver_label_details" not in data_list[i]:
            data_list[i]["song_silver_label_details"] = {}
        final_score_dict = data_list[i]["final_score_dict_sort"]
        final_score_dict_sort = {k: v for k, v in sorted(final_score_dict.items(),
                                 key=lambda item: item[1][4],
                                 reverse=True)}
        for one_silver_label in data_list[i]["song_silver_labels"]:
            if one_silver_label in final_score_dict_sort:
                final_score_dict_sort.pop(one_silver_label)

        for one_key in list(final_score_dict_sort.keys())[0:diversity_topk]:
        #for one_key in list(final_score_dict_sort.keys()):
            if final_score_dict_sort[one_key][3] > valid_threshold and final_score_dict_sort[one_key][6] > match_score_threshold:
                if one_key not in data_list[i]["song_silver_labels"] and one_key not in data_list[i]['song_pseudo_golden_labels']:
                    data_list[i]["song_silver_labels"].append(one_key)
                    if one_key not in added_silver_labels:
                        added_silver_labels[one_key] = 1
        data_list[i]["song_silver_label_details"][str(step)] = data_list[i]["song_silver_labels"]
    return data_list, added_silver_labels

def update_silver_label_average(data_list, average_threshold, valid_threshold, match_score_threshold, step):
    added_silver_labels = {}
    for i in range(len(data_list)):
        if "song_silver_labels" not in data_list[i]:
            data_list[i]["song_silver_labels"] = []
        if "song_silver_label_details" not in data_list[i]:
            data_list[i]["song_silver_label_details"] = {}
        final_score_dict = data_list[i]["final_score_dict_sort"]
        final_score_dict_sort = {k: v for k, v in sorted(final_score_dict.items(),
                                 key=lambda item: item[1][6],
                                 reverse=True)}
        for one_silver_label in data_list[i]["song_silver_labels"]:
            if one_silver_label in final_score_dict_sort:
                final_score_dict_sort.pop(one_silver_label)

        for one_key in list(final_score_dict_sort.keys()):
            # final_score_dict_sort[one_key][0] is updated during update match score.
            if args.ebc_type=='ebc':
                eval_val=final_score_dict_sort[one_key][6]
            else:
                eval_val=(final_score_dict_sort[one_key][4]+final_score_dict_sort[one_key][6]+final_score_dict_sort[one_key][5])/3
            if eval_val >= average_threshold:
                if one_key not in data_list[i]["song_silver_labels"] and one_key not in data_list[i]['song_pseudo_golden_labels']:
                    data_list[i]["song_silver_labels"].append(one_key)
                    if one_key not in added_silver_labels:
                        added_silver_labels[one_key] = 1
        data_list[i]["song_silver_label_details"][str(step)] = copy.deepcopy(data_list[i]["song_silver_labels"])
    return data_list, added_silver_labels

# 当前生成silver label的方法
def update_silver_label_yaoyao(data_list, average_threshold, valid_threshold, match_score_threshold, step):
    added_silver_labels = {}
    for i in range(len(data_list)):
        if "song_silver_labels" not in data_list[i]:
            data_list[i]["song_silver_labels"] = []
        if "song_silver_label_details" not in data_list[i]:
            data_list[i]["song_silver_label_details"] = {}
        
        # todo 每首歌都生成分数表 包含tf-idf++ (site 4) 和 match score (site 6) 和 novelty (site 12) macth score rank (site 13)
        # 分数表 scoer_tables
        score_tabels={}
        final_score_dict = data_list[i]["final_score_dict_sort"]
        for one_key in final_score_dict:
            score_tabels[one_key]={}
            score_tabels[one_key]["tf-idf++"]=final_score_dict[one_key][4]
            score_tabels[one_key]["match score"]=final_score_dict[one_key][6]
            score_tabels[one_key]["novelty score"]=final_score_dict[one_key][12]
            score_tabels[one_key]["match rank"]=final_score_dict[one_key][13]
            # tf-idf++ / math.sqrt(match rank)
            score_tabels[one_key]["divers_score1"]=final_score_dict[one_key][4]/math.sqrt(final_score_dict[one_key][13])
            score_tabels[one_key]["divers_score2"]=math.pow(final_score_dict[one_key][12],2)+math.pow(final_score_dict[one_key][6],2)
        final_score_dict_sort = {k: v for k, v in sorted(final_score_dict.items(),
                                 key=lambda item: item[1][6],
                                 reverse=True)}
        for one_silver_label in data_list[i]["song_silver_labels"]:
            if one_silver_label in final_score_dict_sort:
                final_score_dict_sort.pop(one_silver_label)

        for one_key in list(final_score_dict_sort.keys()):
            # final_score_dict_sort[one_key][0] is updated during update match score.
            if args.ebc_type=='ebc':
                eval_val=final_score_dict_sort[one_key][6]
            else:
                eval_val=(final_score_dict_sort[one_key][4]+final_score_dict_sort[one_key][6]+final_score_dict_sort[one_key][5])/3
            if eval_val >= average_threshold:
                if one_key not in data_list[i]["song_silver_labels"] and one_key not in data_list[i]['song_pseudo_golden_labels']:
                    data_list[i]["song_silver_labels"].append(one_key)
                    if one_key not in added_silver_labels:
                        added_silver_labels[one_key] = 1
        data_list[i]["song_silver_label_details"][str(step)] = copy.deepcopy(data_list[i]["song_silver_labels"])
    return data_list, added_silver_labels

def   update_match_score_main(model_path_list, classification_model, optimizer, vocab, step, running_type, data_folder, data_prefix, sample_method,
                            train_list,test_list,train_song_id_layer_outputs_cls, val_song_id_layer_outputs_cls, train_data_loader, test_data_loader,label_class):
    for model_path in model_path_list:
        # 计算match score 之前先初始化data_loader
        train_data_loader = CustomDatasetBinaryUpdateMatchScore(convert_list_to_dict(train_list), train_song_id_layer_outputs_cls,
                                                                candidate_words_layer_outputs=train_data_loader.candidate_words_layer_outputs,
                                                                song_candidate_words_dict=train_data_loader.song_candidate_words_dict,
                                                                train_data=train_data_loader.train_data)
        train_data_loader.get_all_infos()
        test_data_loader = CustomDatasetBinaryUpdateMatchScore(convert_list_to_dict(test_list), val_song_id_layer_outputs_cls,
                                                               candidate_words_layer_outputs=test_data_loader.candidate_words_layer_outputs,
                                                               song_candidate_words_dict=test_data_loader.song_candidate_words_dict,
                                                               train_data=test_data_loader.train_data)
        test_data_loader.get_all_infos()
        val_loss = load_checkpoint(load_path=model_path, model=classification_model, optimizer=optimizer)
        # match score will be updated
        train_list_new = update_match_score(classification_model, train_data_loader,label_class, data_list=train_list)
        test_list_new = update_match_score(classification_model, test_data_loader,label_class,data_list=test_list)
        # json.dump(train_list_new, open("data/{}/{}_train_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
        # json.dump(test_list_new, open("data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
        return train_list_new, test_list_new

def analyse_silver_labels(data_list, step):
    song_has_new_silver_label = []
    total_new_silver_label_added = []
    unique_silver_label_added = {}
    total_previous_label = []
    total_current_label = []
    for i in range(len(data_list)):
        song_id = get_song_id(data_list[i])
        if step == 0:
            previous_silver_labels = []
        else:
            if str(step-1) in data_list[i]["song_silver_label_details"]:
                previous_silver_labels = data_list[i]["song_silver_label_details"][str(step-1)]
            else:
                previous_silver_labels = []
        if str(step) in data_list[i]["song_silver_label_details"]:
            current_silver_label = data_list[i]["song_silver_label_details"][str(step)]
        else:
            current_silver_label = []
        total_previous_label += previous_silver_labels
        total_current_label += current_silver_label
        new_silver_labels = list(set(current_silver_label) - set(previous_silver_labels))
        if len(new_silver_labels) > 0:
            song_has_new_silver_label.append(song_id)
            total_new_silver_label_added += new_silver_labels
            for one_new_silver_label in new_silver_labels:
                if one_new_silver_label not in unique_silver_label_added:
                    unique_silver_label_added[one_new_silver_label] = 1
                else:
                    unique_silver_label_added[one_new_silver_label] += 1
    total_unique_new_label_add = list(set(total_current_label) - set(total_previous_label))
    return song_has_new_silver_label, total_new_silver_label_added, unique_silver_label_added, total_unique_new_label_add

def update_silver_label_main(step,train_list,test_list, data_folder, data_prefix, topk, threshold, update_method, sample_method):
    # train_list = json.load(open("data/{}/{}_train_t{}.json".format(data_folder, data_prefix, str(step + 1)), "r", encoding='utf-8'))
    # test_list = json.load(open("data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(step + 1)), "r", encoding='utf-8'))

    if "average" in update_method:
        train_list_new, train_silver_label = update_silver_label_average(data_list=train_list, average_threshold=threshold,
                                                                 valid_threshold=threshold,
                                                                 match_score_threshold=threshold, step=step)
        test_list_new, test_silver_label = update_silver_label_average(data_list=test_list, average_threshold=threshold,
                                                               valid_threshold=threshold,
                                                               match_score_threshold=threshold, step=step)

    else:
        train_list_new, train_silver_label = update_silver_label_finetune(data_list=train_list, diversity_topk=topk,
                                                                 valid_threshold=threshold, match_score_threshold=threshold, step=step)
        test_list_new, test_silver_label = update_silver_label_finetune(data_list=test_list, diversity_topk=topk, valid_threshold=threshold,
                                                               match_score_threshold=threshold, step=step)
    song_has_new_silver_label, \
    total_new_silver_label_added, \
    unique_silver_label_added, \
    total_unique_new_label_add = analyse_silver_labels(train_list_new, step=step)
    print("train song_has_new_silver_label:{}".format(len(song_has_new_silver_label)))
    print("train total_new_silver_label_added:{}".format(len(total_new_silver_label_added)))
    print("train unique_silver_label_added_in_this_iter:{}".format(len(unique_silver_label_added)))
    print("train unique_silver_label_added_in_all_iter:{}".format(len(total_unique_new_label_add)))
    logging.info("step: {}\n".format(str(step)))
    logging.info("train song_has_new_silver_label:{}\n".format(len(song_has_new_silver_label)))
    logging.info("train total_new_silver_label_added:{}\n".format(len(total_new_silver_label_added)))
    logging.info("train unique_silver_label_added_in_this_iter:{}\n".format(len(unique_silver_label_added)))
    logging.info("train unique_silver_label_added_in_all_iter: {}\n".format(str(len(total_unique_new_label_add))))

    song_has_new_silver_label, \
    total_new_silver_label_added, \
    unique_silver_label_added, \
    total_unique_new_label_add = analyse_silver_labels(test_list_new, step=step)
    print("test song_has_new_silver_label:{}".format(len(song_has_new_silver_label)))
    print("test total_new_silver_label_added:{}".format(len(total_new_silver_label_added)))
    print("test unique_silver_label_added_in_this_iter:{}".format(len(unique_silver_label_added)))
    print("test unique_silver_label_added_in_all_iter:{}".format(len(total_unique_new_label_add)))
    logging.info("step: {}\n".format(str(step)))
    logging.info("test song_has_new_silver_label:{}\n".format(len(song_has_new_silver_label)))
    logging.info("test total_new_silver_label_added:{}\n".format(len(total_new_silver_label_added)))
    logging.info("test unique_silver_label_added_in_this_iter:{}\n".format(len(unique_silver_label_added)))
    logging.info("test unique_silver_label_added_in_all_iter: {}\n".format(str(len(total_unique_new_label_add))))

    return copy.deepcopy(train_list_new), copy.deepcopy(test_list_new)


def build_fine_tuning_data_main(step, data_folder, data_prefix, topk, threshold, train_list, test_list):
    train_finetune_list = build_fine_tuning_data(train_list)
    test_finetune_list = build_fine_tuning_data(test_list)

    json.dump(train_finetune_list,
              open("data/{}/{}_train_finetune_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
    json.dump(test_finetune_list,
              open("data/{}/{}_test_finetune_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
    return train_finetune_list, test_finetune_list

def get_finetune_data(one_doc_data):
    one_doc_dict = {}
    #for one_key in one_doc_data:
    #    one_doc_dict[one_key] = one_doc_data[one_key]
    one_doc_dict['song_name'] = one_doc_data['song_name']
    one_doc_dict['song_comments_detail_final'] = one_doc_data['song_comments_detail_final']
    one_doc_dict['song_silver_labels'] = one_doc_data['song_silver_labels']
    one_doc_dict['final_score_dict_sort'] = one_doc_data['final_score_dict_sort']
    golden_labels_list = one_doc_data['song_pseudo_golden_labels']
    negative_labels_list = one_doc_data['negative_samples']['doc'] + one_doc_data['negative_samples']['label']
    if len(golden_labels_list) < len(one_doc_data["song_silver_labels"]):
        golden_labels_random_list = list(
            np.random.choice(golden_labels_list, len(one_doc_data["song_silver_labels"]), replace=True))
    else:
        golden_labels_random_list = list(
            np.random.choice(golden_labels_list, len(one_doc_data["song_silver_labels"]), replace=False))
    if len(negative_labels_list) < len(one_doc_data["song_silver_labels"]):
        negative_labels_random_list = list(
            np.random.choice(negative_labels_list, len(one_doc_data["song_silver_labels"]), replace=True))
    else:
        negative_labels_random_list = list(
            np.random.choice(negative_labels_list, len(one_doc_data["song_silver_labels"]), replace=False))
    one_doc_dict['negative_samples'] = {"doc": negative_labels_random_list, "label": []}
    one_doc_dict['song_pseudo_golden_labels'] = golden_labels_random_list
    return one_doc_dict

def build_fine_tuning_data(data_list):
    new_data_list = []
    data_dict = convert_list_to_dict(data_list)
    all_song_id_list = list(data_dict.keys())
    silver_label_song_id_list = []
    for i in range(len(data_list)):
        has_new_silver_label = False
        if "song_silver_labels" in data_list[i] and len(data_list[i]["song_silver_labels"]) > 0:
            step_num = len(data_list[i]["song_silver_label_details"])
            if step_num > 1:
                new_silver_label_list = list(set(data_list[i]["song_silver_label_details"][str(step_num-1)]) - set(data_list[i]["song_silver_label_details"][str(step_num-2)]))
                if len(new_silver_label_list) > 0:
                    has_new_silver_label = True
            else:
                has_new_silver_label = True
            if has_new_silver_label:
                song_id = get_song_id(data_list[i])
                silver_label_song_id_list.append(song_id)
                one_doc_dict = get_finetune_data(data_list[i])
                new_data_list.append(one_doc_dict)
    none_silver_label_song_id_list = list(set(all_song_id_list) - set(silver_label_song_id_list))
    sample_num = min([len(silver_label_song_id_list), len(none_silver_label_song_id_list)])
    none_silver_label_song_id_sample_list = list(np.random.choice(none_silver_label_song_id_list, sample_num, replace=False))
    for one_song_id in none_silver_label_song_id_sample_list:
        new_data_list.append(data_dict[one_song_id])
    return new_data_list

def update_valid_score(data_list, label_cluster_distance_dict):
    #all_valid_score = []
    for i in range(len(data_list)):
        #valid_score_list = []
        for one_word in data_list[i]["final_score_dict_sort"]:
            silver_valid_score = label_cluster_distance_dict[one_word] if one_word in label_cluster_distance_dict else 0
            valid_score = silver_valid_score
            data_list[i]["final_score_dict_sort"][one_word][3] = valid_score
            #valid_score_list.append(valid_score)
            #all_valid_score.append(valid_score)
    #valid_score_average = statistics.mean(all_valid_score)
    #factor = 1.0 / valid_score_average  # rf-t-idf average 1 /  valid average
    # for i in range(len(data_list)):
    #     for one_word in data_list[i]["final_score_dict_sort"]:
    #         data_list[i]["final_score_dict_sort"][one_word][3] = data_list[i]["final_score_dict_sort"][one_word][3] * factor
    return data_list

def update_valid_score_main(step, data_folder, data_prefix, sample_method, vocab, train_list, test_list, save_processed_file):
    train_previous_label_cluster_distance_dict = {}
    for i in range(len(train_list)):
        for one_word in train_list[i]['final_score_dict_sort']:
            if one_word not in train_previous_label_cluster_distance_dict:
                train_previous_label_cluster_distance_dict[one_word] = train_list[i]['final_score_dict_sort'][one_word][3]
    train_words_list = list(train_previous_label_cluster_distance_dict.keys())

    test_previous_label_cluster_distance_dict = {}
    for i in range(len(test_list)):
        for one_word in test_list[i]['final_score_dict_sort']:
            if one_word not in test_previous_label_cluster_distance_dict:
                test_previous_label_cluster_distance_dict[one_word] = test_list[i]['final_score_dict_sort'][one_word][3]
    test_words_list = list(test_previous_label_cluster_distance_dict.keys())

    train_unique_silver_label_in_current_step, train_golden_labels = get_unique_silver_labels_given_data_list(train_list, step)
    test_unique_silver_label_in_current_step, test_golden_labels = get_unique_silver_labels_given_data_list(test_list, step)
    new_train_list = []
    if len(train_unique_silver_label_in_current_step) > 0:
        train_label_cluster_distance_dict, train_label_cluster_distance_dict_detail = compute_label_cluster_distance_vocab(
            vocab=vocab,
            labels_all_list=train_unique_silver_label_in_current_step,
            words_list=train_words_list)
        for one_word in train_words_list:
            new_distance = train_label_cluster_distance_dict[
                one_word] if one_word in train_label_cluster_distance_dict else 0
            previous_most_similar_distance = train_previous_label_cluster_distance_dict[
                one_word] if one_word in train_previous_label_cluster_distance_dict else 0
            train_label_cluster_distance_dict[one_word] = max([new_distance, previous_most_similar_distance])
        new_train_list = update_valid_score(data_list=train_list,
                                            label_cluster_distance_dict=train_label_cluster_distance_dict)
    new_test_list = []
    if len(test_unique_silver_label_in_current_step) > 0:
        test_label_cluster_distance_dict, test_label_cluster_distance_dict_detail = compute_label_cluster_distance_vocab(
            vocab=vocab,
            labels_all_list=test_unique_silver_label_in_current_step,
            words_list=test_words_list)
        for one_word in test_words_list:
            new_distance = test_label_cluster_distance_dict[one_word] if one_word in test_label_cluster_distance_dict else 0
            previous_most_similar_distance = test_previous_label_cluster_distance_dict[
                one_word] if one_word in test_previous_label_cluster_distance_dict else 0
            test_label_cluster_distance_dict[one_word] = max([new_distance, previous_most_similar_distance])
        new_test_list = update_valid_score(data_list=test_list, label_cluster_distance_dict=test_label_cluster_distance_dict)

    if len(new_train_list) == 0:
        new_train_list = train_list
    if len(new_test_list) == 0:
        new_test_list = test_list
    if save_processed_file:
        # json.dump(new_train_list, open("data/{}/{}_train_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w",
        #                                encoding='utf-8'))
        json.dump(new_test_list,
                  open("data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
    return copy.deepcopy(new_train_list), copy.deepcopy(new_test_list)

def update_negative_sample_using_probability_main(step, data_folder, data_prefix, sample_method, negative_sample_method, train_list, test_list):

    '''
    data_list = train_list + test_list
    copper_label_to_song_id_dict = get_copper_label_to_song_id_dict(data_list)  # each copper label appears in which sings
    label_cluster_distance_dict_detail = json.load(open(os.path.join(ROOT_PATH, "data", "label_cluster_distance_dict_detail.json")))
    '''

    train_list_new = compute_negative_sample(train_list, sample_type=negative_sample_method)
    test_list_new = compute_negative_sample(test_list, sample_type=negative_sample_method)

    if step==args.ITERATIONS-1:
        json.dump(train_list_new,
                open("/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/DIVA-Large-Probability-Negative-Sample-Log_rf-Self_training/data/{}/{}_train_t{}.json".format(data_folder, data_prefix , str(step + 1)), "w", encoding='utf-8'))
        json.dump(test_list_new,
                open("/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/DIVA-Large-Probability-Negative-Sample-Log_rf-Self_training/data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))

    return train_list_new, test_list_new

def update_negative_sample_for_silver_label_main(step, data_folder, data_prefix, sample_method):
    train_path = os.path.join("data", "{}".format(data_folder), "{}_train_t{}.json".format(data_prefix, str(step + 1)))
    test_path = os.path.join("data", "{}".format(data_folder), "{}_test_t{}.json".format(data_prefix, str(step + 1)))
    annotation_path = os.path.join("data", "{}".format(data_folder), "{}_annotation_t{}.json".format(data_prefix, str(step + 1)))

    train_list = json.load(open(train_path))
    test_list = json.load(open(test_path))
    annotation_list = json.load(open(annotation_path))
    data_list = train_list + test_list + annotation_list
    copper_label_to_song_id_dict = get_copper_label_to_song_id_dict(data_list)  # each copper label appears in which sings
    label_cluster_distance_dict_detail = json.load(open(os.path.join(ROOT_PATH, "data", "label_cluster_distance_dict_detail.json")))
    train_list_new = compute_negative_sample_for_silver_label(train_list, copper_label_to_song_id_dict, label_cluster_distance_dict_detail, step)
    test_list_new = compute_negative_sample_for_silver_label(test_list, copper_label_to_song_id_dict,
                                                              label_cluster_distance_dict_detail, step)
    annotation_list_new = compute_negative_sample_for_silver_label(annotation_list, copper_label_to_song_id_dict,
                                                              label_cluster_distance_dict_detail, step)

    json.dump(train_list_new,
              open("data/{}/{}_train_t{}.json".format(data_folder, data_prefix , str(step + 1)), "w", encoding='utf-8'))
    json.dump(test_list_new,
              open("data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w", encoding='utf-8'))
    json.dump(annotation_list_new,
              open("data/{}/{}_annotation_t{}.json".format(data_folder, data_prefix, str(step + 1)), "w",
                   encoding='utf-8'))

    return train_list_new, test_list_new, annotation_list_new

def main():
    # args.running_type='train'
    # args.model_type='cat_2'
    # args.model_alias='use_layer_0_last_cat_2_run_fasttext_global_word_embedding'
    # # args.threshold=0.95
    # args.update_method='average'
    # args.sample_method='all'
    # args.negative_sample_method='random'
    # args.optimizer='Adam'
    # # args.diva_label_class='expert'
    # # args.positive_partition=0.07
    # args.lr=0.001
    # # debug iteration epoch 300
    # args.EPOCHS=300
    # args.ITERATIONS=20
    # # args.easy_self_training=True
    
    running_type = args.running_type
    topk = args.diversity_topk
    model_alias = args.model_alias
    threshold = args.threshold
    update_method = args.update_method
    sample_method = args.sample_method
    negative_sample_method = args.negative_sample_method
    
    label_class= args.diva_label_class
    file_version=args.file_version
    tf_idf_type=args.tf_idf_type
    uncertainty_limit=args.uncertainty_limit
    with_tf_idf=True
    with_novelty=True
    with_valid=args.with_valid
    light=args.light
    clusters_num=args.clusters_num
    clustering_times=args.clustering_times
    ebc_thres=args.ebc_thres
    start_iter=args.start_iter
    easy_self_training=args.easy_self_training
    discard_positive=args.discard_positive
    
    joint_score_thres=args.joint_score_thres
    debug=args.debug
    update_thres=args.update_thres
    dumping_weight=args.dumping_weight
    valid_thres= args.valid_thres

    with_valid_semantic=True
    with_valid_disc=True
    only_joint=args.only_joint
    print(with_valid_semantic,with_valid_disc,with_novelty,with_tf_idf)
    
    if debug:
        args.EPOCHS=1
        
    PE_thres=args.PE_thres
    assert negative_sample_method in ["probability", "random"]

    EPOCHS = args.EPOCHS  # 200  # epoch920101
    ITERATIONS = args.ITERATIONS  # 15
    #todo NO ITERATION
    LR = args.lr  # 0.01#0.01  # learning rate
    # BATCH_SIZE = 1536 if LOGIN_NAME == "yzhang" else 768  # batch size for training
    # BATCH_SIZE = 20480 if LOGIN_NAME == "yzhang" else 6144
    BATCH_SIZE = args.BATCH_SIZE  # 20000#int(10240 * 2)
    # step = 3
    encode_type = args.encode_type
    if args.code_start_time != "None":
        CODE_START_TIME = args.code_start_time
    else:
        CODE_START_TIME = utils.CODE_START_TIME
    scheduler_patience = 10
    train_patience = 20
    args_list = ['running_type', 'diversity_topk', 'model_alias', 'threshold', 'update_method', 'sample_method',
                 'negative_sample_method', 'model_type', 'lr', 'optimizer', 'EPOCHS', 'ITERATIONS', 'BATCH_SIZE', 'encode_type','with_valid_semantic','with_valid_disc','with_tf_idf','with_novelty','only_joint']
    attr_list = []
    for arg in args_list:
        print(arg, getattr(args, arg))
        attr_list.append(str(getattr(args, arg)).replace(".", "_"))
    arguments_str = "_".join(attr_list)

    model_prefix = arguments_str
    data_prefix = arguments_str
    # data_folder = "{}_{}_{}_{}".format(data_prefix, model_alias, update_method, sample_method)
    # model_folder = "model_{}_{}_{}_{}".format(model_prefix, model_alias, update_method, sample_method)
    data_folder = "{}_{}".format(arguments_str, CODE_START_TIME)
    model_folder = "model_{}_{}".format(arguments_str, CODE_START_TIME)
    if not os.path.exists("/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/DIVA-Large-Probability-Negative-Sample-Log_rf-Self_training/data/{}".format(data_folder)):
        os.mkdir("/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/DIVA-Large-Probability-Negative-Sample-Log_rf-Self_training/data/{}".format(data_folder))
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    logging.basicConfig(filename="log_{}_{}.txt".format(arguments_str, CODE_START_TIME),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logging.info('Starting time: {}'.format(arguments_str))
    logging.info('Starting time: {}'.format(CODE_START_TIME))

    if running_type == "train" or not os.path.exists(os.path.join(ROOT_PATH, "vocab.json")):
        train_data_segmented_list = json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter_based.json'))
        words_list = []
        labels_all_list = []
        for i in range(len(train_data_segmented_list)):
            comments_list = train_data_segmented_list[i]['song_comments_detail_final']
            labels_list = train_data_segmented_list[i]["song_labels"]
            labels_list2 = train_data_segmented_list[i]["song_silver_labels"] if "song_silver_labels" in \
                                                                                 train_data_segmented_list[i] else []
            for one_comment_id in comments_list:
                for one_sent_id in comments_list[one_comment_id]:
                    if one_sent_id != 'likecnt_weight':
                        words_list += comments_list[one_comment_id][one_sent_id]['song_view_segmented']

            for one_label in labels_list:
                for sub_label in one_label:
                    if sub_label not in labels_all_list:
                        labels_all_list.append(sub_label)
        words_list += labels_all_list
        vocab = Vocab(data=words_list, load_embedding=os.path.join(ROOT_PATH, "vocab.json"), labels=labels_all_list)
        #vocab.save_vocab()
    else:
        # vocab.json exist then load from this json file.
        vocab = Vocab(data=[], load_embedding=os.path.join(ROOT_PATH, "vocab.json"), labels=[])
    if args.use_model == "pre-train":
        classification_model = TextClassificationModelBinaryXLNet().to(device)
    elif args.use_model == "lstm":
        classification_model = TextClassificationModelBinaryLSTM()
    else:
        raise ValueError
    for name, param in classification_model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()
    # loss = focalloss + bceloss
    # criterion = FocalLossBCE()
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classification_model.parameters(), lr=LR)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(classification_model.parameters(), lr=LR)
    else:
        raise ValueError
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=scheduler_patience,  verbose=True)
    '''
    song_id_layer_outputs_cls = json.load(
        open(join(ROOT_PATH, "Preprocess", "train_song_id_layer_outputs_cls.json"), "r", encoding='utf-8'))
    val_song_id_layer_outputs_cls = json.load(
        open(join(ROOT_PATH, "Preprocess", "val_song_id_layer_outputs_cls.json"), "r", encoding='utf-8'))
    
    for one_key in val_song_id_layer_outputs_cls:
        song_id_layer_outputs_cls[one_key] = val_song_id_layer_outputs_cls[one_key]
    '''
    #todo 加载 song_comment_vector
    song_id_layer_outputs_cls = json.load(
        open("DiVa/data/tme_big_data/train_song_id_layer_1_last_1108.json", "r", encoding='utf-8'))
    val_song_id_layer_outputs_cls = json.load(
        open("DiVa/data/tme_big_data/val_song_id_layer_1_last_1108.json", "r", encoding='utf-8'))
    for one_key in val_song_id_layer_outputs_cls:
        song_id_layer_outputs_cls[one_key] = val_song_id_layer_outputs_cls[one_key]

    best_epoch_list = []
    train_data_loader = CustomDatasetBinary(song_id_layer_outputs_cls, label_class=label_class,data_type="train",file_version=file_version,light=light,start_iter=start_iter,discard_positive=discard_positive)
    train_data_loader.process_t0_data()
    # train_data_loader.initial_data_loader_process_and_save()
    # train_data_loader.initial_data_loader_load()
    train_data_loader.initial_data_loader()

    val_data_loader = CustomDatasetBinary(song_id_layer_outputs_cls, label_class=label_class,data_type="val",file_version=file_version,light=light,start_iter=start_iter,discard_positive=discard_positive)
    val_data_loader.process_t0_data()
    # val_data_loader.initial_data_loader_process_and_save()
    # val_data_loader.initial_data_loader_load()
    val_data_loader.initial_data_loader()

    if running_type == "train":
        iter_start = start_iter
        one_iter=0
        print('now is trainng, start_iter is ', iter_start)
        last_diverse_all_tags=0
            
        for one_iter in range(iter_start, ITERATIONS):
            classification_model.train()
            if args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(classification_model.parameters(), lr=LR)
            elif args.optimizer == 'Adadelta':
                optimizer = torch.optim.Adadelta(classification_model.parameters(), lr=LR)
            else:
                raise ValueError
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=scheduler_patience,  verbose=True)
            # todo begin to iter
            if one_iter == iter_start:
                if args.use_model == "pre-train":
                    classification_model = TextClassificationModelBinaryXLNet().to(device)
                elif args.use_model == "lstm":
                    classification_model = TextClassificationModelBinaryLSTM()
                else:
                    raise ValueError
                for name, param in classification_model.named_parameters():
                    if "fc" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if args.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(classification_model.parameters(), lr=LR)
                elif args.optimizer == 'Adadelta':
                    optimizer = torch.optim.Adadelta(classification_model.parameters(), lr=LR)
                else:
                    raise ValueError
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3,
                                                                    patience=scheduler_patience, verbose=True)
            loss_list = []
            acc_list = []
            loss_acc_list = []

            best_val_loss = 100

            trigger_times = 0
            # todo 正常情况
            if one_iter >= len(best_epoch_list): #iter_start + len(best_epoch_list): # skip iter < iter_start
                best_epoch = -1
                min_val_loss = 1000
                for epoch in range(1, EPOCHS + 1):
                    epoch_start_time = time.time()
                    epoch_loss, epoch_acc = train(classification_model, train_data_loader, epoch=epoch,
                          optimizer=optimizer, criterion=criterion, index2label=vocab.index2label, batch_size=BATCH_SIZE, encode_type=encode_type)
                    print('train epoch {:3d} | loss {:8.5f}'.format(epoch, epoch_loss))
                    val_loss, accu_val = evaluate(classification_model, val_data_loader, criterion=criterion, batch_size=BATCH_SIZE)
                    scheduler.step(round(val_loss, 5))
                    # if round(val_loss, 5) < min_val_loss and epoch_loss[-1] < 1:
                    if round(val_loss, 5) < min_val_loss and epoch_loss < 1:
                        min_val_loss = round(val_loss, 5)
                        best_epoch = epoch
                    loss_list.append(round(val_loss, 5))
                    acc_list.append(round(accu_val, 5))
                    loss_acc_list.append([round(val_loss, 5), accu_val])
                    print('valid epoch {:3d} | loss {:8.5f} '.format(epoch, val_loss))
                    logging.info('valid epoch {:3d} | loss {:8.5f}'.format(epoch, val_loss))
                    print('-' * 59)
                    print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch,
                                                                         time.time() - epoch_start_time, ))
                    model_save_path = "{}/{}_model_binary_t{}_{}".format(model_folder, model_prefix, str(one_iter), str(epoch))
                    save_checkpoint(save_path=model_save_path,
                                    model=classification_model,
                                    optimizer=optimizer, valid_loss=0.0)
                    print('-' * 59)
                    # early stop
                    print('-' * 59)
                    print('| early stop | ')
                    if len(loss_list) > 1:
                        print("best val loss: {}".format(str(best_val_loss)))
                        print("current val loss: {}".format(str(loss_list[-1])))
                        if loss_list[-1] >= best_val_loss:
                            trigger_times += 1
                            print("trigger time: {}".format(str(trigger_times)))

                            if trigger_times >= train_patience:
                                print('Early stopping! start update match score and silver labels. ')
                                loss_acc_list_sorted = sorted(loss_acc_list, key=lambda x: (x[0], -x[1])) # loss first, acc second
                                best_epoch_list.append(best_epoch)
                                for kk in range(1, len(loss_list)+1):
                                    if kk != best_epoch:
                                        model_path = "{}/{}_model_binary_t{}_{}".format(model_folder, model_prefix, str(one_iter), str(kk))
                                        # if os.path.exists(model_path):
                                        #     os.remove(model_path)
                                break
                            # if trigger_times > scheduler_patience:
                            #     scheduler.step()
                            #     print("learning rate decay to {}".format(scheduler.get_last_lr()[-1]))
                        else:
                            trigger_times = 0
                            best_val_loss = loss_list[-1]
                            print('trigger times: {}'.format(trigger_times))
                    print('-' * 59)
                if trigger_times < train_patience:
                    # iteration ends because of epoch reach limitation
                    print('reach max epoch number! start update match score, valid score, and silver labels. ')
                    loss_acc_list_sorted = sorted(loss_acc_list, key=lambda x: (x[0], -x[1]))  # loss first, acc second
                    #loss_acc_list_sorted = sorted(loss_acc_list, key=lambda x: (-x[1], x[0])) # acc first, loss second
                    #best_epoch = loss_acc_list.index(loss_acc_list_sorted[0]) + 1  # epoch starts from 1
                    best_epoch_list.append(best_epoch)
                    # delete all the models except the best one.
                    # for kk in range(1, len(loss_list) + 1):
                    #     if kk != best_epoch:
                    #         model_path = "{}/{}_model_binary_{}_t{}_{}".format(model_folder, model_prefix, sample_method,
                    #                                                            str(one_iter), str(kk))
                    #         if os.path.exists(model_path):
                    #             os.remove(model_path)
            
            print('best epoch: {}'.format(str(best_epoch_list[-1])))
            # 当前最好epoch的参数路径
            best_epcoh_model_path = "{}/{}_model_binary_t{}_{}".format(model_folder, model_prefix, str(one_iter), str(best_epoch_list[-1]))
            print('reset silver and negative...')
            # todo 计算新一轮的match score(ebc_score) 更新在site=-1的位置
            train_list_new1, test_list_new1 = update_match_score_main(
                model_path_list=["{}/{}_model_binary_t{}_{}".format(model_folder, model_prefix, str(one_iter), str(best_epoch_list[-1]))],
                classification_model=classification_model,
                optimizer=optimizer,
                vocab=vocab,
                step=one_iter,
                running_type=running_type,
                data_folder=data_folder,
                data_prefix=data_prefix,
                sample_method=sample_method,
                train_list=convert_dict_to_list(train_data_loader.raw_data_dict),
                test_list=convert_dict_to_list(val_data_loader.raw_data_dict),
                train_song_id_layer_outputs_cls=song_id_layer_outputs_cls,
                val_song_id_layer_outputs_cls=song_id_layer_outputs_cls,
                train_data_loader=train_data_loader,
                test_data_loader=val_data_loader,
                label_class=label_class,
                )

            # train_list_new2, test_list_new2 = update_valid_score_main(one_iter, data_folder,
            #                             data_prefix, sample_method,
            # 
            #                             train_list=train_list_new1,
            # todo update silver labels
            if easy_self_training==False:
                train_data_list,test_data_list,diverse_all_tags,stop_flag=update_silver_iter(tf_idf_type=tf_idf_type,iter=one_iter,train_datas=train_list_new1,test_datas=test_list_new1,label_class=label_class,file_version=file_version,uncertainty_limit=uncertainty_limit,with_tf_idf=with_tf_idf,with_novelty=with_novelty,with_valid=with_valid,clusters_num=clusters_num,clustering_times=clustering_times,PE_thres=PE_thres,ebc_thres=ebc_thres,joint_score_thres=joint_score_thres,debug=debug,update_thres=update_thres,dumping_weight=dumping_weight,with_valid_semantic=with_valid_semantic,with_valid_disc=with_valid_disc,only_joint=only_joint,valid_thres=valid_thres)
            
                if stop_flag==0:
                    print('iteration over at iter {}!'.format(str(one_iter+1)))
                    break
            
            if easy_self_training:
                stop_flag,train_data_list,test_data_list=update_silver_simple(iter=one_iter,train_datas=train_list_new1,test_datas=test_list_new1,ebc_thres=ebc_thres)
                if stop_flag==0:
                    print('iter over; iterated {} times'.format(str(one_iter+1)))
                    break
            
            print('update silver!')
            if easy_self_training==False:
                if diverse_all_tags==last_diverse_all_tags:
                    print('iter over; iterated {} times'.format(str(one_iter+1)))
                    break
                else:
                    last_diverse_all_tags=diverse_all_tags

            # 更新训练/eval data loader 根据list数据重新init
            train_data_loader.reprocess_t0_data(data_lst=train_data_list)
            # train_data_loader.initial_data_loader_process_and_save()
            # train_data_loader.initial_data_loader_load()
            train_data_loader.initial_data_loader()

            val_data_loader.reprocess_t0_data(data_lst=test_data_list)
            # val_data_loader.initial_data_loader_process_and_save()
            # val_data_loader.initial_data_loader_load()
            val_data_loader.initial_data_loader()
            
            

if __name__ == '__main__':
    main()

