import logging
import statistics
from statistics import median

import torch
import json, time, sys, csv, copy, math
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import convert_list_to_dict, ROOT_PATH
from discovery_score import compute_sf, get_stop_words
import xlsxwriter, sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import argmax, sqrt
from collections import OrderedDict


def precision(rank, ground_truth):
    # Precision is meaningless when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / np.arange(1, len(rank) + 1)
    return result


def recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    # relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    relevant_num = np.cumsum([min(idx + 1, len(ground_truth)) for idx, _ in enumerate(rank)])
    result = [p / r_num if r_num != 0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg / idcg
    return result

def preprocess_before_compute_hr(predicted_label, ref_labels):
    final_score_dict_sort = ref_labels[0]
    final_score_labels = list(final_score_dict_sort.keys())
    predicted_label_tmp = predicted_label.view(predicted_label.size()[0]).tolist()
    for k in range(len(predicted_label_tmp)):
        final_score_dict_sort[final_score_labels[k]].append(predicted_label_tmp[k])
        # final_score_dict_sort[final_score_labels[k]][0] = final_score_dict_sort[final_score_labels[k]][0] + \
        #                                                  final_score_dict_sort[final_score_labels[k]][-1]
        final_score_dict_sort[final_score_labels[k]][0] = final_score_dict_sort[final_score_labels[k]][4]
    final_score_dict_sort = {k: v for k, v in sorted(final_score_dict_sort.items(),
                                                     key=lambda item: item[1][0],
                                                     reverse=True)}
    predicted_label_tmp = list(final_score_dict_sort.keys())
    return predicted_label_tmp, ref_labels[1]


def compute_hr(predicted_label, predicted_label_score, ref_labels_list, ref_labels_score_list, k_list, result_dict):
    '''
    final_score_dict_sort = ref_labels[0]
    final_score_labels = list(final_score_dict_sort.keys())
    predicted_label_tmp = predicted_label.view(predicted_label.size()[0]).tolist()
    for k in range(len(predicted_label_tmp)):
        final_score_dict_sort[final_score_labels[k]].append(predicted_label_tmp[k])
        #final_score_dict_sort[final_score_labels[k]][0] = final_score_dict_sort[final_score_labels[k]][0] + \
        #                                                  final_score_dict_sort[final_score_labels[k]][-1]
        final_score_dict_sort[final_score_labels[k]][0] = final_score_dict_sort[final_score_labels[k]][4]
    final_score_dict_sort = {k: v for k, v in sorted(final_score_dict_sort.items(),
                                                     key=lambda item: item[1][0],
                                                     reverse=True)}
    predicted_label_tmp = list(final_score_dict_sort.keys())
    '''
    ref_label_search_dict = {}
    hr_k_list, precision_k_list, recall_k_list, f1_k_list = [], [], [], []
    for i in range(len(ref_labels_list)):
        ref_label_search_dict[ref_labels_list[i]] = 1
    for one_k in k_list:
        if one_k == "N":
            k = len(ref_labels_list)
        else:
            k = one_k
        # compute hr given k
        # sort and extract topk
        # convert score to 0 and 1
        if type(k) is int:
            predicted_label_tmp2 = predicted_label[0:k]
        elif type(k) is float:
            predicted_label_tmp2 = predicted_label
        else:
            predicted_label_tmp2 = predicted_label

        map_k = map(rank=predicted_label_tmp2, ground_truth=ref_label_search_dict)[-1] if len(predicted_label_tmp2) > 0 else 0.0
        ndcg_k = ndcg(rank=predicted_label_tmp2, ground_truth=ref_label_search_dict)[-1] if len(predicted_label_tmp2) > 0 else 0.0
        hits_list = [1 if one_pred_label in ref_label_search_dict else 0 for one_pred_label in predicted_label_tmp2] if len(predicted_label_tmp2) > 0 else [0]
        # compute tp@k
        #precision_score_k = metrics.precision_score(predicted_label_score[0:k], ref_labels_score_list[0:k])
        #recall_score_k = metrics.recall_score(predicted_label_score[0:k], ref_labels_score_list[0:k])
        #f1_score_k = metrics.f1_score(predicted_label_score[0:k], ref_labels_score_list[0:k])
        acc_k = float(sum(hits_list))
        hr_k = float(acc_k) / len(ref_labels_list)
        tp_k = float(sum(hits_list))
        fp_k = float(sum(
            [1 if hits_list[i] == 0 else 0 for i in range(len(hits_list))]))
        fn_k = float(len(ref_labels_list) - tp_k)

        #precision_k = tp_k / len(predicted_label_tmp2) if len(predicted_label_tmp2) > 0 else 0.0
        #recall_k = tp_k / len(ref_labels_list)
        #f1_k = ((2 * precision_k * recall_k) / (precision_k + recall_k)) if (precision_k + recall_k) > 0 else 0
        hr_k_list.append(hr_k)
        #precision_k_list.append(precision_k)
        #recall_k_list.append(recall_k)
        #f1_k_list.append(f1_k)
        result_dict[one_k]["hr_k"].append((acc_k, len(ref_labels_list)))
        result_dict[one_k]["map_k"].append(map_k)
        result_dict[one_k]["ndcg_k"].append(ndcg_k)
    return result_dict



def hr_related_evaluation(predicted_label, predicted_labels_scores_list,
                          ref_labels, ref_labels_score_list, model_threshold,
                          k_list, model_name, data_type, logging, what_best,
                          threshold_all_half=1.0, topk=10):
    result_dict = {}
    final_result_dict = {}
    for k in k_list:
        result_dict[k] = {"hr_k": [], "precision_k": [], "recall_k": [], "f1_k": [], "map_k": [], "ndcg_k": []}
    total_count = len(predicted_label)
    predicted_labels_scores_list_all = []
    ref_labels_score_list_all = []
    predicted_labels_scores_list_for_auc = []
    ref_labels_score_list_for_auc = []
    predicted_label_for_map = []
    ref_labels_for_map = []
    for i in range(len(predicted_labels_scores_list)):
        predicted_labels_scores_list_all += predicted_labels_scores_list[i][0:int(len(predicted_labels_scores_list[i])*threshold_all_half)]
        ref_labels_score_list_all += ref_labels_score_list[i][0:int(len(predicted_labels_scores_list[i])*threshold_all_half)]
        predicted_labels_scores_list_for_auc += predicted_labels_scores_list[i][0:topk]
        ref_labels_score_list_for_auc += ref_labels_score_list[i][0:topk]
        predicted_label_for_map += predicted_label[i][0:topk]
        ref_labels_for_map += ref_labels[i][0:topk]

    '''
    upper_score = max(predicted_labels_scores_list_all)
    lower_score = min(predicted_labels_scores_list_all)
    median_score = statistics.median(predicted_labels_scores_list_all)
    step = (2*median_score - lower_score) / 100
    score = lower_score
    f1_score_list = []
    threshold_list = []
    while score < upper_score:
        predicted_labels_scores_list_all_binary = [1 if predicted_labels_scores_list_all[j] >= score else 0 for j in range(len(predicted_labels_scores_list_all))]
        f1_score = metrics.f1_score(ref_labels_score_list_all, predicted_labels_scores_list_all_binary)
        f1_score_list.append(f1_score)
        threshold_list.append(score)
        score += step
    max_f1_score = max(f1_score_list)
    index = f1_score_list.index(max_f1_score)
    best_threshold = threshold_list[index]
    plt.plot(
        threshold_list,
        f1_score_list,
        color="darkorange",
        lw=2,
        label="F1 score given threshold",
    )
    plt.scatter(threshold_list[index], f1_score_list[index], marker='o', color='black', label='Best')
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, upper_score])
    plt.ylim([0.0, max_f1_score + 0.05])
    plt.xlabel("Thresholds")
    plt.ylabel("F1 Score")
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("F1 Score {}.png".format(model_name))
    '''
    '''
    json.dump(predicted_labels_scores_list_all, open("{}predicted_labels_scores_list_all".format(model_name), "w", encoding='utf-8'))
    json.dump(ref_labels_score_list_all, open("{}ref_labels_score_list_all".format(model_name), "w", encoding='utf-8'))
    fpr, tpr, thresholds = metrics.roc_curve(ref_labels_score_list_for_auc, predicted_labels_scores_list_for_auc)
    J = tpr - fpr
    ix = argmax(J)
    best_threshold = thresholds[ix]
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.savefig("AUC {}.png".format(model_name))
    #plt.show()
    '''
    factor = sum(ref_labels_score_list_for_auc) / sum(ref_labels_score_list_all)

    precision, recall, thresholds = metrics.precision_recall_curve(ref_labels_score_list_for_auc, predicted_labels_scores_list_for_auc, pos_label=1)
    recall = recall*factor
    fscore = (2 * precision * recall) / (precision + recall)
    fscore_remove_nan = np.nan_to_num(fscore)
    # locate the index of the largest f score
    if what_best == "f1":
        ix = argmax(fscore_remove_nan)
    elif what_best == "recall":
        ix = argmax(recall)
    elif what_best == "precision":
        ix = argmax(precision)
    else:
        raise ValueError

    print("best_{}_threshold: {}".format(what_best, thresholds[ix]))
    best_threshold = thresholds[ix] if model_threshold is None else model_threshold
    print("for model {}".format(model_name))
    if model_threshold is None:
        print('Best Threshold=%.5f, F-Score=%.5f, Precision, Recall' % (best_threshold, fscore_remove_nan[ix]), precision[ix], recall[ix])
    else:
        print('Adopt Threshold from Validation=%.5f' % best_threshold)


    # plt.plot(recall, precision, marker='.', label=model_name)
    # plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # # axis labels
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # # show the plot
    # #plt.show()
    # plt.savefig("F1 Score {}.png".format(model_name))
    #logging.info("\n")
    for i in range(len(predicted_label)):
        result_dict = compute_hr(
            predicted_label[i],
            predicted_labels_scores_list[i],
            ref_labels[i],
            ref_labels_score_list[i],
            k_list,
            result_dict
        )
    for one_k in result_dict:
        predicted_labels_scores_list_topk = []
        predicted_labels_list_topk = []
        ref_labels_score_list_topk = []
        k = one_k
        song_new_label_number = []
        for i in range(len(predicted_labels_scores_list)):
            if k == "N":
                k = len(predicted_labels_scores_list[i])
            predicted_labels_scores_list_topk += predicted_labels_scores_list[i][0:k]
            ref_labels_score_list_topk += ref_labels_score_list[i][0:k]
            predicted_labels_list_topk += predicted_label[i][0:k]
            predicted_labels_scores_list_topk_binary_one_song = [1 if one_score >= best_threshold else 0 for one_score in
                                                                 predicted_labels_scores_list[i][0:k]]
            song_new_label_number.append(sum(predicted_labels_scores_list_topk_binary_one_song))
        predicted_labels_scores_list_topk_binary = [1 if one_score >= best_threshold else 0 for one_score in
                                                   predicted_labels_scores_list_topk]
        #song_new_label_number = []
        #for jj in range(0, len(predicted_labels_scores_list_topk_binary), k):
        #    song_new_label_number.append(sum(predicted_labels_scores_list_topk_binary[jj:jj+k]))
        new_labels = []
        for jj in range(len(predicted_labels_scores_list_topk_binary)):
            if predicted_labels_scores_list_topk_binary[jj]:
                new_labels.append(predicted_labels_list_topk[jj])
        new_labels_set = set(new_labels)
        existing_labels = []
        for one_ref in ref_labels:
            existing_labels += one_ref
        existing_labels_set = set(existing_labels)
        over_lap_labels = new_labels_set & existing_labels_set
        newly_added_labels = new_labels_set - existing_labels_set

        precision_score = metrics.precision_score(ref_labels_score_list_topk, predicted_labels_scores_list_topk_binary)
        true_positive = [1 if ref_labels_score_list_topk[i] == 1 and predicted_labels_scores_list_topk_binary[i] == 1 else 0 for i in range(len(ref_labels_score_list_topk))]
        recall_score = float(sum(true_positive)) / sum(ref_labels_score_list_all)
        #recall_score = metrics.recall_score(ref_labels_score_list_all, predicted_labels_scores_list_topk_binary)
        #f1_score = metrics.f1_score(ref_labels_score_list_all, predicted_labels_scores_list_topk_binary)
        positive_number_above_threshold_average = sum(predicted_labels_scores_list_topk_binary) / len(predicted_labels_scores_list)

        print("average positive number above the threshold: {}".format(positive_number_above_threshold_average))
        print("extracted labels numbers: median:{}, min:{}, max:{}".format(median(song_new_label_number),
                                                                                  min(song_new_label_number),
                                                                                  max(song_new_label_number), ))
        logging.info("average positive number above the threshold: {}".format(positive_number_above_threshold_average))
        logging.info("extracted labels numbers: median:{}, min:{}, max:{}".format(median(song_new_label_number),
                                                                                 min(song_new_label_number),
                                                                                 max(song_new_label_number)))
        logging.info("unique new labels numbers: {},\n unique existing labels numbers: {}, \n"
                    "unique overlap between new labels and existing labels: {}, \n"
                    "extend existing labels: {}".format(len(new_labels_set),
                                                       len(existing_labels_set),
                                                       len(over_lap_labels),
                                                        len(newly_added_labels)))

        f1_score = ((2*precision_score*recall_score) / (precision_score + recall_score)) if (precision_score + recall_score) != 0 else 0
        hr_k_acc = sum([one_song[0] for one_song in result_dict[one_k]["hr_k"]])
        hr_k_total = sum([one_song[1] for one_song in result_dict[one_k]["hr_k"]])
        hr_k_average = round(hr_k_acc / hr_k_total, 5)
        precision_k = round(precision_score, 5)
        recall_k = round(recall_score, 5)
        f1_k = round(f1_score, 5)
        map_k_average = np.round(np.average(np.array(result_dict[one_k]["map_k"])), 5)
        ndcg_k_average = np.round(np.average(np.array(result_dict[one_k]["ndcg_k"])), 5)
        # map_k_average = sum(result_dict[one_k]["map_k"]) / total_count
        print(
            'K {} | hr_k {:8.5f} | precision_k {:8.5f} | recall_k {:8.5f} | f1_k {:8.5f} | map_k {:8.5f} | ndcg {:8.5f}'.format(
                one_k,
                hr_k_average,
                precision_k,
                recall_k,
                f1_k,
                map_k_average,
                ndcg_k_average
            ))

        logging.info(
            'K {} | hr_k {:8.5f} | precision_k {:8.5f} | recall_k {:8.5f} | f1_k {:8.5f} | map_k {:8.5f} | ndcg {:8.5f}'.format(
                one_k,
                hr_k_average,
                precision_k,
                recall_k,
                f1_k,
                map_k_average,
                ndcg_k_average
            )
        )
        final_result_dict[one_k] = {}
        final_result_dict[one_k]["hr_k"] = hr_k_average
        final_result_dict[one_k]["precision_k"] = precision_k
        final_result_dict[one_k]["recall_k"] = recall_k
        final_result_dict[one_k]["f1_k"] = f1_k
        final_result_dict[one_k]["map_k"] = map_k_average
        final_result_dict[one_k]["ndcg_k"] = ndcg_k_average
    return final_result_dict, [thresholds, precision, recall, fscore_remove_nan, best_threshold]


def get_threshold(predicted_label, predicted_labels_scores_list,
                          ref_labels, ref_labels_score_list, model_name, topk, what_best):

    predicted_labels_scores_list_all = []
    ref_labels_score_list_all = []
    predicted_labels_scores_list_for_auc = []
    ref_labels_score_list_for_auc = []
    predicted_label_for_map = []
    ref_labels_for_map = []
    for i in range(len(predicted_labels_scores_list)):
        predicted_labels_scores_list_all += predicted_labels_scores_list[i]
        ref_labels_score_list_all += ref_labels_score_list[i]
        predicted_labels_scores_list_for_auc += predicted_labels_scores_list[i][0:topk]
        ref_labels_score_list_for_auc += ref_labels_score_list[i][0:topk]
        predicted_label_for_map += predicted_label[i][0:topk]
        ref_labels_for_map += ref_labels[i][0:topk]

    factor = sum(ref_labels_score_list_for_auc) / sum(ref_labels_score_list_all)

    precision, recall, thresholds = metrics.precision_recall_curve(ref_labels_score_list_for_auc, predicted_labels_scores_list_for_auc, pos_label=1)
    recall = recall*factor
    fscore = (2 * precision * recall) / (precision + recall)
    fscore_remove_nan = np.nan_to_num(fscore)
    # locate the index of the largest f score
    if what_best == "f1":
        ix = argmax(fscore_remove_nan)
    elif what_best == "recall":
        ix = argmax(recall)
    elif what_best == "precision":
        ix = argmax(precision)
    else:
        raise ValueError
    #ix = argmax(recall)
    best_threshold = thresholds[ix]

    return [thresholds, precision, recall, fscore_remove_nan, best_threshold]

def get_annotation_data(json_data_file_path, annotation_folder, annotation_index2song_id_file):

    annotation_index2song_id = json.load(open(annotation_index2song_id_file, "r", encoding='utf-8'))
    if type(json_data_file_path) is not list:
        annotation_data_list = json.load(open(json_data_file_path, "r", encoding="utf-8"))
    else:
        annotation_data_list = json_data_file_path
    annotation_data_dict = convert_list_to_dict(annotation_data_list)
    all_files = [f for f in listdir(annotation_folder) if isfile(join(annotation_folder, f))]
    print(len(all_files))
    annotation_data_list_new = []
    for one_file in all_files:
        index, file_type = one_file.split("_")
        index = int(index) - 3000
        song_id = annotation_index2song_id[index]
        annotation_dict = extract_annotation_content(join(annotation_folder, one_file))
        annotation_content = annotation_data_dict[song_id]
        annotation_content["annotation_dict"] = annotation_dict
        annotation_data_list_new.append(annotation_content)
    return annotation_data_list_new


def extract_annotation_content(file_path):
    ref_labels = []
    not_labels = []
    with open(file_path, "r", newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        for row in csvreader:
            if count == 0:
                # skip first line (head)
                count += 1
                continue
            label, is_ref_label = row[0].rsplit(",", 1)
            if int(is_ref_label) == 1:
                ref_labels.append(label)
            else:
                not_labels.append(label)
            count += 1
    return {"ref_labels": ref_labels, "not_labels": not_labels}

def evaluate_by_iterations(iteration, k_list, what_best):
    workbook = xlsxwriter.Workbook('evaluation_results_iterations.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for i in range(iteration[0], iteration[1]):
        worksheet.write(row, col+2+i-iteration[0], "iteration={}".format(str(i)))
    for k in range(iteration[0], iteration[1]):
        annotation_data_list = get_annotation_data(
            json_data_file_path=join("data", "top1000_0.7_annotation_t{}.json".format(str(k))),
            annotation_folder=join(ROOT_PATH, "20220213_top200_1000_songs"),
            annotation_index2song_id_file=join(ROOT_PATH, "index2song_id_top200_1000samples.json")
            )

        # normalized tf-idf is in col 5
        predicted_labels_list = []
        ref_labels_list = []
        for i in range(len(annotation_data_list)):
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(annotation_data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][3] + item[1][4] + item[1][6],
                                                          reverse=True)}
            # item[1][3]+item[1][4]+
            # print(final_score_dict_sort_tf_idf_sorted)
            predicted_labels_list.append([one_key for one_key in final_score_dict_sort_tf_idf_sorted])
            ref_labels_list.append(annotation_data_list[i]['annotation_dict']['ref_labels'])
        print("processing iter: {}".format(str(k)))
        final_results = hr_related_evaluation(predicted_label=predicted_labels_list,
                                              ref_labels=ref_labels_list,
                                              k_list=k_list, what_best=what_best)
        if k == iteration[0]:
            for one_k in final_results:
                row += 1
                worksheet.write(row, col, "k={}".format(str(one_k)))
                col += 1
                for one_metric in final_results[one_k]:
                    worksheet.write(row, col, "{}".format(one_metric))
                    worksheet.write(row, col+1+k-iteration[0], round(final_results[one_k][one_metric], 3))
                    row += 1
                row -= 1
                col -= 1
        else:
            row = 1
            col = 1+1+k-iteration[0]
            for one_k in final_results:
                for one_metric in final_results[one_k]:
                    worksheet.write(row, col, round(final_results[one_k][one_metric], 3))
                    row += 1
    workbook.close()

def evaluate_among_multiple_results(annotation_file_list, model_name_list, k_list, what_best):
    workbook = xlsxwriter.Workbook('evaluation_multiple_results.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for i in range(len(model_name_list)):
        worksheet.write(row, col + 2 + i, "{}".format(model_name_list[i]))
    for k in range(len(annotation_file_list)):
        annotation_data_list = get_annotation_data(
            json_data_file_path=join("data", annotation_file_list[k]),
            annotation_folder=join(ROOT_PATH, "20220213_top200_1000_songs"),
            annotation_index2song_id_file=join(ROOT_PATH, "index2song_id_top200_1000samples.json")
        )

        # normalized tf-idf is in col 5
        predicted_labels_list = []
        ref_labels_list = []
        for i in range(len(annotation_data_list)):

            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(annotation_data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][3] + item[1][4] + item[1][6],
                                                          reverse=True)}
            # item[1][3]+item[1][4]+
            # print(final_score_dict_sort_tf_idf_sorted)
            predicted_labels_list.append([one_key for one_key in final_score_dict_sort_tf_idf_sorted])
            ref_labels_list.append(annotation_data_list[i]['annotation_dict']['ref_labels'])
        print("processing file: {}".format(model_name_list[k]))
        final_results = hr_related_evaluation(predicted_label=predicted_labels_list,
                                              ref_labels=ref_labels_list,
                                              k_list=k_list, what_best=what_best)
        if k == 0:
            for one_k in final_results:
                row += 1
                worksheet.write(row, col, "k={}".format(str(one_k)))
                col += 1
                for one_metric in final_results[one_k]:
                    worksheet.write(row, col, "{}".format(one_metric))
                    worksheet.write(row, col + 1 + k, round(final_results[one_k][one_metric], 3))
                    row += 1
                row -= 1
                col -= 1
        else:
            row = 1
            col = 1 + 1 + k
            for one_k in final_results:
                for one_metric in final_results[one_k]:
                    worksheet.write(row, col, round(final_results[one_k][one_metric], 3))
                    row += 1

    workbook.close()

def re_statistic_silver_labels(file_path_list, model_name_list):
    workbook = xlsxwriter.Workbook('statistic_silver_label.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    assert len(file_path_list) == len(model_name_list)
    for k in range(len(file_path_list)):
        file_path = file_path_list[k]
        file_content = json.load(open(file_path, "r", encoding='utf-8'))
        original_label_system_tmp = []
        for i in range(len(file_content)):
            original_label_system_tmp += file_content[i]['song_labels']
        original_label_system_set = set(original_label_system_tmp)
        unique_silver_label_added_in_previous_iter_set = set([])
        statistics_names_list = ["song_has_new_silver_label",
                                 "total_new_silver_label_added",
                                 "unique_silver_label_added_in_this_iter",
                                 "unique_silver_label_expand_label_system"]
        if k == 0:  # first file, write the row name
            for j in range(len(statistics_names_list)):
                worksheet.write(row, col, "{}".format(statistics_names_list[j]))
                for model_name_index in range(len(model_name_list)):
                    row += 1
                    model_name = model_name_list[model_name_index]
                    worksheet.write(row, col, "{}".format(model_name))
                row += 2
        for iter_index in range(11):
            songs_have_silver_labels = 0
            total_new_silver_label_added = []

            for i in range(len(file_content)):
                current_silver_labels = file_content[i]['song_silver_label_details'][str(iter_index)]
                if iter_index == 0:
                    previous_silver_labels = []

                else:
                    previous_silver_labels = file_content[i]['song_silver_label_details'][str(iter_index-1)]
                new_silver_labels = list(set(current_silver_labels) - set(previous_silver_labels))
                if len(new_silver_labels) > 0:
                    songs_have_silver_labels += 1
                    total_new_silver_label_added += new_silver_labels

            unique_silver_label_added_in_this_iter_set = set(total_new_silver_label_added)

            unique_silver_label_expand_label_system = list(unique_silver_label_added_in_this_iter_set -
                                                           unique_silver_label_added_in_previous_iter_set -
                                                           original_label_system_set)

            print("processing iteration: {}".format(str(iter_index)))
            print("song_has_new_silver_label: {}".format(str(songs_have_silver_labels)))
            print("total_new_silver_label_added: {}".format(str(len(total_new_silver_label_added))))
            print("unique_silver_label_added_in_this_iter: {}".format(str(len(unique_silver_label_added_in_this_iter_set))))
            print("unique_silver_label_expand_label_system: {}".format(str(len(unique_silver_label_expand_label_system))))
            print("\n")
            statistics_results_list = [songs_have_silver_labels,
                                       len(total_new_silver_label_added),
                                       len(unique_silver_label_added_in_this_iter_set),
                                       len(unique_silver_label_expand_label_system)]
            row = 0
            col = 1+iter_index
            worksheet.write(row, col, iter_index+1)
            row = 1 + k
            for j in range(len(statistics_results_list)):
                one_result = statistics_results_list[j]
                worksheet.write(row + j * (len(model_name_list) + 2), col, one_result)


            unique_silver_label_added_in_previous_iter_set = set(list(unique_silver_label_added_in_this_iter_set) + \
                                                             list(unique_silver_label_added_in_previous_iter_set))
    workbook.close()
    print("debug")

def compare_tf_idf_rf_idf(file_path, k=50):
    workbook = xlsxwriter.Workbook('compare_tf_idf_rf_idf.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    annotation_data_list = get_annotation_data(json_data_file_path=file_path,
                                               annotation_folder=join(ROOT_PATH, "20220213_top200_1000_songs"),
                                               annotation_index2song_id_file=join(ROOT_PATH,
                                                                                  "index2song_id_top200_1000samples.json")
                                               )
    tf_idf_labels_list = []
    rf_idf_labels_list = []
    ref_labels_list = []
    for i in range(len(annotation_data_list)):
        final_score_dict_sort_tf_idf_sorted = {k: v for k, v in sorted(annotation_data_list[i]['final_score_dict_sort'].items(),
                                 key=lambda item: item[1][5],
                                 reverse=True)}
        #print(final_score_dict_sort_tf_idf_sorted)
        tf_idf_labels_list.append([one_key for one_key in final_score_dict_sort_tf_idf_sorted][0:k])

        final_score_dict_sort_tf_idf_sorted_2 = {k: v for k, v in
                                               sorted(annotation_data_list[i]['final_score_dict_sort'].items(),
                                                      key=lambda item: item[1][4],
                                                      reverse=True)}
        rf_idf_labels_list.append([one_key for one_key in final_score_dict_sort_tf_idf_sorted_2][0:k])

        ref_labels_list.append(annotation_data_list[i]['annotation_dict']['ref_labels'])

    for i in range(len(ref_labels_list)):
        col = 0
        worksheet.write(row, col, "ref labels")
        for j in range(len(ref_labels_list[i])):
            col += 1
            worksheet.write(row, col, ref_labels_list[i][j])
        row += 1
        col = 0
        worksheet.write(row, col, "rf-idf match labels")
        worksheet.write(row+1, col, "rf-idf not match labels")
        rf_idf_match_list = list(set(rf_idf_labels_list[i]) & set(ref_labels_list[i]))
        rf_idf_not_match_list = list(set(rf_idf_labels_list[i]) - set(ref_labels_list[i]))
        for j in range(len(rf_idf_match_list)):
            col += 1
            worksheet.write(row, col, rf_idf_match_list[j])
        col = 0
        row += 1
        for j in range(len(rf_idf_not_match_list)):
            col += 1
            worksheet.write(row, col, rf_idf_not_match_list[j])

        row += 1
        col = 0
        worksheet.write(row, col, "tf-idf match labels")
        worksheet.write(row + 1, col, "tf-idf not match labels")
        tf_idf_match_list = list(set(tf_idf_labels_list[i]) & set(ref_labels_list[i]))
        tf_idf_not_match_list = list(set(tf_idf_labels_list[i]) - set(ref_labels_list[i]))
        for j in range(len(tf_idf_match_list)):
            col += 1
            worksheet.write(row, col, tf_idf_match_list[j])
        col = 0
        row += 1
        for j in range(len(tf_idf_not_match_list)):
            col += 1
            worksheet.write(row, col, tf_idf_not_match_list[j])
        row+=1

        tf_match_sf_not_match = list(set(tf_idf_match_list) - set(rf_idf_match_list))
        #for one_word in tf_match_sf_not_match:
        sf_match_and_tf_match_sf_not_match_details = {}
        for one_review_id in annotation_data_list[i]['song_comments_detail_final']:
            for one_sent_id in annotation_data_list[i]['song_comments_detail_final'][one_review_id]:
                if one_sent_id not in 'likecnt_weight':
                    for one_review_word in annotation_data_list[i]['song_comments_detail_final'][one_review_id][one_sent_id]['song_view_segmented']:
                        if one_review_word in rf_idf_match_list+tf_match_sf_not_match:
                            if one_review_word not in sf_match_and_tf_match_sf_not_match_details:
                                sf_match_and_tf_match_sf_not_match_details[one_review_word] = {}
                                sf_match_and_tf_match_sf_not_match_details[one_review_word]["tf"] = \
                                    annotation_data_list[i]['final_score_dict_sort'][one_review_word][5]/annotation_data_list[i]['final_score_dict_sort'][one_review_word][2]
                                sf_match_and_tf_match_sf_not_match_details[one_review_word]["sf"] = \
                                    annotation_data_list[i]['final_score_dict_sort'][one_review_word][1]
                                sf_match_and_tf_match_sf_not_match_details[one_review_word]["our_idf"] = annotation_data_list[i]['final_score_dict_sort'][one_review_word][4]/annotation_data_list[i]['final_score_dict_sort'][one_review_word][1]
                                sf_match_and_tf_match_sf_not_match_details[one_review_word]["original_idf"] = \
                                    annotation_data_list[i]['final_score_dict_sort'][one_review_word][2]
                            sf_match_and_tf_match_sf_not_match_details[one_review_word][" ".join(annotation_data_list[i]['song_comments_detail_final'][one_review_id][one_sent_id]['song_view_segmented'])] = annotation_data_list[i]['song_comments_detail_final'][one_review_id]['likecnt_weight']
        sf_match_details = {}
        for one_word in rf_idf_match_list:
            sf_match_details[one_word] = sf_match_and_tf_match_sf_not_match_details[one_word]
        tf_match_sf_not_match_details = {}
        for one_word in tf_match_sf_not_match:
            tf_match_sf_not_match_details[one_word] = sf_match_and_tf_match_sf_not_match_details[one_word]

        col = 0
        worksheet.write(row, col, "tf_match_sf_not_match_details")
        col += 1
        for one_word in tf_match_sf_not_match_details:
            one_line_string = one_word + "\n"
            for one_key in tf_match_sf_not_match_details[one_word]:
                one_line_string += " ".join([one_key[0:15], str(tf_match_sf_not_match_details[one_word][one_key])])
                one_line_string += "\n"
            worksheet.write(row, col, one_line_string)
            col += 1
        row += 1
        col = 0
        worksheet.write(row, col, "sf_match_details")
        col += 1
        for one_word in sf_match_details:
            one_line_string = one_word + "\n"
            for one_key in sf_match_details[one_word]:
                one_line_string += " ".join([one_key[0:20], str(sf_match_details[one_word][one_key])])
                one_line_string += "\n"
            worksheet.write(row, col, one_line_string)
            col += 1

        row += 2
    workbook.close()

    return rf_idf_labels_list, tf_idf_labels_list, ref_labels_list

def diversity_of_silver_labels(folder_path):
    for iter_index in range(1, 12): # 1-11
        file_get_match_score = json.load(open(join("")))
        file_get_valid_score = json.load(open(join("")))

    print("debug")


def re_compute_sf_tf_idf_score(data_list, remove_overlap):
    # compute sf , our-idf, and original-idf first
    all_rf_dict_list = []
    all_tf_dict_list = []
    word_df = {}
    frq_stop_words, specific_words_dict = get_stop_words()
    stop_words_dict = {}
    for one_word in frq_stop_words:
        stop_words_dict[one_word] = 1
    for i in range(len(data_list)):
        one_song_dict, rf_dict = compute_sf(data_list[i]["song_comments_detail_final"],
                                            stop_words_dict=stop_words_dict, remove_overlap=remove_overlap)
        all_rf_dict_list.append(rf_dict)
        for one_word in rf_dict:
            if one_word in word_df:
                word_df[one_word] += 1
            else:
                word_df[one_word] = 1
        # compute tf
        tf_freq = {}
        for comment_id in data_list[i]['song_comments_detail_final']:
            for one_sent_id in data_list[i]['song_comments_detail_final'][comment_id]:
                if one_sent_id != 'likecnt_weight':
                    for one_word in data_list[i]['song_comments_detail_final'][comment_id][one_sent_id]['song_view_segmented']:
                        if one_word not in stop_words_dict and len(one_word) > 1:
                            if one_word not in tf_freq:
                                tf_freq[one_word] = 1
                            else:
                                tf_freq[one_word] += 1
        all_tf_dict_list.append(tf_freq)
    all_doc_diversity_score_dict = {}
    assert len(all_tf_dict_list) == len(all_rf_dict_list)
    # compute rf, idf, t-idf
    for i in range(len(data_list)):
        final_score_dict = {}
        song_name = data_list[i]['song_name']
        rf_dict = all_rf_dict_list[i]
        tf_dict = all_tf_dict_list[i]
        for one_word in rf_dict:
            if one_word not in stop_words_dict:
                our_idf_score = min([math.log(len(data_list)/(word_df[one_word] + 1)), math.log(word_df[one_word] + 1)]) # t-idf
                original_idf_score = math.log(len(data_list) / (word_df[one_word] + 1)) # idf
                rf_score = rf_dict[one_word]["rf_score"]
                tf_score = tf_dict[one_word]
                score = rf_score * our_idf_score
                final_score_dict[one_word] = [score, rf_score, tf_score, our_idf_score, original_idf_score]
        all_doc_diversity_score_dict[song_name] = final_score_dict
    return all_doc_diversity_score_dict

def evaluate_sf_tf_idf_score(annotation_file_path, train_test_annotation_file_path_list, normalize=False, remove_overlap=False,
                             save_file1=join("data", "top1000_0.7_annotation_t10_evaluate_tf_rf_original_idf.json"),
                             save_file2=join("data", "top1000_0.7_annotation_t10_evaluate_tf_rf_our_idf.json")):
    data_list = []
    for one_file in train_test_annotation_file_path_list:
        one_data_list = json.load(open(one_file))
        data_list += one_data_list
    all_doc_diversity_score_dict = re_compute_sf_tf_idf_score(data_list, remove_overlap=remove_overlap)
    annotation_file_content = json.load(open(annotation_file_path, "r", encoding='utf-8'))
    for i in range(len(annotation_file_content)):
        song_name = annotation_file_content[i]["song_name"]
        doc_score_dict = all_doc_diversity_score_dict[song_name]
        rf_original_idf = []
        tf_original_idf = []
        for one_word in annotation_file_content[i]['final_score_dict_sort']:
            score, rf_score, tf_score, our_idf_score, original_idf_score = doc_score_dict[one_word]
            annotation_file_content[i]['final_score_dict_sort'][one_word][1] = rf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][2] = original_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][4] = rf_score * original_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][5] = tf_score * original_idf_score
            rf_original_idf.append(rf_score * original_idf_score)
            tf_original_idf.append(tf_score * original_idf_score)
        if normalize:
            max_rf_original_idf = max(rf_original_idf)
            min_rf_original_idf = min(rf_original_idf)
            max_tf_original_idf = max(tf_original_idf)
            min_tf_original_idf = min(tf_original_idf)

            for one_word in annotation_file_content[i]['final_score_dict_sort']:
                annotation_file_content[i]['final_score_dict_sort'][one_word][4] = (annotation_file_content[i]['final_score_dict_sort'][one_word][4] -
                                                                                    min_rf_original_idf) / (max_rf_original_idf - min_rf_original_idf)
                annotation_file_content[i]['final_score_dict_sort'][one_word][5] = (annotation_file_content[i]['final_score_dict_sort'][one_word][5] -
                                                                                    min_tf_original_idf) / (max_tf_original_idf - min_tf_original_idf)

    json.dump(annotation_file_content, open(save_file1, "w", encoding='utf-8'))
    for i in range(len(annotation_file_content)):
        song_name = annotation_file_content[i]["song_name"]
        doc_score_dict = all_doc_diversity_score_dict[song_name]
        rf_our_idf = []
        tf_our_idf = []
        for one_word in annotation_file_content[i]['final_score_dict_sort']:
            score, rf_score, tf_score, our_idf_score, original_idf_score = doc_score_dict[one_word]
            annotation_file_content[i]['final_score_dict_sort'][one_word][1] = rf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][2] = our_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][4] = rf_score * our_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][5] = tf_score * our_idf_score
            rf_our_idf.append(rf_score * our_idf_score)
            tf_our_idf.append(tf_score * our_idf_score)
        if normalize:
            max_rf_our_idf = max(rf_our_idf)
            min_rf_our_idf = min(rf_our_idf)
            max_tf_our_idf = max(tf_our_idf)
            min_tf_our_idf = min(tf_our_idf)

            for one_word in annotation_file_content[i]['final_score_dict_sort']:
                annotation_file_content[i]['final_score_dict_sort'][one_word][4] = (annotation_file_content[i][
                                                                                        'final_score_dict_sort'][one_word][
                                                                                        4] -
                                                                                    min_rf_our_idf) / (
                                                                                               max_rf_our_idf - min_rf_our_idf)
                annotation_file_content[i]['final_score_dict_sort'][one_word][5] = (annotation_file_content[i][
                                                                                        'final_score_dict_sort'][one_word][
                                                                                        5] -
                                                                                    min_tf_our_idf) / (
                                                                                               max_tf_our_idf - min_tf_our_idf)

    json.dump(annotation_file_content,
              open(save_file2, "w", encoding='utf-8'))


def get_sf_tf_idf_score(annotation_file_path, train_test_annotation_file_path_list, remove_overlap=False):
    data_list = []
    for one_file in train_test_annotation_file_path_list:
        one_data_list = json.load(open(one_file))
        data_list += one_data_list
    all_doc_diversity_score_dict = re_compute_sf_tf_idf_score(data_list, remove_overlap=remove_overlap)
    annotation_file_content = json.load(open(annotation_file_path, "r", encoding='utf-8'))
    tf_idf_all = []
    rf_idf_all = []
    valida_score_all = []
    for i in range(len(annotation_file_content)):
        song_name = annotation_file_content[i]["song_name"]
        doc_score_dict = all_doc_diversity_score_dict[song_name]
        rf_our_idf = []
        tf_our_idf = []
        for one_word in annotation_file_content[i]['final_score_dict_sort']:
            score, rf_score, tf_score, our_idf_score, original_idf_score = doc_score_dict[one_word]
            rf_t_idf_score = math.tanh((rf_score * our_idf_score)/10)
            tf_t_idf_score = math.tanh((tf_score * our_idf_score)/10)
            annotation_file_content[i]['final_score_dict_sort'][one_word][1] = rf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][2] = our_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][4] = rf_t_idf_score
            annotation_file_content[i]['final_score_dict_sort'][one_word][5] = tf_t_idf_score

            #annotation_file_content[i]['final_score_dict_sort'][one_word][3] = annotation_file_content[i]['final_score_dict_sort'][one_word][3] / 2
            valida_score_all.append(annotation_file_content[i]['final_score_dict_sort'][one_word][3])
            rf_our_idf.append(rf_t_idf_score)
            tf_our_idf.append(tf_t_idf_score)
            rf_idf_all.append(rf_t_idf_score)
            tf_idf_all.append(tf_t_idf_score)
    return annotation_file_content, rf_idf_all, tf_idf_all, valida_score_all

def get_silver_labels_using_threshold(content_list, threshold):
    all_silver_labels = []
    songs_have_silver_labels = []
    for i in range(len(content_list)):
        one_song_silver_labels = []
        silver_candidate_list_sort = {k: v for k, v in
         sorted(content_list[i]['final_score_dict_sort'].items(),
                key=lambda item: item[1][3]+item[1][4]+item[1][6],
                reverse=True)}
        for one_word in silver_candidate_list_sort:
            if one_word not in content_list[i]['song_pseudo_golden_labels']:
                if (silver_candidate_list_sort[one_word][3]+
                    silver_candidate_list_sort[one_word][4]+
                    silver_candidate_list_sort[one_word][6])/3 >= threshold:
                        one_song_silver_labels.append(one_word)
        all_silver_labels += one_song_silver_labels
        if len(one_song_silver_labels) > 0:
            songs_have_silver_labels.append(content_list[i]['song_name'])
    return all_silver_labels, songs_have_silver_labels

def compare_silver_labels_using_different_threshold(files_path_list, threshold):
    print("debug")

def extract_annotation_set(training_list, original_annotation_list, save_name):
    train_dict = {}
    for i in range(len(training_list)):
        train_dict[training_list[i]["song_name"]] = training_list[i]
    annotation_list = []
    for i in range(len(original_annotation_list)):
        if original_annotation_list[i]["song_name"] in train_dict:
            annotation_list.append(train_dict[original_annotation_list[i]["song_name"]])

    annotation_data_list = get_annotation_data(
        json_data_file_path=annotation_list,
        annotation_folder=join(ROOT_PATH, "20220213_top200_1000_songs"),
        annotation_index2song_id_file=join(ROOT_PATH, "index2song_id_top200_1000samples.json")
    )
    if save_name is not None:
        json.dump(annotation_data_list, open(save_name, "w", encoding='utf-8'))
    else:
        return annotation_data_list

def get_prediction_references(data_list, data_type, model_name, what_best):
    predicted_labels_list = []
    predicted_labels_scores_list = []
    ref_labels_list = []
    ref_labels_score_list = []
    for i in range(len(data_list)):
        predicted_scores_dict = OrderedDict()
        # 200 labels provided to annotators to annotate
        if data_type == "annotation":
            ref_labels = data_list[i]['annotation_dict']['ref_labels'] + \
                               data_list[i]['annotation_dict']['not_labels']
        elif data_type == "validation":
            ref_labels = list(data_list[i]['final_score_dict_sort'].keys())
        else:
            raise ValueError
        ref_labels_dict = {}
        for one_label in ref_labels:
            ref_labels_dict[one_label] = 1
        if "tf-idf" in model_name or "tf-t-idf" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][5],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key][5]
        elif "tf" == model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: float(item[1][5])/item[1][2],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key][5] / float(final_score_dict_sort_tf_idf_sorted[one_key][2])
        elif "rf-idf" in model_name or "rf-t-idf" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][4],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                if "tanh" in model_name:
                    predicted_scores_dict[one_key] = math.tanh(final_score_dict_sort_tf_idf_sorted[one_key][4])
                else:
                    predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key][4]

        elif "textrank" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]["textrank"].items(),
                                                          key=lambda item: item[1],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key]
        elif "embeddingrank" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]["embeddingrank"].items(),
                                                          key=lambda item: item[1],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key]

        elif "self-training" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][6],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key][6]
        elif "valid_score" in model_name:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: item[1][3],
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = final_score_dict_sort_tf_idf_sorted[one_key][3]
        else:
            final_score_dict_sort_tf_idf_sorted = {k: v for k, v in
                                                   sorted(data_list[i]['final_score_dict_sort'].items(),
                                                          key=lambda item: (item[1][3] + item[1][4] + item[1][6])/3,
                                                          reverse=True)}
            for one_key in final_score_dict_sort_tf_idf_sorted:
                predicted_scores_dict[one_key] = (final_score_dict_sort_tf_idf_sorted[one_key][3] +
                                                  final_score_dict_sort_tf_idf_sorted[one_key][4] +
                                                  final_score_dict_sort_tf_idf_sorted[one_key][6]) / 3

        one_doc_predicted_labels_list = []
        one_doc_ref_labels_score_list = []
        one_doc_predicted_labels_scores_list = []
        if data_type == "annotation":
            ref_positive_labels = data_list[i]['annotation_dict']['ref_labels']
        elif data_type == "validation":
            ref_positive_labels = data_list[i]['song_pseudo_golden_labels']
        else:
            raise ValueError
        for one_key in predicted_scores_dict:
            if one_key in ref_labels_dict:
                one_doc_predicted_labels_list.append(one_key)
                one_doc_ref_labels_score_list.append(
                    1 if one_key in ref_positive_labels else 0)
                one_doc_predicted_labels_scores_list.append(predicted_scores_dict[one_key])

        predicted_labels_list.append(copy.deepcopy(one_doc_predicted_labels_list))
        predicted_labels_scores_list.append(copy.deepcopy(one_doc_predicted_labels_scores_list))

        ref_labels_list.append(copy.deepcopy(ref_positive_labels))
        ref_labels_score_list.append(copy.deepcopy(one_doc_ref_labels_score_list))
    return predicted_labels_list, predicted_labels_scores_list, ref_labels_list, ref_labels_score_list

def evaluate_results_metrics(data_file_list, model_name_list, model_threshold_list, k_list, data_type,
                            save_excel_file_name, top_k, what_best, remove_matches, one_iter, log_file_name="log_file.txt", train_eval="train"):
    assert len(data_file_list) == len(model_name_list)
    logging.basicConfig(filename=log_file_name,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    workbook = xlsxwriter.Workbook(save_excel_file_name)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    #workbook_top_k = xlsxwriter.Workbook('{}_textrank_emebeddingrank_self_training_top{}.xlsx'.format(data_type, top_k))
    workbook_top_k_detail = xlsxwriter.Workbook('{}_{}_top_{}_detail.xlsx'.format(data_type, "_".join(model_name_list[:2]), top_k))
    worksheet_top_k_detail = workbook_top_k_detail.add_worksheet()
    row_top_k_detail = 0
    col_top_k_detail = 0

    workbook_top_k = xlsxwriter.Workbook('{}_{}_top_{}.xlsx'.format(data_type, "_".join(model_name_list), top_k))
    worksheet_top_k = workbook_top_k.add_worksheet()
    row_top_k = 0
    col_top_k = 0

    results_for_threshold = {}
    for i in range(len(model_name_list)):
        worksheet.write(row, col + 2 + i, "{}".format(model_name_list[i]))
    for k in range(len(data_file_list)):
        results_for_threshold[model_name_list[k]] = []
        if data_type == "annotation":
            data_list = get_annotation_data(
                json_data_file_path=data_file_list[k],
                annotation_folder=join(ROOT_PATH, "20220213_top200_1000_songs"),
                annotation_index2song_id_file=join(ROOT_PATH, "index2song_id_top200_1000samples.json")
            )
        elif data_type == "validation":
            data_list = json.load(open(data_file_list[k], "r", encoding='utf-8'))
        else:
            raise ValueError
        model_name = model_name_list[k]
        predicted_labels_list, \
        predicted_labels_scores_list, \
        ref_labels_list, \
        ref_labels_score_list = get_prediction_references(data_list, data_type, model_name, what_best=what_best)

        if remove_matches:
            predicted_labels_list, \
            predicted_labels_scores_list, \
            ref_labels_list, \
            ref_labels_score_list, data_list = remove_previous_matched_labels(data_list,
                                                                                           [predicted_labels_list,
                                                                                            predicted_labels_scores_list,
                                                                                            ref_labels_list,
                                                                                            ref_labels_score_list,
                                                                                            ],
                                                                                           topk=top_k,
                                                                                           one_iter=one_iter,
                                                                                           train_eval=train_eval
                                                                                           )



        save_prediction_details(model_name, data_type, data_list, [predicted_labels_list, predicted_labels_scores_list, ref_labels_list, ref_labels_score_list])
        print("processing file: {}".format(model_name_list[k]))
        logging.info("processing file: {}".format(model_name_list[k]))
        final_results, precision_recall = hr_related_evaluation(predicted_label=predicted_labels_list,
                                              predicted_labels_scores_list=predicted_labels_scores_list,
                                              ref_labels=ref_labels_list,
                                              ref_labels_score_list=ref_labels_score_list,
                                              model_threshold=model_threshold_list[k],
                                              k_list=k_list,
                                              model_name=model_name_list[k],
                                              data_type=data_type,
                                            logging=logging,
                                              threshold_all_half=1,
                                              topk=top_k,
                                                what_best=what_best)
        thresholds, precision, recall, fscore_remove_nan, best_threshold = precision_recall
        precision_recall_add_map_k = [thresholds, precision, recall, fscore_remove_nan, best_threshold, final_results[top_k]["map_k"]]
        results_for_threshold[model_name_list[k]] = precision_recall_add_map_k
        write_head_line = True if k == 0 else False
        save_precision_results_detail(worksheet_top_k_detail, row_top_k_detail+k*len(thresholds), col_top_k_detail, model_name, data_type, precision_recall_add_map_k, write_head_line=write_head_line)
        row, col = save_metrics_k(worksheet, k, k_list, final_results, row, col)
        row_top_k, col_top_k = save_metrics_topk(worksheet_top_k, k, data_type, final_results, model_name, row_top_k, col_top_k, precision_recall_add_map_k)

    workbook.close()
    workbook_top_k_detail.close()
    workbook_top_k.close()
    return results_for_threshold, [predicted_labels_list, predicted_labels_scores_list, ref_labels_list, ref_labels_score_list]

def save_metrics_topk(worksheet, k, data_type, final_results, model_name, row, col, precision_recall_add_map_k,
                      head_line=("topk", "模型", "阈值", "Pecision", "Recall", "F1", "MAP", "数据")):
    thresholds, precision, recall, fscore_remove_nan, best_threshold, map_k = precision_recall_add_map_k
    if k == 0:
        for i in range(len(head_line)):
            worksheet.write(row, col+i, head_line[i])
        row += 1
    topk = list(final_results.keys())[0]
    worksheet.write(row, col, topk)
    worksheet.write(row, col + 1, model_name)
    worksheet.write(row, col + 2, best_threshold)
    worksheet.write(row, col + 3, final_results[topk]['precision_k'])
    worksheet.write(row, col + 4, final_results[topk]['recall_k'])
    worksheet.write(row, col + 5, final_results[topk]['f1_k'])
    worksheet.write(row, col + 6, final_results[topk]['map_k'])
    worksheet.write(row, col + 7, data_type)
    row += 1
    return row, col

def save_metrics_k(worksheet, k, k_list, final_results, row, col):
    if k == 0:
        row += 1
        for one_metric in final_results[list(final_results.keys())[0]]:
            worksheet.write(row, col, "{}".format(one_metric))
            for name_index in range(len(k_list)):
                one_name = k_list[name_index]
                worksheet.write(row, col + 1, "{}".format(one_name))
                row += 1
    col = 2
    final_results_key_list = list(final_results.keys())
    for kk in range(len(final_results_key_list)):
        row = 1
        for one_metric_index in range(len(final_results[final_results_key_list[kk]])):
            one_metric = list(final_results[final_results_key_list[kk]].keys())[one_metric_index]
            # row: 1+len(k_list)*one_metric_index+kk
            # col: 2 + k
            worksheet.write(row + len(k_list) * one_metric_index + kk,
                            col + k,
                            final_results[final_results_key_list[kk]][one_metric]
                            )
    return row, col

def save_precision_results_detail(worksheet, row, col, model_name, data_type, precision_recall, write_head_line=False):
    head_line = ["模型", "阈值", "Pecision", "Recall", "F1", "MAP", "数据"]
    if write_head_line:
        for i in range(len(head_line)):
            worksheet.write(row, col + i, head_line[i])
    thresholds, precision, recall, fscore_remove_nan, best_threshold, map_k = precision_recall
    for j in range(len(thresholds)):
        row += 1
        info_list = [model_name, thresholds[j], precision[j], recall[j], fscore_remove_nan[j], map_k, data_type]
        for k in range(len(info_list)):
            worksheet.write(row, col + k, info_list[k])


def find_best_threshold(results_list, what_best):
    thresholds_dict = {}
    for one_model_name in results_list:
        thresholds_dict[one_model_name] = {}
        thresholds, precision, recall, fscore_remove_nan, best_threshold, map_k = results_list[one_model_name]
        if what_best == "recall":
            ix = argmax(recall)
        elif what_best == "precision":
            ix = argmax(precision)
        elif what_best == "f1":
            ix = argmax(fscore_remove_nan)
        else:
            raise ValueError
        best_threshold = thresholds[ix]
        thresholds_dict[one_model_name]["best_threshold"] = best_threshold
    return thresholds_dict

def save_prediction_details(model_name, data_type, data_list, prediction_details):
    predicted_labels_list, predicted_labels_scores_list, ref_labels_list, ref_labels_score_list = prediction_details
    save_file_name = "prediction_detail_{}_{}.txt".format(model_name, data_type)
    save_content = ""
    for i in range(len(predicted_labels_list)):
        save_content += "song index is: {}\n".format(i)
        save_content += "song ref labels are: {}\n".format(",".join(ref_labels_list[i]))
        if data_type == "annotation":
            save_content += "song golden labels are: {}\n".format(",".join(data_list[i]["song_pseudo_golden_labels"]))
        save_content += "candidate words,        score,          positive\n"
        for j in range(len(predicted_labels_list[i])):
            save_content += "{},        {},          {}\n".format(predicted_labels_list[i][j], predicted_labels_scores_list[i][j], ref_labels_score_list[i][j])
        if i >= 100:
            break
    open(save_file_name, "w", encoding='utf-8').write(save_content)

def get_threshold_from_validation_set(data_list, save_data_file, topk, model_name, what_best, one_iter, train_eval):
    # data_list = json.load(open(data_file, "r", encoding='utf-8'))
    predicted_labels_list, \
    predicted_labels_scores_list, \
    ref_labels_list, \
    ref_labels_score_list = get_prediction_references(data_list, "validation", model_name=model_name, what_best=what_best)

    predicted_labels_list_remove_match, \
    predicted_labels_scores_list_remove_match, \
    ref_labels_list_remove_match, \
    ref_labels_score_list_remove_match, data_list = remove_previous_matched_labels(data_list,
                                                                    [predicted_labels_list,
                                                                     predicted_labels_scores_list,
                                                                     ref_labels_list,
                                                                     ref_labels_score_list,
                                                                     ],
                                                                        topk=topk,
                                                                        one_iter=one_iter
                                                                    )
    if train_eval == "train":
        json.dump(data_list, open(save_data_file, "w", encoding='utf-8'))
    else:
        pass
    precision_recall = get_threshold(predicted_labels_list_remove_match, predicted_labels_scores_list_remove_match, ref_labels_list_remove_match,
                                     ref_labels_score_list_remove_match, model_name, topk, what_best=what_best)
    thresholds, precision, recall, fscore_remove_nan, best_threshold = precision_recall
    if what_best == "f1":
        ix = argmax(fscore_remove_nan)
    elif what_best == "recall":
        ix = argmax(recall)
    elif what_best == "precision":
        ix = argmax(precision)
    else:
        raise ValueError
    logging.info("validation best precision {}, recall {}, f1 {}".format(precision[ix], recall[ix], fscore_remove_nan[ix]))
    print("validation best precision {}, recall {}, f1 {}".format(precision[ix], recall[ix], fscore_remove_nan[ix]))
    return best_threshold

def get_annotation_prediction_reference(data_file, model_name, what_best):
    data_list = json.load(open(data_file, "r", encoding='utf-8'))
    predicted_labels_list, \
    predicted_labels_scores_list, \
    ref_labels_list, \
    ref_labels_score_list = get_prediction_references(data_list, "annotation", model_name=model_name, what_best=what_best)
    return predicted_labels_list, predicted_labels_scores_list, ref_labels_list, ref_labels_score_list

def remove_previous_matched_labels(data_list, predicted_ref_details, one_iter, train_eval="train", topk=50):
    predicted_labels_list, \
    predicted_labels_scores_list, \
    ref_labels_list, \
    ref_labels_score_list = predicted_ref_details
    predicted_labels_list_remove_match = copy.deepcopy(predicted_labels_list)
    predicted_labels_scores_list_remove_match = copy.deepcopy(predicted_labels_scores_list)
    ref_labels_list_remove_match = copy.deepcopy(ref_labels_list)
    ref_labels_score_list_remove_match = copy.deepcopy(ref_labels_score_list)
    for i in range(len(data_list)):
        previous_matched_labels = []
        if "matched_labels" not in data_list[i]:
            assert one_iter == 1
        else:
            for one_iter_labels in data_list[i]["matched_labels"][0:one_iter-1]:
                previous_matched_labels += one_iter_labels
        # previous_matched_labels = data_list[i]["previous_matched_labels_topk"]
        for one_matched_word in previous_matched_labels:
            # pop out the matched labels from the prediction and reference
            index1 = predicted_labels_list_remove_match[i].index(one_matched_word)
            predicted_labels_list_remove_match[i].pop(index1)
            predicted_labels_scores_list_remove_match[i].pop(index1)
            ref_labels_score_list_remove_match[i].pop(index1)
            if one_matched_word in ref_labels_list_remove_match[i]:
                index2 = ref_labels_list_remove_match[i].index(one_matched_word)
                ref_labels_list_remove_match[i].pop(index2)
    predicted_labels_list_remove_match_2 = []
    predicted_labels_scores_list_remove_match_2 = []
    ref_labels_list_remove_match_2 = []
    ref_labels_score_list_remove_match_2 = []
    for i in range(len(data_list)):
        if len(ref_labels_list_remove_match[i]) > 0:
            predicted_labels_list_remove_match_2.append(copy.deepcopy(predicted_labels_list_remove_match[i]))
            predicted_labels_scores_list_remove_match_2.append(
                copy.deepcopy(predicted_labels_scores_list_remove_match[i]))
            ref_labels_list_remove_match_2.append(
                copy.deepcopy(ref_labels_list_remove_match[i]))
            ref_labels_score_list_remove_match_2.append(
                copy.deepcopy(ref_labels_score_list_remove_match[i]))

        # for j in range(len(predicted_labels_list_remove_match)):
        #     matched_labels = [predicted_labels_list_remove_match[j][kk] for kk in range(len(predicted_labels_list_remove_match[j])) if predicted_labels_scores_list_remove_match[j][kk] >= threshold]
        #     #matched_labels = predicted_labels_list_remove_match[j][0:topk]
        #     if "matched_labels" not in data_list[j]:
        #         data_list[j]["matched_labels"] = []
        #     if len(data_list[j]["matched_labels"]) < one_iter:
        #         # for iter 0, 0-1 < 0
        #         # for iter 1, 1-1 < 1
        #         # if re-run iter 1, 2 - 1 < 1 false
        #         data_list[j]["matched_labels"].append(matched_labels)
        #     else:
        #         data_list[j]["matched_labels"][one_iter-1] = matched_labels
    return predicted_labels_list_remove_match_2, \
           predicted_labels_scores_list_remove_match_2, \
           ref_labels_list_remove_match_2, \
           ref_labels_score_list_remove_match_2, \
           data_list

def remove_previous_matched_labels_old(data_list, predicted_ref_details, one_iter, train_eval="train", topk=50):
    predicted_labels_list, \
    predicted_labels_scores_list, \
    ref_labels_list, \
    ref_labels_score_list = predicted_ref_details
    predicted_labels_list_remove_match = copy.deepcopy(predicted_labels_list)
    predicted_labels_scores_list_remove_match = copy.deepcopy(predicted_labels_scores_list)
    ref_labels_list_remove_match = copy.deepcopy(ref_labels_list)
    ref_labels_score_list_remove_match = copy.deepcopy(ref_labels_score_list)
    for i in range(len(data_list)):
        previous_matched_labels = []
        if "matched_labels_topk" not in data_list[i] and "matched_labels_steps" not in data_list[i]:
            assert one_iter == 0
        elif "matched_labels_topk" in data_list[i]:
            for one_iter_labels in data_list[i]["matched_labels_topk"][0:one_iter]:
                previous_matched_labels += one_iter_labels
        elif "matched_labels_steps" in data_list[i]:
            for one_iter_labels in data_list[i]["matched_labels_steps"][0:one_iter]:
                previous_matched_labels += one_iter_labels
        else:
            raise ValueError
        #previous_matched_labels = data_list[i]["previous_matched_labels_topk"]
        for one_matched_word in previous_matched_labels:
            # pop out the matched labels from the prediction and reference
            index1 = predicted_labels_list_remove_match[i].index(one_matched_word)
            predicted_labels_list_remove_match[i].pop(index1)
            predicted_labels_scores_list_remove_match[i].pop(index1)
            ref_labels_score_list_remove_match[i].pop(index1)
            index2 = ref_labels_list_remove_match[i].index(one_matched_word)
            ref_labels_list_remove_match[i].pop(index2)
    predicted_labels_list_remove_match_2 = []
    predicted_labels_scores_list_remove_match_2 = []
    ref_labels_list_remove_match_2 = []
    ref_labels_score_list_remove_match_2 = []
    for i in range(len(data_list)):
        if len(ref_labels_list_remove_match[i]) > 0:
            predicted_labels_list_remove_match_2.append(copy.deepcopy(predicted_labels_list_remove_match[i]))
            predicted_labels_scores_list_remove_match_2.append(copy.deepcopy(predicted_labels_scores_list_remove_match[i]))
            ref_labels_list_remove_match_2.append(
                copy.deepcopy(ref_labels_list_remove_match[i]))
            ref_labels_score_list_remove_match_2.append(
                copy.deepcopy(ref_labels_score_list_remove_match[i]))
    if train_eval == "train":
        for j in range(len(predicted_labels_list_remove_match)):
            matched_labels = list(set(predicted_labels_list_remove_match[j][0:topk]) & set(ref_labels_list_remove_match[j]))
            if len(data_list[j]["matched_labels_topk"]) - 1 < one_iter:
                # for iter 0, 0-1 < 0
                # for iter 1, 1-1 < 1
                # if re-run iter 1, 2 - 1 < 1 false
                data_list[j]["matched_labels_topk"].append(matched_labels)
            else:
                data_list[j]["matched_labels_topk"][one_iter] = matched_labels
    return predicted_labels_list_remove_match_2, \
           predicted_labels_scores_list_remove_match_2, \
           ref_labels_list_remove_match_2, \
           ref_labels_score_list_remove_match_2, \
           data_list

def compute_all_score_average():
    train_content = json.load(open(join("data", "top1000_from_validation_average_all", "top1000_from_validation_train_t1.json"), "r", encoding='utf-8'))
    match_score_all = []
    valid_score_all = []
    rf_t_idf_score_all = []
    for i in range(len(train_content)):
        for one_word in train_content[i]["final_score_dict_sort"]:
            match_score_all.append(train_content[i]["final_score_dict_sort"][one_word][6])
            valid_score_all.append(train_content[i]["final_score_dict_sort"][one_word][3])
            rf_t_idf_score_all.append(train_content[i]["final_score_dict_sort"][one_word][4])
    print("train match average is : {}".format(statistics.mean(match_score_all)))
    print("train match median is : {}".format(statistics.median(match_score_all)))
    print("\n")
    print("train valid average is : {}".format(statistics.mean(valid_score_all)))
    print("train valid median is : {}".format(statistics.median(valid_score_all)))
    print("\n")
    print("train rf-t-idf average is : {}".format(statistics.mean(rf_t_idf_score_all)))
    print("train rf-t-idf median is : {}".format(statistics.median(rf_t_idf_score_all)))
    print("\n")
    test_content = json.load(
        open(join("data", "top1000_from_validation_average_all", "top1000_from_validation_test_t1.json"), "r",
             encoding='utf-8'))
    match_score_all = []
    valid_score_all = []
    rf_t_idf_score_all = []
    for i in range(len(test_content)):
        for one_word in test_content[i]["final_score_dict_sort"]:
            match_score_all.append(test_content[i]["final_score_dict_sort"][one_word][6])
            valid_score_all.append(train_content[i]["final_score_dict_sort"][one_word][3])
            rf_t_idf_score_all.append(train_content[i]["final_score_dict_sort"][one_word][4])
    print("train match average is : {}".format(statistics.mean(match_score_all)))
    print("train match median is : {}".format(statistics.median(match_score_all)))
    print("\n")
    print("train valid average is : {}".format(statistics.mean(valid_score_all)))
    print("train valid median is : {}".format(statistics.median(valid_score_all)))
    print("\n")
    print("train rf-t-idf average is : {}".format(statistics.mean(rf_t_idf_score_all)))
    print("train rf-t-idf median is : {}".format(statistics.median(rf_t_idf_score_all)))
    print("\n")

def read_file(file_path):
    file_list = json.load(open(file_path, "r", encoding='utf-8'))
    return file_list







