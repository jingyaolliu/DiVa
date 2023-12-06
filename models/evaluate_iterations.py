import json
from evaluation_util import *
from utils import ROOT_PATH
from os.path import join

def get_threshold(topk, data_prefix, data_folder, one_iter, what_best):
    data_list = json.load(open("data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(one_iter + 1)), "r", encoding='utf-8'))
    best_threshold_from_validation_set = get_threshold_from_validation_set(data_list=data_list,
                save_data_file="data/{}/{}_test_t{}.json".format(data_folder, data_prefix, str(one_iter + 1)),
                                              topk=topk,
                                              model_name="DIVA-Large", what_best=what_best, one_iter=one_iter, train_eval="eval")
    return best_threshold_from_validation_set

def evaluate_annotation_performance_top_k(topk, data_list, one_iter, model_name, best_threshold, previous_matched_labels, matched_labels_steps, what_best):
    # data_file = "data/{}/annotation_t{}.json".format(data_folder, str(one_iter + 1))
    # predicted_labels_list, \
    # predicted_labels_scores_list, \
    # ref_labels_list, \
    # ref_labels_score_list = get_annotation_prediction_reference(data_file, model_name, what_best=what_best)
    predicted_labels_list, \
    predicted_labels_scores_list, \
    ref_labels_list, \
    ref_labels_score_list = get_prediction_references(data_list, "annotation", model_name=model_name,
                                                      what_best=what_best)
    for i in range(len(previous_matched_labels)):
        for one_matched_word in previous_matched_labels[i]:
            # pop out the matched labels from the prediction and reference
            index1 = predicted_labels_list[i].index(one_matched_word)
            predicted_labels_list[i].pop(index1)
            predicted_labels_scores_list[i].pop(index1)
            ref_labels_score_list[i].pop(index1)
            index2 = ref_labels_list[i].index(one_matched_word)
            ref_labels_list[i].pop(index2)
    if one_iter == 0:
        matched_labels_steps[one_iter] = []
    for i in range(len(predicted_labels_list)):
        matched_labels = list(set(predicted_labels_list[i][0:topk]) & set(ref_labels_list[i]))
        if one_iter not in matched_labels_steps:
            matched_labels_steps[one_iter] = []
        if one_iter == 0:
            previous_matched_labels.append(matched_labels)
        else:
            previous_matched_labels[i] += matched_labels
        matched_labels_steps[one_iter].append(matched_labels)
    predicted_labels_scores_list_topk = []
    ref_labels_score_list_topk = []
    ref_labels_score_list_all = []
    for i in range(len(predicted_labels_scores_list)):
        predicted_labels_scores_list_topk += predicted_labels_scores_list[i][0:topk]
        ref_labels_score_list_topk += ref_labels_score_list[i][0:topk]
        ref_labels_score_list_all += ref_labels_score_list[i]
    predicted_labels_scores_list_topk_binary = [1 if one_score >= best_threshold else 0 for one_score in
                                                predicted_labels_scores_list_topk]
    precision_score = metrics.precision_score(ref_labels_score_list_topk, predicted_labels_scores_list_topk_binary)
    true_positive = [1 if ref_labels_score_list_topk[i] == 1 and predicted_labels_scores_list_topk_binary[i] == 1 else 0
                     for i in range(len(ref_labels_score_list_topk))]
    recall_score = float(sum(true_positive)) / sum(ref_labels_score_list_all)
    f1_score = ((2 * precision_score * recall_score) / (precision_score + recall_score)) if (precision_score + recall_score) != 0 else 0

    return previous_matched_labels, matched_labels_steps, [precision_score, recall_score, f1_score]

def main():
    all_best_threshold_list = []
    what_best = "recall"
    print(what_best)
    previous_matched_labels = []
    matched_labels_steps = {}
    for one_iter in range(0, 6):
        threshold = "from_validation"
        update_method = "average"
        sample_method = "all"
        data_prefix = "top{}_{}".format(str("1000"), str(threshold))
        data_folder = "{}_{}_{}".format(data_prefix, update_method, sample_method)

        best_threshold_from_validation_set = get_threshold(topk=10,
                                                           data_prefix=data_prefix,
                                                           data_folder=data_folder,
                                                           one_iter=one_iter,
                                                           what_best=what_best)
        topk = 50
        model_name = "DIVA-Large"
        data_file = "data/{}/annotation_t{}.json".format(data_folder, str(one_iter + 1))
        data_list = json.load(open(data_file, "r", encoding='utf-8'))
        previous_matched_labels, matched_labels_steps, results = evaluate_annotation_performance_top_k(topk,
                                              data_list,
                                              one_iter,
                                              model_name,
                                              best_threshold=best_threshold_from_validation_set,
                                              previous_matched_labels=previous_matched_labels,
                                              matched_labels_steps=matched_labels_steps, what_best=what_best)
        precision_score, recall_score, f1_score = results
        all_best_threshold_list.append(best_threshold_from_validation_set)
        print(one_iter)
        print(best_threshold_from_validation_set)
        print(",".join(previous_matched_labels[0]))
        print(",".join(matched_labels_steps[one_iter][0]))
        print("precision: {}".format(precision_score))
        print("recall: {}".format(recall_score))
        print("f1 score : {}".format(f1_score))
        print("program end")

#main
if __name__ == '__main__':
    main()



