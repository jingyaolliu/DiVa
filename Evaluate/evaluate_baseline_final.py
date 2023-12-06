from __future__ import annotations
import argparse
import json
from re import S
import re
from sre_constants import MAX_UNTIL
# from sys import last_traceback 
import time
# from regex import L
import math
# from regex import R

from yaml import load
from numpy import negative
import pandas as pd
# import parser
from soft_match_metric import lexical_soft_match
from soft_match_metrics_bertscore import get_bert_score
# from sklearn.datasets import load_boston

# from evaluation_util import *
# from utils import ROOT_PATH
from os.path import join
import numpy as np
import torch
from sklearn.metrics import precision_score,recall_score,f1_score,ndcg_score,dcg_score
import pandas as pd
import copy
from napkinxc.measures import Jain_et_al_inverse_propensity,psprecision_at_k,psndcg_at_k,psdcg_at_k,count_labels
import openpyxl
import matplotlib.pyplot as plt
import argparse
from yao_eval_utils import *


from get_trained_labels import get_all_trained_labels
parser = argparse.ArgumentParser()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# golden expert annotated+golden annotated+expert
parser.add_argument('--label_type', type=str, required=False, default='golden')
parser.add_argument('--methods', type=str, required=False, default='all_positive')
parser.add_argument('--eval_metrics', type=str, required=False, default='topk',help='topk,threshold')


args = parser.parse_args()

args.eval_metrics='threshold'

args.label_type = 'golden'
# args.label_type = 'annotated+golden'
# args.label_type = 'annotated+expert'
# args.label_type = 'expert'

# args.methods='tf-idf'
# args.methods='rf-idf'
# args.methods='tf-trunc_idf'
# args.methods='rf-trunc_idf'
# args.methods='embeddingrank'
# args.methods='text-rank-remove-stopwords-first'
# args.methods='text-rank-remove-stopwords-later'
# args.methods='nnpu'
# args.methods='mlc'



#! ???????????s

topk_workbook_name='/baseline-{}({}).xlsx'.format(args.methods,args.label_type)
threshold_workbookname='/iter1ebc_threshold_result_golden_0320.xlsx'

#todo---------------------??????--------------------------
#todo ?????????????test_list?? final_score_dict?е????
#todo ???????????????
#todo eval_an=based_(macro){precision:,recall:,f1:,} PS_based_{psp:,psndcg} NORMPS_based{NormPSP:,NormPSndcg}
#todo ????soft-match ???? 2022-11-1

#?-------------------????-------------------------------
#? label????=[golden_only,golden+annotation]
#?????????????λ??(_??score_) rf = ??????????? tf
#? _3_ : rf-idf //// _4_ : rf-trunc_idf //// _7_ : embeddingrank //// _8_ : text-rank(???????????) //// _9_ : text_rank(????????????)
#? _10_ : tf-idf //// _11_ : tf-trunc_idf

#*---------------------????--------------------------------
#* ????1: topk:10
#* ????2??best_metric_threshold

#? EVALUATION ?????
ROOT_PATH='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate'
ROOT_PATH_preprocess='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data'
# ?????????? 
ScoreColName='final_score_dict_sort'
GoldenColName='song_pseudo_golden_labels'
# ?????????
LabelColName='song_annotated_labels'

# train label embedding ???λ??
label_embd_path0_train='/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_id_candidate_words_shuffle.npy'
label_embd_path1_train='/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_candidate_words_embedding_shuffle.npy'
# test label embeding ???λ??
label_embd_path0_test="/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_id_candidate_words.txt"
label_embd_path1_test="/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_candidate_words_embedding.npy"


def load_data(test_file_name):
    if 'home' in test_file_name:
        test_lst=json.load(open(test_file_name))
    else:
        test_lst = json.load( open('/'.join([ROOT_PATH_preprocess, test_file_name])))
    return test_lst

#f_path1 ??????е?candidate_w(txt??npy) f_path2??????е?candidate_w_vec(npy)
# f_path1,2 ??train f_path3,4??test
def load_embeddings(f_path0,f_path1):
    if '.npy' in f_path0:
        candidates=np.load(f_path0)
    else:
        # for txt file
        f = open(f_path0,encoding = "utf-8")
        candidates=f.read()
        candidates=candidates.split('\n')
    l2index=get_word2index(candidates)
    candidates_embedding=np.load(f_path1)
    return l2index,candidates_embedding

#todo ???label_embeddings
def embeddings_merge(l2index_train,l2index_test,candidates_embedding_train,candidates_embedding_test):
    l2index={}
    candidates_embedding=[]
    # ?????train
    index=0
    for l_info in l2index_train:
        if l_info not in l2index:
            l2index[l_info]=len(l2index)
            candidates_embedding.append(candidates_embedding_train[index])
            index+=1
    # ?????test
    index=0
    for l_info in l2index_test:
        if l_info not in l2index:
            l2index[l_info]=len(l2index)
            candidates_embedding.append(candidates_embedding_test[index])
            index+=1
    return l2index,candidates_embedding
    
# ????word2index ???????candidatew ??????????
def get_word2index(candidate_w_array):
    # l2index{label_info(song_id,label):vec}
    l2index={}
    index=0
    for l_info in candidate_w_array:
        l_info=re.sub('\n','',l_info)
        l2index[l_info]=index
        index+=1
    return l2index

###################################################################################################
#! SOFT MATCH(SEMATIC)????
###################################################################################################
def greedy_cos_idf_labels_term(
    ref_embedding,
    hyp_embedding,
    threshold
):
    """
        Compute greedy matching based on cosine similarity.

        Args:
            - :param: `ref_embedding` (torch.Tensor):
                    embeddings of reference sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
                    labels_embeding:[1,labels_num,768*2]
            - :param: `ref_lens` (list of int): list of reference sentence length. labels_num
            - :param: `hyp_embedding` (torch.Tensor):
                    embeddings of candidate sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
                    candidate_w embedding [1,1,768*2]
            - :param: `hyp_lens` (list of int): list of candidate sentence length.
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    word_recall = sim.max(dim=1)[0][0]
    max_sim=torch.max(word_recall)
    # ????
    if max_sim>threshold:
        return 1
    # ??????
    else:
        return 0

def semantic_soft_match(song_id,pred_ls,golden_ls,l2index,label_embedings,threshold,device):
    """
        Compute y_pred based on semantic soft matching.

        Args:
            - :param: `pred_l` (list):
            - :param: `golden_l` (list of int): list of reference sentence length. labels_num
            - :param: `w2embeding` (dict:{w:w_vec}): w_vec:[768*2]tensor
            - :param: `threshold` (float).
            - :param: `device` (torch.device): cuda // cpu.
    """
  
    y_pred=[]
    golden_embds=[]
    for golden_l in golden_ls:
        key_tmp=song_id+'|'+golden_l
        try:
            label_index=l2index[key_tmp]
        except KeyError as Exception:
            print('no embeddings:{}'.format(key_tmp))
            continue
        label_vec=label_embedings[label_index]
        golden_embds.append(torch.tensor(label_vec).to(device))
    
    if len(golden_embds)==0:
        return [0]*len(pred_ls)
    ref_embds=torch.stack(golden_embds)
    ref_embds=torch.stack([ref_embds])

    # ????pred_embedding [1*1*(768*2)]
    for pred_l in pred_ls:
        try:
            pred_l_index=l2index[song_id+'|'+pred_l]
        except KeyError as Exception:
            print('no embeddings:{}'.format(song_id+'|'+pred_l))
            continue
        pred_vec=label_embedings[pred_l_index]
        # input pre
        hyp_embds=torch.stack([torch.tensor(pred_vec)])
        hyp_embds=torch.stack([hyp_embds]).to(device)
        y_pred.append(greedy_cos_idf_labels_term(ref_embds,hyp_embds,threshold))
    
    return y_pred
###################################################################################################
#! SOFT MATCH(SEMATIC)????
###################################################################################################


# ?????????
def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

#todo propensity score calculation by jain 
def cal_propensity_score_jain(label_class,data_lst,LabelColName=LabelColName,GoldenColName=GoldenColName,A=0.55,B=1.5):
    train_data=[]
    # label2index={w:index}
    label2index={}
    for song_comment in data_lst:
        golden_lst=song_comment[GoldenColName]
        if label_class=='golden with annotation':
            annotations_label_lst=song_comment['song_annotated_labels']
            song_labels=list(set(golden_lst+annotations_label_lst))
        else:
            # label_class= golen only
            song_labels=list(set(golden_lst))
        # ??song label?????????????
        song_lables_index=[]
        for song_label in song_labels:
            if song_label not in label2index:
                length_of_labels=len(label2index)
                label2index[song_label]=length_of_labels
                song_lables_index.append(length_of_labels)
            else:
                song_lables_index.append(label2index[song_label])
        train_data.append(song_lables_index)
    reverse_p=Jain_et_al_inverse_propensity(train_data)
    return label2index,reverse_p


#todo ???propensity_score
#* ???????-golden_label, ???label
# label class??? golden only/ golden with annotation
def cal_propensity_score(label_class,data_lst,LabelColName=LabelColName,GoldenColName=GoldenColName,A=0.55,B=1.5):
    # label_dict={w:w_count}
    label_dict={}
    for song_comment in data_lst:
        golden_lst=song_comment[GoldenColName]
        if label_class=='golden with annotation':
            annotations_label_lst=song_comment['song_annotated_labels']
            song_labels=list(set(golden_lst+annotations_label_lst))
        else:
            # label_class= golen only
            song_labels=list(set(golden_lst))
        for song_l in song_labels:
            if song_l not in label_dict:
                label_dict[song_l]=1
            else:
                label_dict[song_l]+=1
    # ????label??????????
    N=len(data_lst)
    C=(math.log(N)-1)*math.pow((1+B),A)
    # propensity_score?????? inverse_p_dict={w: pl}
    inverse_p_dict={}
    for l in label_dict:
        Nl=label_dict[l]
        inverse_p_dict[l]=1+C*math.pow((Nl+B),-A)
    # count=0????????invers_p
    inverse_p_dict['0']=1+C*math.pow(1.5,-0.5)
    return inverse_p_dict
    

#todo ????jain?????????y_true,y_pred
# sorted_list=[candiate_w,score_list]   label_lst?м??true???sorted_list????????
def get_eval_input_jain(label_lst,sorted_list,label2index):
    y_true=[]
    y_pred=[]
    # ????y_true
    for label in label_lst:
        y_true.append(label2index[label])
    # ????y_pred
    for candiadate_l in sorted_list:
        pred_l=candiadate_l[0]
        if pred_l in label2index:
            y_pred.append(label2index[pred_l])
        else:
            y_pred.append(-1)
    return y_true,y_pred

#todo ???song_id 
#? song_id λ??dict[song_name] song_id??\d+ ------> song_id_\d+


#todo ????softmax??????????????
# l2index??????label?????ebedding???? 
# default label_class=golden only
def soft_match_cal(label_class,data_lst,site,inverse_p_dict,l2index,label_embeddings,threshold,device,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName):
    p_lst=[]
    f1_lst=[]
    recall_lst=[]

    psp_lst=[]
    ideal_psp_lst=[]

    psndcg_lst=[]
    ideal_psndcg_lst=[]

    for song_com in data_lst:
        # ???song_id
        song_name=song_com['song_name']
        song_id=norm_song_id(song_name)
        golden_lst=song_com[GoldenColName]
        if label_class=='golden with annotation':
            annotation_lst=song_com['song_annotated_labels']
            label_lst=list(set(golden_lst+annotation_lst))
        else:
            label_lst=list(set(golden_lst))
        # ???predict data
        score_dict=song_com[ScoreColName]
        # ????y_pred y_pred:len candidates.length y_pred:seq candidates.sequence
        # ????propensity_score
        y_true=[]
        p_inverse=[]
        y_score=[]
        for candidate_w in score_dict:
            y_score.append(score_dict[candidate_w][site])
            if candidate_w in label_lst:
                y_true.append(1)
                p_inverse.append(inverse_p_dict[candidate_w])
            else:
                y_true.append(0)
                p_inverse.append(inverse_p_dict["0"])
        y_true=np.array(y_true)
        p_inverse=np.array(p_inverse)
        y_score=np.array(y_score)
        #todo soft match ????y_pred
        candidates_lst=list(score_dict.keys())
        y_pred=semantic_soft_match(song_id,candidates_lst,label_lst,l2index,label_embeddings,threshold,device)
        # ????tp+fp
        positive_num=y_pred.count(1)
        y_pred=np.array(y_pred)

        # ????????????
         # ???????
        p_tmp=precision_score(y_true,y_pred,average='binary')
        # 0- 0positive 1- 1positive
        p_lst.append(p_tmp)
        # ????recall
        recall_tmp=recall_score(y_true,y_pred,average='binary')
        recall_lst.append(recall_tmp)
        # ????f1
        f1_tmp=f1_score(y_true,y_pred,average='binary')        
        f1_lst.append(f1_tmp)
        
        # ????propensity?????
        # ????psp
        # ???none???average??????????????[[],[]]??????????
        psp_tmp=precision_score([y_true],[y_pred],average=None,zero_division=0)
        try:
            psp_tmp*=p_inverse
        except ValueError:
            print(psp_tmp.shape,np.array(p_inverse).shape)

        psp_lst.append(psp_tmp.sum()/positive_num)
            # print(psp_tmp.sum()/topk)

        #????normalize propensity score?????????
        # norm-psp
        # dcg_score(y_true=[[],[]],y_pred) y_true---to be rank
        # ideal_psp_tmp=precision_score([observed_lst],[ideal_predict_lst_topk],average=None,zero_division=0)
        # ideal_psp_tmp*=weight_metric
        ideal_psp_none=p_inverse * y_true 
        ideal_psp_none=abs(np.sort(-ideal_psp_none))
        ideal_psp_none=ideal_psp_none[0:positive_num]
        ideal_psp_none=np.array(ideal_psp_none)
        ideal_psp=ideal_psp_none.sum()/positive_num
        ideal_psp_lst.append(ideal_psp)

        # ????psndcg
        # ndcg(true_relevance----???????,pred_score---????????)
        ps_true_relevance=[p_inverse*y_true]
        true_relevance=[y_true]
        pred_score=[y_score * y_pred]
        ps_dcg_tmp=dcg_score(ps_true_relevance,pred_score,k=positive_num)
        ideal_dcg_tmp=dcg_score(true_relevance,ps_true_relevance,k=positive_num)
        psndcg_tmp=ps_dcg_tmp/ideal_dcg_tmp
        psndcg_lst.append(psndcg_tmp)

        #norm-psndcg
        ideal_psdcg_tmp=dcg_score(ps_true_relevance,ps_true_relevance,k=positive_num)
        ideal_psndcg_tmp=ideal_psdcg_tmp/ideal_dcg_tmp
        ideal_psndcg_lst.append(ideal_psndcg_tmp)
    
    
    p=np.array(p_lst).mean()
    r=np.array(recall_lst).mean()
    f1=np.array(f1_lst).mean()
    psp=np.array(psp_lst).mean()
    norm_psp=np.array(psp_lst).sum()/np.array(ideal_psp_lst).sum()
    psndcg=np.array(psndcg_lst).mean()
    norm_psndcg=np.array(psndcg_lst).sum()/np.array(ideal_psndcg_lst).sum()
    return p,r,f1,psp,norm_psp,psndcg,norm_psndcg

# soft match p,f1,recall
# true_num?????????label????
def soft_match_precision_f1_recall(song_id,y_true,y_pred,true_num,l2index,label_embeddings,positive_num,threshold,candidate_w_lst,label_lst,device,weight=None):
    # precision????
    numerator=0
    # tp
    tp=0
    if weight==None:
        weight=np.ones(len(y_pred))
    soft_match_true=semantic_soft_match(song_id,candidate_w_lst[0:positive_num],label_lst,l2index,label_embeddings,threshold,device)
    for i in range(0,positive_num):
        if y_true[i]==y_pred[i]:
            numerator+=weight[i]
            tp+=1
        else:
            if soft_match_true[i]==1:
                numerator+=weight[i]
                tp+=1
    p=numerator/positive_num
    r=tp/true_num
    f1=2*tp/(positive_num+true_num)

    return p,r,f1

#todo soft match eval score
def soft_eval(final_score_dict,sort_candidate_lst,label_lst,topk,site,weight_metric=None):
    
    '''
    input:
        --final_score_dict:<dict> {candidate_label:[score]*15} 
            bleu score is in site -2
        --score_candidate_lst:<list> list of sorted candidate labels
            useful when ebc and baseling is running
        --label_lst: <list>
            mostly goldens
        --topk :<int>
            topk candidate labels are of use
        --weight_metric:<list[float]> list of inverse propensity score
            default is None -> normal p else is psp 
            length=topk
    output:
        --bleu metric [precision, recall, f1]   
    '''
    if weight_metric==None:
        weight_metric=np.ones(topk)
    else:
        weight_metric=np.array(weight_metric[:topk])
    tp=0
    fp=0
    true_num=len(label_lst)
    positive_num=topk
    if site==-2:
        threshold=threshold_for_bleu
    else:
        threshold=threshold_for_semantic
    for i in range(0,topk):
        if sort_candidate_lst[i] in label_lst:
            tp+=weight_metric[i]
        else:
            if soft_by_threshold:
                if final_score_dict[sort_candidate_lst[i]][site]>threshold:
                    tp+=weight_metric[i]
            else:
                tp+=final_score_dict[sort_candidate_lst[i]][site]*weight_metric[i]
    if positive_num!=0:
        p=tp/positive_num
    else:
        p=0
    r=tp/true_num
    f1=2*tp/(positive_num+true_num)
    return p,r,f1

#todo soft match by bleu site=-2
def soft_bleu_eval(final_score_dict,sort_candidate_lst,label_lst,topk,weight_metric=None):
    site=-2
    return soft_eval(final_score_dict,sort_candidate_lst,label_lst,topk,site,weight_metric)
#todo soft match by bert score site=-3
def soft_bertscore_eval(final_score_dict,sort_candidate_lst,label_lst,topk,weight_metric=None):
    site=-3
    return soft_eval(final_score_dict,sort_candidate_lst,label_lst,topk,site,weight_metric)

#todo topk


#获取训练时的所有标??(注意此处该如何选择)
def get_trained_labels(label_class=None,golden_with_silver=False,iter=0):
    train_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter{}_5.json'.format(iter)
    data_lst=json.load(open(train_path,'r'))
    train_labels_golden=[]
    train_labels_expert=[]
    for data in data_lst:
  
        tmp_labels_golden=data['song_pseudo_golden_labels']
        if golden_with_silver:
            silver_labels=data['song_all_silver_labels']
            tmp_labels_golden=list(set(tmp_labels_golden)|set(silver_labels))

        tmp_labels_expert=data['song_labels']
        if golden_with_silver:
            silver_labels=data['song_all_silver_labels']
            tmp_labels_expert=list(set(tmp_labels_expert)|set(silver_labels))
        # if 'golden' in label_class:
        #     tmp_labels=data['song_pseudo_golden_labels']
        #     if golden_with_silver:
        #         silver_labels=data['song_all_silver_labels']
        #         tmp_labels=list(set(tmp_labels)|set(silver_labels))
        # else:
        #     tmp_labels=data['song_labels']
        #     if golden_with_silver:
        #         silver_labels=data['song_all_silver_labels']
        #         tmp_labels=list(set(tmp_labels)|set(silver_labels))
        train_labels_golden+=tmp_labels_golden
        train_labels_expert+=tmp_labels_expert
    #save 
    train_labels_expert=list(set(train_labels_expert))
    train_labels_golden=list(set(train_labels_golden))
    save_folder='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/iter_labels/'
    file_name='all_trained_labels(golden-expert).json'
    train_data_dict={}
    train_data_dict['golden']=train_labels_golden
    train_data_dict['expert']=train_labels_expert
    json.dump(train_data_dict,open(save_folder+file_name,'w'))
    print('saved')
    
    return list(set(train_labels))
# get_trained_labels()
# print('done')

def soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,true_lenth,soft_type):
    tp=0
    if soft_type=='semantic':
        soft_func=get_bert_score
    else:
        soft_func=lexical_soft_match
    for label in pred_labels:
        if label in true_labels:
            tmp_soft_score=1
        else:
            if '|'.join([song_name,label]) in semantic_score_dict:
                tmp_soft_score=semantic_score_dict['|'.join([song_name,label])]
                tmp_soft_score=max(tmp_soft_score,0)
            else:
                tmp_soft_score=soft_func(label,true_labels)
                tmp_soft_score=max(tmp_soft_score,0)
        tp+=tmp_soft_score
    
    if len(pred_labels)==0:
        p=0
    else:
        p=tp/len(pred_labels)
    if true_lenth==0:
        r=0
    else:
        r=tp/true_lenth
    # print('-----soft------len(true_labels)',len(true_labels))
    if p+r==0:
        f1=0
    else:
        f1=2*p*r/(p+r)
    return p,r,f1


# topk new--0418
# def cal_score_topk(label2index,labels_set,song_info_dict,label_class,data_lst,topk,site,inverse_p_dict,semantic_score_dict,lexical_score_dict,trained_labels,all_positive=False,exact_match=False,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName):
#     print('topk calculation.....')
#     """
#     label_set : trained labels(seen labels)
#     """
#     jain_true_labels=[]
#     jain_pred_labels=[]
    
#     p_lst=[]
#     semantic_soft_p_lst=[]
#     bleu_soft_p_lst=[]
#     f1_lst=[]
#     semantic_soft_f1_lst=[]
#     bleu_soft_f1_lst=[]
#     recall_lst=[]
#     semantic_soft_recall_lst=[]
#     bleu_soft_recall_lst=[]

#     psp_lst=[]
#     semantic_soft_psp_lst=[]
#     bleu_soft_psp_lst=[]
#     ideal_psp_lst=[]

#     psndcg_lst=[]
#     ndcg_lst=[]
#     norm_psndcg_lst=[]
#     ideal_psndcg_lst=[]
#     pred_unseen_labels=[]
#     test_unseen_labels=[]
    
#     song_new_labels_len=[]
#     song_unseen_new_labels_len=[]
    
   
#     for song_com in data_lst:
#         # get song name
#         song_name=song_com['song_name']
#         song_id=norm_song_id(song_name)
#         if song_name not in song_info_dict:
#             continue
#         # get all related labels
#         golden_labels=song_info_dict[song_name]['golden_labels']
#         expert_labels=song_info_dict[song_name]['expert_labels']
#         annotated_golden_labels=song_info_dict[song_name]['annotated_golden_labels']
#         annotated_expert_labels=song_info_dict[song_name]['annotated_expert_labels']
#         score_dict=song_com[ScoreColName]
#         all_candidates=list(score_dict.keys())
#         # candidate_sort=sorted(score_dict.items(),key=lambda val:val[1][site],reverse=True)
        
#         #get truth
#         golden_lst=song_com[GoldenColName]
#         if label_class=='annotated+golden':
#             label_lst=annotated_golden_labels
#         elif label_class=='golden':
#             label_lst=golden_labels
#         elif label_class=='expert':
#             label_lst=expert_labels
#         elif label_class=='annotated+expert':
#             label_lst=annotated_expert_labels
#         true_labels=label_lst
#         # old labels -> 开始标??
#         if 'golden' in label_class:
#             old_labels=golden_labels
#         else:
#             old_labels=expert_labels
#         # 用true labels去扩充final_score_dict_sort
#         for true_l in true_labels:
#             if true_l not in score_dict:
#                 score_dict[true_l]=[0]*15
        
#         #sort again
#         candidate_sort=sorted(score_dict.items(),key=lambda val:val[1][site],reverse=True)
#         candidates_in_labels=list(set(labels_set)&set(all_candidates))
#         # list of tuple(label_index,label_score) | (int/str,float)
#         pred_labels_jain=[]
#         true_labels_jain=[]
#         if all_positive:
#             pred_labels=all_candidates
#         elif exact_match:
#             pred_labels=candidates_in_labels
#         else:
#             # get pred_labels
#             pred_labels=[]
#             for i in range(topk):
#                 pred_labels.append(candidate_sort[i][0])
                
#         # prepare true labels and pred labels
#         for label in true_labels:
#             true_labels_jain.append(label2index[label])
#         for label in pred_labels:
#             if all_positive or exact_match:
#                 tmp_tuple=(label2index[label],1)
#             else:
#                 tmp_tuple=(label2index[label],score_dict[label][site])
#         jain_true_labels.append(true_labels_jain)
#         jain_pred_labels.append(pred_labels_jain)
            
#         song_new_labels=list(set(pred_labels)-set(old_labels))
#         song_unseen_new_labels=list(set(song_new_labels)-set(trained_labels))
#         song_new_labels_len.append(len(song_new_labels))
#         song_new_labels_len.append(len(song_unseen_new_labels))
        
#         # if site==11:
#         test_unseen_labels+=true_labels
#         pred_unseen_labels+=pred_labels
        
#         # get predict_list
#         predict_lst=[]
#         for i in range(len(candidate_sort)):
#             if candidate_sort[i][0] in pred_labels:
#                 predict_lst.append(1)
#             else:
#                 predict_lst.append(0)
#         pred_length=predict_lst.count(1)
#         observed_lst=[]
#         score_lst=[]
#         for i in range(len(candidate_sort)):
#             if all_positive or exact_match:
#                 if candidate_sort[i][0] in pred_labels:
#                     score_lst.append(1)
#                 else:
#                     score_lst.append(0)
#             else:
#                 score_lst.append(candidate_sort[i][1][site])
#             if candidate_sort[i][0] in true_labels:
#                 observed_lst.append(1)
#             else:
#                 observed_lst.append(0)
#         true_lenth=observed_lst.count(1)
#         if true_lenth!=len(true_labels):
#             print(true_lenth,len(true_labels))
#         # pred_score
#         # for candidate_l in candidate_sort:
#         #     # print(candidate_l[0])
#         #     score_lst.append(candidate_l[1][site])
#         #     candidate_w_lst.append(candidate_l[0])
#         #     if candidate_l[0] in label_lst:
#         #         observed_lst.append(1)
#         #         truth_len+=1
#         #     else:
#         #         observed_lst.append(0)
 
#         weight_metric=[]
#         for candidate_l in candidate_sort:
#             if candidate_l[0] in inverse_p_dict:
#                 weight_metric.append(inverse_p_dict[candidate_l[0]])
#             else:
#                 weight_metric.append(inverse_p_dict['0'])
 
#         predict_lst=np.array(predict_lst)
#         observed_lst=np.array(observed_lst)
#         score_lst=np.array(score_lst)
#         ideal_predict_lst_topk=predict_lst & observed_lst

#         p_tmp=precision_score(observed_lst,predict_lst,average='binary')
#         p_lst.append(p_tmp)
#         recall_tmp=recall_score(observed_lst,predict_lst,average='binary')
#         recall_lst.append(recall_tmp)
#         f1_tmp=f1_score(observed_lst,predict_lst,average='binary')        
#         f1_lst.append(f1_tmp)
        
#         #todo get semantic soft p,r,f1
#         semantic_soft_p,semantic_soft_r,semantic_soft_f1=soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,true_lenth,soft_type='semantic')
#         semantic_soft_p_lst.append(semantic_soft_p)
#         semantic_soft_recall_lst.append(semantic_soft_r)
#         semantic_soft_f1_lst.append(semantic_soft_f1)
#         lexical_soft_p,lexical_soft_r,lexical_soft_f1=soft_metrics(song_name,true_labels,pred_labels,lexical_score_dict,true_lenth,soft_type='lexical')
#         bleu_soft_p_lst.append(lexical_soft_p)
#         bleu_soft_recall_lst.append(lexical_soft_r)
#         bleu_soft_f1_lst.append(lexical_soft_f1)
#         if recall_tmp>semantic_soft_r or recall_tmp>lexical_soft_r:
#             print('hhhhhhhhh',recall_tmp,semantic_soft_r,lexical_soft_r)
        
#         #psp
#         ps_predict_lst=predict_lst*weight_metric
#         psp_tmp=precision_score([observed_lst],[predict_lst],average=None,zero_division=0)
#         try:
#             psp_tmp*=weight_metric
#         except ValueError:
#             print(psp_tmp.shape,np.array(weight_metric).shape)

#         if pred_length>0:
#             psp_lst.append(psp_tmp.sum()/pred_length)
#         else:
#             psp_lst.append(0)
            
#         # ndcg psndcg
#         ps_true_relevance=[weight_metric*observed_lst]
#         true_relevance=[observed_lst]
#         pred_score=[score_lst]
#         ps_dcg_tmp=dcg_score(ps_true_relevance,pred_score,k=topk)
#         dcg_tmp=dcg_score(true_relevance,pred_score,k=topk)
#         ideal_dcg_tmp=dcg_score(true_relevance,ps_true_relevance,k=topk)
#         psndcg_tmp=ps_dcg_tmp/ideal_dcg_tmp
#         ndcg_tmp=dcg_tmp/ideal_dcg_tmp
#         psndcg_lst.append(psndcg_tmp)
#         ndcg_lst.append(ndcg_tmp)
        
#         # norm psp psndcg
#         ideal_psp_none=weight_metric * observed_lst 
#         ideal_psp_none=abs(np.sort(-ideal_psp_none))
#         ideal_psp_none=ideal_psp_none[0:pred_length]
#         ideal_psp_none=np.array(ideal_psp_none)
#         ideal_psp=ideal_psp_none.sum()/topk
#         ideal_psp_lst.append(ideal_psp)
#         # if ideal_psp_lst.sum()!=0:
#         #     norm_psp_lst.append(psp_tmp.sum()/ideal_psp_lst.sum())
#         # else:
#         #     norm_psp_lst.append(0)
#         #norm-psndcg
#         ideal_psdcg_tmp=dcg_score(ps_true_relevance,ps_true_relevance,k=topk)
#         ideal_psndcg_tmp=ideal_psdcg_tmp/ideal_dcg_tmp
#         ideal_psndcg_lst.append(ideal_psndcg_tmp)
#         norm_psndcg_lst.append(ps_dcg_tmp/ideal_psdcg_tmp)
        
#     # get nlsr
#     test_unseen_labels=list(set(test_unseen_labels)-set(trained_labels))
#     pred_unseen_labels=list(set(pred_unseen_labels)-set(trained_labels))
#     if len(test_unseen_labels)==0:
#         nlsr=len(pred_unseen_labels)
#     else:
#         nlsr=len(pred_unseen_labels)/len(test_unseen_labels)

#     p=np.array(p_lst).mean()
#     semantic_soft_p=np.array(semantic_soft_p_lst).mean()
#     lexical_soft_p=np.array(bleu_soft_p_lst).mean()
#     recall=np.array(recall_lst).mean()
#     semantic_soft_recall=np.array(semantic_soft_recall_lst).mean()
#     lexical_soft_recall=np.array(bleu_soft_recall_lst).mean()
#     f1=np.array(f1_lst).mean()
#     semantic_soft_f1=np.array(semantic_soft_f1_lst).mean()
#     lexical_soft_f1=np.array(bleu_soft_f1_lst).mean()
    
#     psp=np.array(psp_lst).mean()
#     ndcg=np.array(ndcg_lst).mean()
#     psndcg=np.array(psndcg_lst).mean()
    
#     norm_psp=np.array(psp_lst).sum()/np.array(ideal_psp_lst).sum()
#     norm_psndcg=np.array(psndcg_lst).sum()/np.array(ideal_psndcg_lst).sum()
    
#     # per_song_new_label_num=np.array(song_new_labels_len).mean()
#     # per_song_unseen_new_label_num=np.array(song_unseen_new_labels_len).mean()
    
#     return_info={}
#     return_info['p']=p
#     return_info['semantic_soft_p']=semantic_soft_p
#     return_info['lexical_soft_p']=lexical_soft_p
    
#     return_info['r']=recall
#     return_info['semantic_soft_r']=semantic_soft_recall
#     return_info['lexical_soft_r']=lexical_soft_recall
    
#     return_info['f1']=f1
#     return_info['semantic_soft_f1']=semantic_soft_f1
#     return_info['lexical_soft_f1']=lexical_soft_f1
    
#     return_info['psp']=psp
#     return_info['norm_psp']=norm_psp
#     return_info['ndcg']=ndcg
#     return_info['psndcg']=psndcg
#     return_info['norm_psndcg']=norm_psndcg
    
#     return_info['nlsr']=nlsr
    
#     return return_info


def cal_score_threshold_old(threshold,labels_set,song_info_dict,label_class,data_lst,site,inverse_p_dict,semantic_score_dict,lexical_score_dict,trained_labels,all_positive=False,exact_match=False,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName):
    print('threshold {} calculation.....'.format(threshold))
    """
    label_set : trained labels(seen labels)
    """
    
    p_lst=[]
    semantic_soft_p_lst=[]
    bleu_soft_p_lst=[]
    f1_lst=[]
    semantic_soft_f1_lst=[]
    bleu_soft_f1_lst=[]
    recall_lst=[]
    semantic_soft_recall_lst=[]
    bleu_soft_recall_lst=[]

    psp_lst=[]
    semantic_soft_psp_lst=[]
    bleu_soft_psp_lst=[]
    ideal_psp_lst=[]

    psndcg_lst=[]
    ndcg_lst=[]
    norm_psndcg_lst=[]
    ideal_psndcg_lst=[]
    pred_unseen_labels=[]
    test_unseen_labels=[]
    
    song_new_labels_len=[]
    song_unseen_new_labels_len=[]
    
   
    for song_com in data_lst:
        # get song name
        song_name=song_com['song_name']
        song_id=norm_song_id(song_name)
        if song_name not in song_info_dict:
            continue
        # get all related labels
        golden_labels=song_info_dict[song_name]['golden_labels']
        expert_labels=song_info_dict[song_name]['expert_labels']
        annotated_golden_labels=song_info_dict[song_name]['annotated_golden_labels']
        annotated_expert_labels=song_info_dict[song_name]['annotated_expert_labels']
        score_dict=song_com[ScoreColName]
        all_candidates=list(score_dict.keys())
        # candidate_sort=sorted(score_dict.items(),key=lambda val:val[1][site],reverse=True)
        
        #get truth
        golden_lst=song_com[GoldenColName]
        if label_class=='annotated+golden':
            label_lst=annotated_golden_labels
        elif label_class=='golden':
            label_lst=golden_labels
        elif label_class=='expert':
            label_lst=expert_labels
        elif label_class=='annotated+expert':
            label_lst=annotated_expert_labels
        true_labels=label_lst
        # old labels -> 开始标??
        if 'golden' in label_class:
            old_labels=golden_labels
        else:
            old_labels=expert_labels
        # 用true labels去扩充final_score_dict_sort
        for true_l in true_labels:
            if true_l not in score_dict:
                score_dict[true_l]=[0]*15
        
        #sort again
        candidate_sort=sorted(score_dict.items(),key=lambda val:val[1][site],reverse=True)
        candidates_in_labels=list(set(labels_set)&set(all_candidates))
        if all_positive:
            pred_labels=all_candidates
        elif exact_match:
            pred_labels=candidates_in_labels
        else:
            # get pred_labels
            pred_labels=[]
            for i in range(len(candidate_sort)):
                if candidate_sort[i][1][site]>=threshold:
                    pred_labels.append(candidate_sort[i][0])
                else:
                    break
                    
        song_new_labels=list(set(pred_labels)-set(old_labels))
        song_unseen_new_labels=list(set(song_new_labels)-set(trained_labels))
        song_new_labels_len.append(len(song_new_labels))
        song_new_labels_len.append(len(song_unseen_new_labels))
        
        # if site==11:
        test_unseen_labels+=true_labels
        pred_unseen_labels+=pred_labels
        
        # get predict_list
        predict_lst=[]
        for i in range(len(candidate_sort)):
            if candidate_sort[i][0] in pred_labels:
                predict_lst.append(1)
            else:
                predict_lst.append(0)
        pred_length=predict_lst.count(1)
        observed_lst=[]
        score_lst=[]
        for i in range(len(candidate_sort)):
            if all_positive or exact_match:
                if candidate_sort[i][0] in pred_labels:
                    score_lst.append(1)
                else:
                    score_lst.append(0)
            else:
                score_lst.append(candidate_sort[i][1][site])
            if candidate_sort[i][0] in true_labels:
                observed_lst.append(1)
            else:
                observed_lst.append(0)
        true_lenth=observed_lst.count(1)
        if true_lenth!=len(true_labels):
            print(true_lenth,len(true_labels))
        # pred_score
        # for candidate_l in candidate_sort:
        #     # print(candidate_l[0])
        #     score_lst.append(candidate_l[1][site])
        #     candidate_w_lst.append(candidate_l[0])
        #     if candidate_l[0] in label_lst:
        #         observed_lst.append(1)
        #         truth_len+=1
        #     else:
        #         observed_lst.append(0)
 
        weight_metric=[]
        for candidate_l in candidate_sort:
            if candidate_l[0] in inverse_p_dict:
                weight_metric.append(inverse_p_dict[candidate_l[0]])
            else:
                weight_metric.append(inverse_p_dict['0'])
 
        predict_lst=np.array(predict_lst)
        observed_lst=np.array(observed_lst)
        score_lst=np.array(score_lst)
        ideal_predict_lst_topk=predict_lst & observed_lst

        p_tmp=precision_score(observed_lst,predict_lst,average='binary',zero_division=0)
        p_lst.append(p_tmp)
        recall_tmp=recall_score(observed_lst,predict_lst,average='binary',zero_division=0)
        recall_lst.append(recall_tmp)
        f1_tmp=f1_score(observed_lst,predict_lst,average='binary',zero_division=0) 
        f1_lst.append(f1_tmp)
        
        # get semantic soft p,r,f1
        semantic_soft_p,semantic_soft_r,semantic_soft_f1=soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,true_lenth,soft_type='semantic')
        semantic_soft_p_lst.append(semantic_soft_p)
        semantic_soft_recall_lst.append(semantic_soft_r)
        semantic_soft_f1_lst.append(semantic_soft_f1)
        lexical_soft_p,lexical_soft_r,lexical_soft_f1=soft_metrics(song_name,true_labels,pred_labels,lexical_score_dict,true_lenth,soft_type='lexical')
        bleu_soft_p_lst.append(lexical_soft_p)
        bleu_soft_recall_lst.append(lexical_soft_r)
        bleu_soft_f1_lst.append(lexical_soft_f1)
        if recall_tmp>semantic_soft_r or recall_tmp>lexical_soft_r:
            print('hhhhhhhhh',recall_tmp,semantic_soft_r,lexical_soft_r)
        
        #psp
        ps_predict_lst=predict_lst*weight_metric
        psp_tmp=precision_score([observed_lst],[predict_lst],average=None,zero_division=0)
        try:
            psp_tmp*=weight_metric
        except ValueError:
            print(psp_tmp.shape,np.array(weight_metric).shape)
        
        if pred_length==0:
            psp_lst.append(0)
        else:
            psp_lst.append(psp_tmp.sum()/pred_length)
            
        # ndcg psndcg
        ps_true_relevance=[weight_metric*observed_lst]
        true_relevance=[observed_lst]
        pred_score=[score_lst]
        ps_dcg_tmp=dcg_score(ps_true_relevance,pred_score,k=pred_length)
        dcg_tmp=dcg_score(true_relevance,pred_score,k=pred_length)
        ps_ideal_dcg_tmp=dcg_score(true_relevance,ps_true_relevance,k=true_lenth)
        ideal_dcg_tmp=dcg_score(true_relevance,true_relevance,k=true_lenth)
        psndcg_tmp=ps_dcg_tmp/ps_ideal_dcg_tmp
        ndcg_tmp=dcg_tmp/ideal_dcg_tmp
        psndcg_lst.append(psndcg_tmp)
        ndcg_lst.append(ndcg_tmp)
        
        # norm psp psndcg
        ideal_psp_none=weight_metric * observed_lst 
        ideal_psp_none=abs(np.sort(-ideal_psp_none))
        ideal_psp_none=ideal_psp_none[0:true_lenth]
        ideal_psp_none=np.array(ideal_psp_none)
        if true_lenth>0:
            ideal_psp=ideal_psp_none.sum()/true_lenth
        else: 
            ideal_psp=0
        ideal_psp_lst.append(ideal_psp)
        # if ideal_psp_lst.sum()!=0:
        #     norm_psp_lst.append(psp_tmp.sum()/ideal_psp_lst.sum())
        # else:
        #     norm_psp_lst.append(0)
        #norm-psndcg
        ideal_psdcg_tmp=dcg_score(ps_true_relevance,ps_true_relevance,k=true_lenth)
        ideal_psndcg_tmp=ideal_psdcg_tmp/ideal_dcg_tmp
        ideal_psndcg_lst.append(ideal_psndcg_tmp)
        norm_psndcg_lst.append(ps_dcg_tmp/ideal_psdcg_tmp)
        
    # get nlsr
    test_unseen_labels=list(set(test_unseen_labels)-set(trained_labels))
    pred_unseen_labels=list(set(pred_unseen_labels)-set(trained_labels))
    if len(test_unseen_labels)==0:
        nlsr=len(pred_unseen_labels)
    else:
        nlsr=len(pred_unseen_labels)/len(test_unseen_labels)

    p=np.array(p_lst).mean()
    semantic_soft_p=np.array(semantic_soft_p_lst).mean()
    lexical_soft_p=np.array(bleu_soft_p_lst).mean()
    recall=np.array(recall_lst).mean()
    semantic_soft_recall=np.array(semantic_soft_recall_lst).mean()
    lexical_soft_recall=np.array(bleu_soft_recall_lst).mean()
    f1=np.array(f1_lst).mean()
    semantic_soft_f1=np.array(semantic_soft_f1_lst).mean()
    lexical_soft_f1=np.array(bleu_soft_f1_lst).mean()
    
    psp=np.array(psp_lst).mean()
    ndcg=np.array(ndcg_lst).mean()
    psndcg=np.array(psndcg_lst).mean()
    
    if np.array(ideal_psp_lst).sum()!=0:
        norm_psp=np.array(psp_lst).sum()/np.array(ideal_psp_lst).sum()
    else:
        norm_psp=0
    if np.array(ideal_psndcg_lst).sum()!=0:
        norm_psndcg=np.array(psndcg_lst).sum()/np.array(ideal_psndcg_lst).sum()
    else:
        norm_psndcg=0
    
    # per_song_new_label_num=np.array(song_new_labels_len).mean()
    # per_song_unseen_new_label_num=np.array(song_unseen_new_labels_len).mean()
    
    return_info={}
    return_info['p']=p
    return_info['semantic_soft_p']=semantic_soft_p
    return_info['lexical_soft_p']=lexical_soft_p
    
    return_info['r']=recall
    return_info['semantic_soft_r']=semantic_soft_recall
    return_info['lexical_soft_r']=lexical_soft_recall
    
    return_info['f1']=f1
    return_info['semantic_soft_f1']=semantic_soft_f1
    return_info['lexical_soft_f1']=lexical_soft_f1
    
    return_info['psp']=psp
    return_info['norm_psp']=norm_psp
    return_info['ndcg']=ndcg
    return_info['psndcg']=psndcg
    return_info['norm_psndcg']=norm_psndcg
    
    return_info['nlsr']=nlsr
    
    return return_info

#todo

# add additional infos to dict keys
def dict_keys_add_info(dict,info_str):
    new_dict={}
    for key in dict:
        new_dict[key+info_str]=dict[key]
    return new_dict
        

def method2site(method):
    site=0
    if method=='rf-idf':
        site=3
    elif method=='rf-trunc_idf':
        site=4
    elif method=='embeddingrank':
        #todo debug
        # continue
        site=7
    elif method=='text-rank-remove-stopwords-first':
        site=8
    elif method=='text-rank-remove-stopwords-later':
        site=9
    elif method=='tf-idf':
        site=10
    elif method=='tf-trunc_idf':
        site=11
    elif method=='ebc' or method=='mlc' or method=='nnpu' or method=='lightxml' or method=='diva':
        #todo debug
        # continue
        site=-1
    elif 'diva' in method:
        site=-1
    return site
           

# ???soft match score
def get_soft_match_score(soft_match_s_folder,label_type):
    filename_semantic='bert_score_dict({}).json'.format(label_type)
    filename_lexical='levenshtein_score_dict({}).json'.format(label_type)
    # laod soft match score
    print('----------------load:{}/{}--------------------'.format(filename_lexical,filename_semantic))
    semantic_softmatch_score_dict=json.load(open(soft_match_s_folder+filename_semantic,'r',encoding='utf-8'))
    print('over')
    lexical_softmatch_score_dict=json.load(open(soft_match_s_folder+filename_lexical,'r',encoding='utf-8'))
    return semantic_softmatch_score_dict,lexical_softmatch_score_dict

def get_label2index():
    availabel_candidates=np.load()

def get_threshold_lst(method,label_class):
    if label_class=='expert' or label_class=='golden':
        if method=='embeddingrank':
            threshold_lst=[0.001]
        elif method=='tf-idf':
            threshold_lst=[0.010,0.015,0.020]
        elif method=='text-rank-remove-stopwords-first':
            threshold_lst=[0.005]
        elif 'mlc' in method:
            threshold_lst=[0.10,0.15]
        elif method == 'lightxml' in method:
            threshold_lst=[0.90]
        elif 'diva-' in method or 'diva_' in method:
            threshold_lst=[0.7]
        elif 'self-training' in method:
            threshold_lst=[0.99]     
        elif 'joint' in method or 'groov' in method:
            threshold_lst=[0]
        elif method=='diva':
            threshold_lst=[0.90]
        elif method=='nnpu':
            threshold_lst=[0.85]
        else:
            threshold_lst=[0]
            print('threshold set to 0')
            
    elif label_class=='annotated+expert' or label_class=='annotated+golden':
        if method=='embeddingrank':
            threshold_lst=[0.001]
        elif method=='tf-idf':
            threshold_lst=[0.010,0.015,0.020]
        elif method=='text-rank-remove-stopwords-first':
            threshold_lst=[0.005]
        elif 'mlc' in method:
            threshold_lst=[0.10,0.15]
        elif 'self-training' in method:
            threshold_lst=[0.99]
        elif 'diva-' in method or 'diva_' in method:
            threshold_lst=[0.70]
        elif 'joint' in method or 'groov' in method:
            threshold_lst=[0]
        elif method=='diva' or method=='lightxml':
            threshold_lst=[0.90]
        elif method=='nnpu':
            threshold_lst=[0.95]
        else:
            threshold_lst=[0]
            print('threshold set to 0')
    return threshold_lst
        
    
def main(method,label_class,test_file_name,eval_metrics,joint_func=False,iter=False,iter_num=0,f1_threshold_dict=None,threshold_lst=None):
    # get trained labels for propensity cal
    # label_class_simple golden / expert
    if f1_threshold_dict!=None:
    # sort f1_threshold_dict by f1_threshold
        sort_f1_threshold_dict=sorted(f1_threshold_dict.items(),key=lambda x:x[1]['f1'],reverse=True)
        # 选取top3的threshold
        top3_f1_threshold_lst=[]
        for i in range(3):
            top3_f1_threshold_lst.append(sort_f1_threshold_dict[i][1]['threshold'])
        max_thres=np.array(top3_f1_threshold_lst).max()
        min_thres=np.array(top3_f1_threshold_lst).min()
        interval=(max_thres-min_thres)/3
        thres_mid1=min_thres+interval
        thres_mid2=min_thres+interval*2
        threshold_lst=[min_thres,thres_mid1,thres_mid2,max_thres]
    
    if threshold_lst!=None:
        threshold_lst=threshold_lst
    f1_threshold_dict={}
    
    if 'golden' in label_class:
        label_class_simple='golden'
    elif 'expert' in label_class:
        label_class_simple='expert'
    print('label_type:{}---methods:{}'.format(label_class,method))
    # if iter:
    #     json_file_name = 'iter/{}/baseline-{}({}).json'.format(label_class,method,label_class)
    #     trained_labels_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/trained_labels/trained_labels_{}_iter{}.txt'.format(label_class_simple,iter_num)
    # else:
    json_file_name = '{}/baseline-{}({}).json'.format(label_class,method,label_class)
    trained_labels_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/trained_labels/trained_labels_{}.txt'.format(label_class_simple)

    label2index=np.load('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/trained_labels/label2index.npy',allow_pickle=True).item()
    # load trained labels
    len_label2index=len(label2index)
    label2index['unk']=len_label2index
    all_train_labels=get_all_trained_labels(trained_labels_path,label2index)

    
    # padding all_train_labels to len(label2index) with value pad_val
    
    # labels_count=count_labels(all_train_labels)
    
    jain_invs_p_dict=Jain_et_al_inverse_propensity(all_train_labels,label2index)
    # jain_invs_p_dict.pad(a, (0, len(label2index)-len(all_train_labels)), 'constant', constant_values=(0, pad_val))
    
    all_positive_flag,exact_match_flag=False,False
    # label_class=args.label_type
    # method=args.methods
    if method=='all_positive':
        all_positive_flag=True
    elif method=='exact_match':
        exact_match_flag=True
    if method=='groov':
        groov_flag=True
    else:
        groov_flag=False
    # eval_metrics=args.eval_metrics
    print('recent eval metrics:',eval_metrics)
    
    soft_match_s_folder='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/soft_match_score/'
    semantic_score_dict,lexical_score_dict=get_soft_match_score(soft_match_s_folder,label_class)
    print('------soft match score loaded------')
    # song_info_dict=json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/test_song_infos_new2.json','r'))
    song_info_dict=json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/test_song_infos_newbs.json','r'))
    print('-----song info(labels) loaded-----')
    train_labels_dict=json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/iter_labels/all_trained_labels(golden-expert).json','r'))
    
    
    if 'golden' in label_class:
        # 所有golden label
        golden_vec_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/golden_dict_1536_300_pca512.npy'
        golden_vec_dict=np.load(golden_vec_path,allow_pickle=True).item()
        golden_labels=list(golden_vec_dict.keys())
        labels_set=golden_labels
        trained_labels=train_labels_dict['golden']
    
    #所有candidate label 
    else:
        vec_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/candidate_words_context_embedding_dict_1536_300.npy'
        vec_dict=np.load(vec_path,allow_pickle=True).item()
        all_candidates=list(vec_dict.keys())
        #所有expert tags
        all_tags_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/iter_labels/all_tags.json'
        # load all tags
        all_tags=json.load(open(all_tags_path,'r'))
        # 选取有wordvector的expert tags
        labels_set=list(set(all_tags).intersection(set(all_candidates)))
        trained_labels=train_labels_dict['expert']
        trained_labels=list(set(trained_labels).intersection(set(all_candidates)))
    print('------labels set(trained labels) loaded------')
    
    # load 测试数据
    data_lst=load_data(test_file_name)

    # topk
    eval_type=eval_metrics
    topk_candidates=[1,3,5,10,20,30,40]
    if all_positive_flag or exact_match_flag:
        topk_candidates=[1]
    topk=topk_candidates[0]
    # propensity score
    # inverse_p_dict=cal_propensity_score(label_class,data_lst)
    inverse_p_dict=jain_invs_p_dict
    # save path 但现在只输出到界??
    folder_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_evaluate_baseline/final_json_{}/'

    # excle
    # workbook=openpyxl.Workbook()
    # worksheet=workbook.active
    # worksheet.title='result'
    all_resualt_df=pd.DataFrame([{'method':method}])
    if eval_type=='topk':
        print('in topk term....')
        site=method2site(method)
        for topk in topk_candidates:
            
            #?label_class,data_lst,topk,site,inverse_p_dict,label2index,jain_pl,threshold,device,l2index,label_embeddings,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName
            resualt,topk_f1_threshold_dict=cal_score_topk(label2index,labels_set, song_info_dict, label_class, data_lst, topk, site, inverse_p_dict, semantic_score_dict, lexical_score_dict, trained_labels,all_positive_flag,exact_match_flag)
            info_str='@{}'.format(topk)
            new_resualt=dict_keys_add_info(resualt, info_str)
            new_resualt['method']=method
            print('topk:{} anaylis finished...'.format(topk))
            new_resualt_df=pd.DataFrame([new_resualt])
            
            #condate new_resualt_df to all_resualt_df
            all_resualt_df=pd.merge(all_resualt_df,new_resualt_df,on='method',how='outer')
            f1_threshold_dict[topk]=topk_f1_threshold_dict

        # final_eval_dict={}
        # final_eval_dict['method']=method
        # final_eval_dict['label_class']=label_class
        # final_eval_dict['detail']=resault_dict
        
        # json.dump(final_eval_dict,open(folder_path.format(eval_metrics)+json_file_name,'w'))
        # save_json(final_eval_dict,folder_path+json_file_name)
        # print('json saved...')
                # with open("baseline_eval_topk.txt", "a") as fp:
                    # д??????????
                    # if site==3 and topk==1:
                    #     fp.write(''.join(['-------------------------------------------\n???????????',current_time()]))
                    # method_str='method:'+method+'\n'
                    # fp.write(method_str)
                    # fp.write(json.dumps(evaluate_score_dict[method], ensure_ascii=False))
                    # fp.write('\ncal by jain...\n')
                    # fp.write(json.dumps(jain_info_dict, ensure_ascii=False))
                    # fp.write('\n')
    
    elif eval_type=='threshold':
        print('in threshold term....')
        # resualt_dict={} treshodl :resault info
        resualt_dict={}
        for threshold in threshold_lst:
            site=method2site(method)
            #?label_class,data_lst,topk,site,inverse_p_dict,label2index,jain_pl,threshold,device,l2index,label_embeddings,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName
            resualt=cal_score_threshold(method,label2index,threshold,labels_set, song_info_dict, label_class, data_lst, site, inverse_p_dict, semantic_score_dict, lexical_score_dict, trained_labels,all_positive_flag,exact_match_flag,joint_func,groov_flag)
            print('begin preparing for json...')
            resualt_dict[threshold]=resualt
        # sort resualt dict by p / r /f1 / psp
        # resualt_dict_sort_by_p=sorted(resualt_dict.items(),key=lambda x:x[1]['p'],reverse=True)
        # best_treshold_for_p,resualt_info_best_p=resualt_dict_sort_by_p[0]
        # resualt_info_best_p['threshold']=best_treshold_for_p
        # info_str='(threshold-best p)'
        # resualt_info_best_p=dict_keys_add_info(resualt_info_best_p, info_str)
        # resualt_info_best_p['method']=method
        # info_best_p_df=pd.DataFrame([resualt_info_best_p])
        # all_resualt_df=pd.merge(all_resualt_df,info_best_p_df,on='method',how='outer')
        
        # resualt_dict_sort_by_r=sorted(resualt_dict.items(),key=lambda x:x[1]['r'],reverse=True)
        # best_treshold_for_r,resualt_info_best_r=resualt_dict_sort_by_r[0]
        # resualt_info_best_r['threshold']=best_treshold_for_r
        # info_str='(threshold-best r)'
        # resualt_info_best_r=dict_keys_add_info(resualt_info_best_r, info_str)
        # resualt_info_best_r['method']=method
        # info_best_r_df=pd.DataFrame([resualt_info_best_r])
        # all_resualt_df=pd.merge(all_resualt_df,info_best_r_df,on='method',how='outer')
        
        resualt_dict_sort_by_f1=sorted(resualt_dict.items(),key=lambda x:x[1]['f1'],reverse=True)
        best_treshold_for_f1,resualt_info_best_f1=resualt_dict_sort_by_f1[0]
        resualt_info_best_f1['threshold']=best_treshold_for_f1
        info_str='(threshold-best f1)'
        resualt_info_best_f1=dict_keys_add_info(resualt_info_best_f1, info_str)
        resualt_info_best_f1['method']=method
        info_best_f1_df=pd.DataFrame([resualt_info_best_f1])
        all_resualt_df=pd.merge(all_resualt_df,info_best_f1_df,on='method',how='outer')
        
        # resault_dict_sort_by_psp=sorted(resualt_dict.items(),key=lambda x:x[1]['psp'],reverse=True)
        # best_treshold_for_psp,resualt_info_best_psp=resault_dict_sort_by_psp[0]
        # resualt_info_best_psp['threshold']=best_treshold_for_psp
        # info_str='(threshold-best psp)'
        # resualt_info_best_psp=dict_keys_add_info(resualt_info_best_psp, info_str)
        # resualt_info_best_psp['method']=method
        # info_best_psp_df=pd.DataFrame([resualt_info_best_psp])
        # all_resualt_df=pd.merge(all_resualt_df,info_best_psp_df,on='method',how='outer')
        
        # resualt_dict_new={}
        # resault_dict_new['method']=method
        # resault_dict_new['label_class']=label_class
        # resualt_dict_new['best_threshold_info']={}
        # resualt_dict_new['best_threshold_info']['best_treshold_for_p']={}
        # resualt_dict_new['best_threshold_info']['resualt_info_best_p']['bset_threshold']=best_treshold_for_p
        # resault_dict_new['best_threshold_info']['resualt_info_best_p']['info_dict']=resualt_info_best_p
        
        # resault_dict_new['best_threshold_info']['best_treshold_for_r']={}
        # resault_dict_new['best_threshold_info']['resualt_info_best_r']['bset_threshold']=best_treshold_for_r
        # resault_dict_new['best_threshold_info']['resualt_info_best_r']['info_dict']=resualt_info_best_r
        
        # resault_dict_new['best_threshold_info']['best_treshold_for_f1']={}
        # resault_dict_new['best_threshold_info']['resualt_info_best_f1']['bset_threshold']=best_treshold_for_f1
        # resault_dict_new['best_threshold_info']['resualt_info_best_f1']['info_dict']=resualt_info_best_f1
        
        # resault_dict_new['best_threshold_info']['best_treshold_for_psp']={}
        # resault_dict_new['best_threshold_info']['resualt_info_best_psp']['bset_threshold']=best_treshold_for_psp
        # resault_dict_new['best_threshold_info']['resualt_info_best_psp']['info_dict']=resualt_info_best_psp
        
        # json.dump(resault_dict_new,open(folder_path.format(eval_metrics)+json_file_name,'w'))
        # save_json(final_eval_dict,folder_path+json_file_name)
        print('method :{} best threshold cal'.format(method))
        
        #save resault_dict_new to json
        
    return all_resualt_df,f1_threshold_dict

if __name__ == '__main__':
    iter=False
    ablition=False
    have_iteration=9
    # golden / expert / annotated+golden / annotated+experts
    save_folder='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_evaluate_baseline/final_csv4/{}/'
    # method
    file_name='{}.csv'
    
    test_file_name_ebc='diva_data/test_iter0_diva_baseline_goldenexpert.json'
    test_file_name_diva='diva_data/test_iter_based_expert_0.json'
    
    test_file_name_lightxml_golden='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/baseline/LightXML/test_eval.json'
    test_file_name_lightxml_expert='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/baseline/LightXML/test_eval_expert_new.json'

    test_file_joint_func='diva_data/test_iter1_diva_golden_semantic_valid_001.json'
    
    test_file_joint_func2='diva_data/test_iter1_diva_golden_update_top03.json'
    test_file_joint_func3='diva_data/test_iter1_diva_golden_update_top04.json'
    test_file_joint_func4='diva_data/test_iter1_diva_golden_update_top05.json'
    test_file_joint_func5='diva_data/test_iter1_diva_golden_update_top06.json'
    test_file_joint_func_final='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter1_yaoyao_0519_diva_03.json'
    
    test_diva_static=test_file_joint_func_final
    # test_file_joint_func_final='diva_data/test_iter1_0503_old_diva_dumping05_light.json'
    
    test_file_name_diva_golden='diva_data/test_iter{}_diva_baseline_expertexpert.json'
    
    # test_diva_iter4='diva_data/diva_golden.json'
    test_diva_iter1='diva_data/test_iter2_0503_old_diva_dumping05.json'
    test_diva_iter2='diva_data/test_iter3_0503_old_diva_dumping05.json'
    test_diva_iter3='diva_data/test_iter4_0503_old_diva_dumping05.json'
    test_diva_iter4='diva_data/test_iter5_0503_old_diva_dumping05.json'
    test_diva_iter5='diva_data/test_iter6_0503_old_diva_dumping05.json'
    test_diva_iter6='diva_data/test_iter7_0503_old_diva_dumping05.json'
    test_diva_iter7='diva_data/test_iter8_0503_old_diva_dumping05.json'
    test_diva_iter8='diva_data/test_iter9_0503_old_diva_dumping05.json'
    test_diva_iter9='diva_data/test_iter10_0503_old_diva_dumping05.json'
    test_diva_iter10='diva_data/test_iter11_0503_old_diva_dumping05.json'
    
    bs_diva_03='diva_data/test_iter2_yaoyao_0519_diva_03.json'
    bs_diva_03_iter='diva_data/test_iter{}_yaoyao_0519_diva_03.json'
    bs_diva_01='diva_data/test_iter2_0519_diva_up_01.json'
    bs_diva_05='diva_data/test_iter2_yaoyao_0519_diva_05.json'
    bs_diva_07='diva_data/test_iter2_yaoyao_0519_diva_07.json'
    bs_diva_light='diva_data/test_iter2_0519_diva_03_light.json'
    bs_diva_light_iter='diva_data/test_iter{}_0519_diva_03_light.json'
    bs_diva_ebc_thres='diva_data/test_iter2_0519_diva_ebc_thres_{}.json'
    
    bs_ablition_dict={
        'bs_diva_only_joint':'diva_data/test_iter2_0519_diva_03_only_joint.json',
        # 'bs_diva_no_novelty':'diva_data/test_iter2_0519_diva_no_novelty.json',
        # 'bs_diva_no_tf_idf':'diva_data/test_iter2_0519_diva_no_tf_idf.json',
        # 'bs_diva_no_valid_disc':'diva_data/test_iter2_0519_diva_no_valid_disc.json',
        # 'bs_diva_no_valid_semantic':'diva_data/test_iter2_0519_diva_no_valid_semantic.json',
        # 'bs_diva_no_valid':'diva_data/test_iter2_0519_diva_no_valid.json',
        # 'bs_diva_no_PE':'diva_data/test_iter2_0519_diva_no_PE.json'
    }
    # bs_diva_iter1_01='diva_data/test_iter2_0518_diva_base_01.json'
    # bs_diva_iter2_01='diva_data/test_iter2_0518_diva_base_01.json'
    # bs_diva_iter3_01='diva_data/test_iter3_0518_diva_base_01.json'
    # bs_diva_iter4_01='diva_data/test_iter4_0518_diva_base_01.json'
    # bs_diva_iter1_05='diva_data/test_iter2_0518_diva_base_05.json'
    # bs_diva_iter1_005='diva_data/test_iter2_0517_diva_base_005.json'
    # bs_diva_no_novelty='diva_data/test_iter2_0518_diva_base_01_no_novelty.json'
    # bs_diva_no_tf_idf='diva_data/test_iter2_0518_diva_base_01_no_tf_idf.json'
    # bs_diva_no_valid_disc='diva_data/test_iter2_0518_diva_base_01_no_valid_disc.json'
    # bs_diva_no_valid='diva_data/test_iter2_0518_diva_base_01_no_valid.json'
    # test_diva_iter4='diva_data/test_iter4_0503_noon_diva_golden.json'
    
    # test_diva_light='diva_data/test_iter11_0503_old_diva_dumping05_light.json'
    test_diva_light_2='diva_data/test_iter3_0505_diva_light.json'
    test_diva_light_1='diva_data/test_iter2_0505_diva_light.json'
    test_diva_light_3='diva_data/test_iter4_0505_diva_light.json'
    
    test_diva_others='diva_data/test_iter1_0503_diva_golden_dumping_05.json'
    
    test_file_diva_self_training='diva_data/test_iter_self_training.json'
    # test_file_ebc_self_training='diva_data/test_iter_golden_self_training.json'
    test_file_ebc_self_training='diva_data/diva_self_training_golden_10.json'
    
    #todo chat gpt
    test_file_diva_chat_gpt='diva_data/test_iter_based_chat_gpt.json'
    
    test_file_groov='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/baseline/GROOV/test_groov_golden.json'
    
    test_diva_all_tags='diva_data/test_diva_with_all_tags.json'
    test_self_training_diva_all_tags='diva_data/test_diva_with_all_tags(self-training).json'
    
    label_classes_unsupervise=['golden','annotated+golden','annotated+expert','expert']
    label_classes_golden=['golden','annotated+golden']
    label_classes_expert=['expert','annotated+expert']
    
    # method : f_path 
    ablition_study_dict={'diva_no_tf_idf':'diva_data/test_iter2_0504_diva_no_tf_idf.json',\
                        #  'diva_no_novelty':'diva_data/test_iter2_0504_diva_no_novelty.json',\
                        #  'diva_no_valid_disc':'diva_data/test_iter2_0504_diva_no_valid_disc.json',\
                        #  'diva_no_valid_semantic':'diva_data/test_iter2_0504_diva_no_valid_semantic.json',\
                        #  'diva_no_valid':'diva_data/test_iter2_0504_diva_no_valid.json',\
                        #  'diva_only_joint':'diva_data/test_iter2_0504_diva_only_joint.json',\
                        #  'diva_update_thres025':'diva_data/test_iter2_0504_diva_pram_update_thres025.json',\
                        #  'diva_update_thres075':'diva_data/test_iter2_0504_diva_pram_update_thres075.json',\
                        #  'diva_dumping_no':'diva_data/test_iter2_0504_diva_pram_dumping_no.json',\
                        #  'diva_dumping_025':'diva_data/test_iter2_0504_diva_pram_dumping_025.json',\
                        #  'diva_dumping_075':'diva_data/test_iter2_0504_diva_pram_dumping075.json',\
                            'diva_update_thres01':'diva_data/test_iter4_0504_diva_pram_update_thres01.json',\
                         }
    
    baseline_methods_unsuperviase=['all_positive','exact_match','tf-idf','rf-idf','tf-trunc_idf','rf-trunc_idf','embeddingrank','text-rank-remove-stopwords-first','text-rank-remove-stopwords-later']
    # baseline_methods=['mlc']
    # args.methods='exact_match'
    if iter:
        save_folder='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_evaluate_baseline/final_csv/iter/{}/'
        for label_class in label_classes_expert:
            for i in range(have_iteration):
                method='diva_iter{}'.format(i)
                all_resualt_df=pd.DataFrame([{'method':method}])
                print('begin evaluate method:',method,'label_class:',label_class,'iter:',i)
                for eval_metrics in ['topk','threshold']:
                    print('eval_metrics:',eval_metrics)
                    tmp_resualt_df=main(method,label_class,test_file_name_diva_golden.format(i),iter=True,eval_metrics=eval_metrics)
                    all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
                all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
    if ablition:
         save_folder='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_evaluate_baseline/final_csv4/{}/'
         for method in ablation_file_dict.keys():
            all_resualt_df=pd.DataFrame([{'method':method}])
            print('begin evaluate method:',method)
            for eval_metrics in ['topk','threshold']:
                print('eval_metrics:',eval_metrics)
                tmp_resualt_df=main(method,'expert',ablation_file_dict[method],iter=False,eval_metrics=eval_metrics)
                all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
            all_resualt_df.to_csv(save_folder.format(method)+file_name.format(method),index=False)
    else:
        # debug embeddingrank
        # for label_class in label_classes_expert:
        #     method='embeddingrank'
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     for eval_metrics in ['topk','threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         tmp_resualt_df=main(method,label_class,test_file_name_ebc,iter=False,eval_metrics=eval_metrics)
        #         all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
                

        # # # todo golden ebc
        # for label_class in label_classes_golden:
        #     method='ebc'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     for eval_metrics in ['topk','threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         tmp_resualt_df=main(method,label_class,test_file_name_ebc,iter=False,eval_metrics=eval_metrics)
        #         all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
            # main(method,label_class,test_file_name_ebc)
        
        # todo diva expert
        # for label_class in label_classes_expert:
        #     method='diva'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # todo golden diva
        # for label_class in label_classes_golden:
        #     method='diva'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_name_ebc,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
       
        # todo diva-self-training
        
        #todo golden diva-self-training
        # for label_class in label_classes_golden:
        #     method='diva-self-training'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_ebc_self_training,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # for label_class in label_classes_golden:
        #     method='diva-10'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_diva_iter10,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        #todo diva iter over
        ebc_thres_lst=['091','093','099','097']
        for i in range(4):
            # iter_tmp=iter+2
            for label_class in label_classes_golden:
                method='bs-diva-ebc-thres{}'.format(ebc_thres_lst[i])
                print('begin evaluate method:',method,'label_class:',label_class)
                all_resualt_df=pd.DataFrame([{'method':method}])
                f1_threshold_dict={}
                for eval_metrics in ['threshold']:
                    print('eval_metrics:',eval_metrics)
                    if eval_metrics=='topk':
                        tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
                        all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
                        print('{}:{}'.format(method,f1_threshold_dict))
                    else:
                        threshold_lst=get_threshold_lst(label_class=label_class,method=method)
                        tmp_resualt_df,_=main(method,label_class,bs_diva_ebc_thres.format(ebc_thres_lst[i]),iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
                        all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
                all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)   
        # for label_class in label_classes_golden:
        #     method='bs-diva-05'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_05,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)       
        # for label_class in label_classes_golden:
        #     method='bs-diva-light'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_light,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # #todo diva-03
        # for label_class in label_classes_golden:
        #     method='bs-diva-03'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_03,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo diva-01
        # for label_class in label_classes_golden:
        #     method='bs-diva-01'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_01,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # #todo diva-05
        # for label_class in label_classes_golden:
        #     method='bs-diva-05'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_05,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo diva-07
        # for label_class in label_classes_golden:
        #     method='bs-diva-07'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_07,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # for label_class in label_classes_golden:
        #     method='bs-diva-01-no-valid-disc'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_no_valid_disc,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # for label_class in label_classes_golden:
        #     method='bs-diva-01-no-valid'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,bs_diva_no_valid,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # for label_class in label_classes_golden:
        #     method='diva-static'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_diva_static,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # for label_class in label_classes_golden:
        #     method='diva-joint-func'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_joint_func_final,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=True)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo lightxml
        # for label_class in label_classes_golden:
        #     method='lightxml'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_name_lightxml_golden,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo groov
        # for label_class in label_classes_golden:
        #     method='groov'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_groov,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo chatgpt
        # for label_class in label_classes_golden:
        #     method='chat_gpt'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_diva_chat_gpt,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # for label_class in label_classes_golden:
        #     method='diva'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_name_ebc,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # todo diva all_tags
        # for label_class in label_classes_golden:
        #     method='diva-joint-func'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_joint_func_final,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=True)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        #todo groov
        # for label_class in label_classes_golden:
        #     method='groov'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_groov,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # for label_class in label_classes_expert:
        #     method='diva(all_tags)'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_diva_all_tags,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        # # #todo lightxml exprt
        # for label_class in label_classes_golden:
        #     method='lightxml'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_lightxml_expert,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('{}:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_file_name_lightxml_golden,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
            # all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
            # main(method,label_class,test_file_name_nnpu)
        
        # for label_class in label_classes_golden:
        #     method='lightxml'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     for eval_metrics in ['topk','threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         tmp_resualt_df=main(method,label_class,test_file_name_lightxml_golden,iter=False,eval_metrics=eval_metrics)
        #         all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
    #    # #todo self-training exprt
    #     for label_class in label_classes_expert:
    #         method='self-training-diva'
    #         print('begin evaluate method:',method,'label_class:',label_class)
    #         all_resualt_df=pd.DataFrame([{'method':method}])
    #         f1_threshold_dict={}
    #         for eval_metrics in ['threshold']:
    #             print('eval_metrics:',eval_metrics)
    #             if eval_metrics=='topk':
    #                 tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_lightxml_expert,iter=False,eval_metrics=eval_metrics)
    #                 all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
    #                 print('method:{} f1_threshold_dict:{}'.format(method,f1_threshold_dict))
    #             else:
    #                 threshold_lst=get_threshold_lst(label_class=label_class,method=method)
    #                 tmp_resualt_df,_=main(method,label_class,test_file_diva_self_training,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst)
    #                 all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
    #         all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)
        
        # todo all tags diva
        # for label_class in label_classes_expert:
        #     method='self-training-diva(all_tags)'
        #     print('begin evaluate method:',method,'label_class:',label_class)
        #     all_resualt_df=pd.DataFrame([{'method':method}])
        #     f1_threshold_dict={}
        #     for eval_metrics in ['threshold']:
        #         print('eval_metrics:',eval_metrics)
        #         if eval_metrics=='topk':
        #             tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_self_training_diva_all_tags,iter=False,eval_metrics=eval_metrics)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #             print('method:{} f1_threshold_dict:{}'.format(method,f1_threshold_dict))
        #         else:
        #             threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #             tmp_resualt_df,_=main(method,label_class,test_self_training_diva_all_tags,iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst)
        #             all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #     all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)

        #todo eval ablition
        # for method in bs_ablition_dict.keys():
        #     for label_class in label_classes_golden:
        #         method=method
        #         print('begin evaluate method:',method,'label_class:',label_class)
        #         all_resualt_df=pd.DataFrame([{'method':method}])
        #         f1_threshold_dict={}
        #         for eval_metrics in ['threshold']:
        #             print('eval_metrics:',eval_metrics)
        #             if eval_metrics=='topk':
        #                 tmp_resualt_df,f1_threshold_dict=main(method,label_class,test_file_name_diva,iter=False,eval_metrics=eval_metrics)
        #                 all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #                 print('{}:{}'.format(method,f1_threshold_dict))
        #             else:
        #                 threshold_lst=get_threshold_lst(label_class=label_class,method=method)
        #                 tmp_resualt_df,_=main(method,label_class,bs_ablition_dict[method],iter=False,eval_metrics=eval_metrics,threshold_lst=threshold_lst,joint_func=False)
        #                 all_resualt_df=pd.merge(all_resualt_df,tmp_resualt_df,on='method',how='outer')
        #         all_resualt_df.to_csv(save_folder.format(label_class)+file_name.format(method),index=False)