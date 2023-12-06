from soft_match_metric import lexical_soft_match
from soft_match_metrics_bertscore import get_bert_score
from napkinxc.measures import Jain_et_al_inverse_propensity,psprecision_at_k,psndcg_at_k,ndcg_at_k
from sklearn.metrics import precision_score,recall_score,f1_score,ndcg_score,dcg_score
import re
import numpy as np

ScoreColName='final_score_dict_sort'
GoldenColName='song_pseudo_golden_labels'
# ?????????
LabelColName='song_annotated_labels'

def norm_song_id(song_name):
    pat=re.compile('song_id: (.*?) ')
    song_id=pat.findall(song_name)[0]
    return 'song_id_'+song_id.strip()

def soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,soft_type):
    tp=0

    tp_r=0
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
        tp_r=0
    else:
        for label in true_labels:
            if label in pred_labels:
                tp_r+=1
            else:
                # cal sot score
                tp_r+=max(soft_func(label,pred_labels),0)
                
    
    if len(pred_labels)==0:
        p=0
    else:
        p=tp/len(pred_labels)
    if len(true_labels)==0:
        r=0
    else:
        # 保证r 0-1
        r=tp_r/len(true_labels)
    # print('-----soft------len(true_labels)',len(true_labels))
    if p+r==0:
        f1=0
    else:
        f1=2*p*r/(p+r)
    return p,r,f1


# (labels_set, song_info_dict, label_class, data_lst, topk, site, inverse_p_dict, semantic_score_dict, lexical_score_dict, trained_labels,all_positive_flag,exact_match_flag
def cal_score_topk(label2index,labels_set,song_info_dict,label_class,data_lst,topk,site,inverse_p_dict,semantic_score_dict,lexical_score_dict,trained_labels,all_positive=False,exact_match=False,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName):
    print('topk calculation.....')
    """
    label_set : trained labels(seen labels)
    """
    p_lst=[]
    r_lst=[]
    f1_lst=[]
    
    jain_true_labels=[]
    jain_pred_labels=[]
    
    semantic_soft_p_lst=[]
    lexical_soft_p_lst=[]

    semantic_soft_f1_lst=[]
    lexical_soft_f1_lst=[]

    semantic_soft_recall_lst=[]
    lexical_soft_recall_lst=[]

    pred_unseen_labels=[]
    test_unseen_labels=[]
    
    # 记录topk首歌第topk个标签的分数
    topk_thres_lst=[]
    # mean val of topk_thres_lst
    topk_mean=0
    
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
        # old labels -> 开始标签
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
        # list of tuple(label_index,label_score) | (int/str,float)
        pred_labels_jain=[]
        true_labels_jain=[]
        if all_positive:
            pred_labels=all_candidates
            topk=len(all_candidates)
        elif exact_match:
            pred_labels=candidates_in_labels
            topk=len(candidates_in_labels)
        else:
            # get pred_labels
            pred_labels=[]
            for i in range(topk):
                pred_labels.append(candidate_sort[i][0])
                if i==topk-1:
                    topk_thres_lst.append(candidate_sort[i][1][site])
                
        # prepare true labels and pred labels
        for label in true_labels:
            true_labels_jain.append(label2index[label])
        for label in pred_labels:
            if all_positive or exact_match:
                tmp_tuple=(label2index[label],1)
            else:
                tmp_tuple=(label2index[label],score_dict[label][site])
            pred_labels_jain.append(tmp_tuple)
        jain_true_labels.append(true_labels_jain)
        jain_pred_labels.append(pred_labels_jain)
            
        # song_new_labels=list(set(pred_labels)-set(old_labels))
        # song_unseen_new_labels=list(set(song_new_labels)-set(trained_labels))
        # song_new_labels_len.append(len(song_new_labels))
        # song_new_labels_len.append(len(song_unseen_new_labels))
        
        # if site==11:
        test_unseen_labels+=true_labels
        pred_unseen_labels+=pred_labels
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
        
        predict_lst=np.array(predict_lst)
        observed_lst=np.array(observed_lst)
        score_lst=np.array(score_lst)

        p_tmp=precision_score(observed_lst,predict_lst,average='binary',zero_division=0)
        p_lst.append(p_tmp)
        recall_tmp=recall_score(observed_lst,predict_lst,average='binary',zero_division=0)
        r_lst.append(recall_tmp)
        f1_tmp=f1_score(observed_lst,predict_lst,average='binary',zero_division=0)        
        f1_lst.append(f1_tmp)
        
        #todo get semantic soft p,r,f1
        semantic_soft_p,semantic_soft_r,semantic_soft_f1=soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,soft_type='semantic')
        semantic_soft_p_lst.append(semantic_soft_p)
        semantic_soft_recall_lst.append(semantic_soft_r)
        semantic_soft_f1_lst.append(semantic_soft_f1)
        lexical_soft_p,lexical_soft_r,lexical_soft_f1=soft_metrics(song_name,true_labels,pred_labels,lexical_score_dict,soft_type='lexical')
        lexical_soft_p_lst.append(lexical_soft_p)
        lexical_soft_recall_lst.append(lexical_soft_r)
        lexical_soft_f1_lst.append(lexical_soft_f1)
        if recall_tmp>semantic_soft_r or recall_tmp>lexical_soft_r:
            print('hhhhhhhhh',recall_tmp,semantic_soft_r,lexical_soft_r)
        
        
    # get nlsr
    test_unseen_labels=list(set(test_unseen_labels)-set(trained_labels))
    pred_unseen_labels=list(set(pred_unseen_labels)-set(trained_labels))
    if len(test_unseen_labels)==0:
        nlsr=len(pred_unseen_labels)
    else:
        nlsr=len(pred_unseen_labels)/len(test_unseen_labels)

    p=np.array(p_lst).mean()
    semantic_soft_p=np.array(semantic_soft_p_lst).mean()
    lexical_soft_p=np.array(lexical_soft_p_lst).mean()
    recall=np.array(r_lst).mean()
    semantic_soft_recall=np.array(semantic_soft_recall_lst).mean()
    lexical_soft_recall=np.array(lexical_soft_recall_lst).mean()
    f1=np.array(f1_lst).mean()
    semantic_soft_f1=np.array(semantic_soft_f1_lst).mean()
    lexical_soft_f1=np.array(lexical_soft_f1_lst).mean()
    
    
    psp,_,_=psprecision_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=False,inv_ps=inverse_p_dict)
    psp=psp[-1]
    norm_psp,_,_=psprecision_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=True,inv_ps=inverse_p_dict)
    norm_psp=norm_psp[-1]
    ndcg=ndcg_at_k(jain_true_labels,jain_pred_labels,k=topk)
    ndcg=ndcg[-1]
    # ndcg=np.array(ndcg_lst).mean()
    # psndcg=np.array(psndcg_lst).mean()
    psndcg,_,_=psndcg_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=False,inv_ps=inverse_p_dict)
    psndcg=psndcg[-1]
    norm_psndcg,_,_=psndcg_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=True,inv_ps=inverse_p_dict)
    norm_psndcg=norm_psndcg[-1]
    
    # per_song_new_label_num=np.array(song_new_labels_len).mean()
    # per_song_unseen_new_label_num=np.array(song_unseen_new_labels_len).mean()
    
    if len(topk_thres_lst)>0:
        topk_mean=np.array(topk_thres_lst).mean()
    f1_threshold_dict={'f1':f1,'threshold':topk_mean}
    
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
    
    return return_info,f1_threshold_dict

# label2index,threshold,labels_set, song_info_dict, label_class, data_lst, site, inverse_p_dict, semantic_score_dict, lexical_score_dict, trained_labels,all_positive_flag,exact_match_flag)
def cal_score_threshold(method,label2index,threshold,labels_set,song_info_dict,label_class,data_lst,site,inverse_p_dict,semantic_score_dict,lexical_score_dict,trained_labels,all_positive=False,exact_match=False,joint_func=False,groov_flag=False,ScoreColName=ScoreColName,LabelColName=LabelColName,GoldenColName=GoldenColName):
    print('topk calculation.....')
    """
    label_set : trained labels(seen labels)
    """
    print(method)
    p_lst=[]
    r_lst=[]
    f1_lst=[]
    
    psp_lst=[]
    best_psp_lst=[]
    
    ndcg_lst=[]
    psndcg_lst=[]
    best_psndcg_lst=[]
    
    
    semantic_soft_p_lst=[]
    lexical_soft_p_lst=[]

    semantic_soft_f1_lst=[]
    lexical_soft_f1_lst=[]

    semantic_soft_recall_lst=[]
    lexical_soft_recall_lst=[]

    pred_unseen_labels=[]
    test_unseen_labels=[]
    pred_all_labels=[]
    
    np_lst=[]
    ns_lst=[]
    nl_lst=[]
    jac_s_lst=[]
    
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
        # old labels -> 开始标签
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
        # list of tuple(label_index,label_score) | (int/str,float)
        pred_labels_jain=[]
        true_labels_jain=[]
        topk=0
        if all_positive:
            pred_labels=all_candidates
            topk=len(all_candidates)
        elif exact_match:
            pred_labels=candidates_in_labels
            topk=len(candidates_in_labels)
        elif joint_func:
            pred_labels=song_com['song_all_silver_labels']
            pred_labels=list(set(pred_labels)-set(golden_labels))
            pred_labels_discard=[]
            for i in range(len(candidate_sort)):
                if candidate_sort[i][1][site]>=0.95:
                    pred_labels_discard.append(candidate_sort[i][0])
                else:
                    break
            pred_labels=list(set(pred_labels)-set(pred_labels_discard))
            topk=len(pred_labels)
        elif groov_flag:
            pred_labels=song_com['groov_pred_naive']
            topk=len(pred_labels)
        elif method=='chat_gpt':
            pred_labels=song_com['chat_gpt_preds']
            topk=len(pred_labels)
        elif method=='diva-static':
            pred_labels=song_com['song_all_silver_labels']
            for i in range(len(candidate_sort)):
                if candidate_sort[i][1][site]>=threshold:
                    pred_labels.append(candidate_sort[i][0])
                else:
                    break
            pred_labels=list(set(pred_labels))
            topk=len(pred_labels)
        else:
            # get pred_labels
            pred_labels=[]
            for i in range(len(candidate_sort)):
                if candidate_sort[i][1][site]>=threshold:
                    pred_labels.append(candidate_sort[i][0])
                    topk+=1
                else:
                    break
        np_s=len(set(pred_labels)-set(true_labels))
        ns_s=len((set(pred_labels)-set(true_labels))&set(trained_labels))
        nl_s=len(set(pred_labels)-set(trained_labels))
        jac_s_up=len(set(true_labels)&(set(pred_labels)|set(golden_labels)))
        jac_s_down=len(set(true_labels)|(set(pred_labels)|set(golden_labels)))
        if jac_s_down==0:
            jac_s=0
        else:
            jac_s=jac_s_up/jac_s_down
        jac_s_lst.append(jac_s)
        np_lst.append(np_s)
        ns_lst.append(ns_s)
        nl_lst.append(nl_s)
        pred_all_labels.extend(pred_labels)
        
        # prepare true labels and pred labels
        for label in true_labels:
            true_labels_jain.append(label2index[label])
        for label in pred_labels:
            if all_positive or exact_match or joint_func or groov_flag or method=='chat_gpt':
                if label not in label2index:
                    tmp_tuple=(label2index['unk'],1)
                else:
                    tmp_tuple=(label2index[label],1)
            else:
                tmp_tuple=(label2index[label],score_dict[label][site])
            pred_labels_jain.append(tmp_tuple)
        jain_true_labels=[]
        jain_pred_labels=[]
        jain_true_labels.append(true_labels_jain)
        jain_pred_labels.append(pred_labels_jain)
        
        if topk==0:
            tmp_psp,best_tmp_psp,tmp_ndcg,best_tmp_psndcg,tmp_psndcg=0,0,0,0,0
        else:
            tmp_psp,_,best_tmp_psp=psprecision_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=False,inv_ps=inverse_p_dict)
            tmp_psp,best_tmp_psp=tmp_psp[-1],best_tmp_psp[-1]
            tmp_ndcg=ndcg_at_k(jain_true_labels,jain_pred_labels,k=topk)
            tmp_ndcg=tmp_ndcg[-1]
            tmp_psndcg,_,best_tmp_psndcg=psndcg_at_k(jain_true_labels,jain_pred_labels,k=topk,normalize=False,inv_ps=inverse_p_dict)
            tmp_psndcg,best_tmp_psndcg=tmp_psndcg[-1],best_tmp_psndcg[-1]
        
        psp_lst.append(tmp_psp)
        best_psp_lst.append(best_tmp_psp)
        
        ndcg_lst.append(tmp_ndcg)
        psndcg_lst.append(tmp_psndcg)
        best_psndcg_lst.append(best_tmp_psndcg)
        
        # song_new_labels=list(set(pred_labels)-set(old_labels))
        # song_unseen_new_labels=list(set(song_new_labels)-set(trained_labels))
        # song_new_labels_len.append(len(song_new_labels))
        # song_new_labels_len.append(len(song_unseen_new_labels))
        
        # if site==11:
        test_unseen_labels+=true_labels
        pred_unseen_labels+=pred_labels
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

        # print('predict_lst',type(predict_lst))
        observed_lst=np.array(observed_lst)
        predict_lst=np.array(predict_lst)
        score_lst=np.array(score_lst)

        p_tmp=precision_score(observed_lst,predict_lst,average='binary',zero_division=0)
        p_lst.append(p_tmp)
        recall_tmp=recall_score(observed_lst,predict_lst,average='binary',zero_division=0)
        r_lst.append(recall_tmp)
        f1_tmp=f1_score(observed_lst,predict_lst,average='binary',zero_division=0)        
        f1_lst.append(f1_tmp)
        
        #todo get semantic soft p,r,f1
        if method=='groov' or method=='chat_gpt':
            semantic_soft_p,semantic_soft_r,semantic_soft_f1=0,0,0
            lexical_soft_p,lexical_soft_r,lexical_soft_f1=0,0,0
            semantic_soft_p_lst.append(semantic_soft_p)
            semantic_soft_recall_lst.append(semantic_soft_r)
            semantic_soft_f1_lst.append(semantic_soft_f1)
            lexical_soft_p_lst.append(lexical_soft_p)
            lexical_soft_recall_lst.append(lexical_soft_r)
            lexical_soft_f1_lst.append(lexical_soft_f1)
        else:
            semantic_soft_p,semantic_soft_r,semantic_soft_f1=soft_metrics(song_name,true_labels,pred_labels,semantic_score_dict,soft_type='semantic')
            semantic_soft_p_lst.append(semantic_soft_p)
            semantic_soft_recall_lst.append(semantic_soft_r)
            semantic_soft_f1_lst.append(semantic_soft_f1)
            lexical_soft_p,lexical_soft_r,lexical_soft_f1=soft_metrics(song_name,true_labels,pred_labels,lexical_score_dict,soft_type='lexical')
            lexical_soft_p_lst.append(lexical_soft_p)
            lexical_soft_recall_lst.append(lexical_soft_r)
            lexical_soft_f1_lst.append(lexical_soft_f1)
     
        
        
    # get nlsr
    test_unseen_labels=list(set(test_unseen_labels)-set(trained_labels))
    pred_unseen_labels=list(set(pred_unseen_labels)-set(trained_labels))
    y_new=len(set(pred_all_labels)-set(trained_labels))
    if len(test_unseen_labels)==0:
        nlsr=len(pred_unseen_labels)
    else:
        nlsr=len(pred_unseen_labels)/len(test_unseen_labels)

    p=np.array(p_lst).mean()
    semantic_soft_p=np.array(semantic_soft_p_lst).mean()
    lexical_soft_p=np.array(lexical_soft_p_lst).mean()
    recall=np.array(r_lst).mean()
    semantic_soft_recall=np.array(semantic_soft_recall_lst).mean()
    lexical_soft_recall=np.array(lexical_soft_recall_lst).mean()
    f1=np.array(f1_lst).mean()
    semantic_soft_f1=np.array(semantic_soft_f1_lst).mean()
    lexical_soft_f1=np.array(lexical_soft_f1_lst).mean()
    
    nps=np.array(np_lst).mean()
    nss=np.array(ns_lst).mean()
    nls=np.array(nl_lst).mean()
    jacs=np.array(jac_s_lst).mean()
    
    ndcg=np.array(ndcg_lst).mean()
    
    psp=np.array(psp_lst).mean()
    if np.array(best_psp_lst).sum()==0:
        norm_psp=0
    else:
        norm_psp=np.array(psp_lst).sum()/np.array(best_psp_lst).sum()
    
    psndcg=np.array(psndcg_lst).mean()
    
    if np.array(best_psndcg_lst).sum()==0:
        norm_psndcg=0
    else:
        norm_psndcg=np.array(psndcg_lst).sum()/np.array(best_psndcg_lst).sum()
    
    
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
    
    return_info['np']=nps
    return_info['ns']=nss
    return_info['nl']=nls
    return_info['y_new']=y_new
    return_info['jac_s']=jacs
    
    return return_info