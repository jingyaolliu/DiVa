# lexical and semantic
import Levenshtein
import numpy as np
import json
# 0 golden, 1 expert ,2 annotated+golden,3 annotated+expert
labels_type=3
print('current label_type:3')

# 计算中文字符串间的编辑距离
def lexical_soft_match(pred_label, true_labels):
    lexical_dists=[]
    for true_label in true_labels:
        lexical_dists.append(Levenshtein.distance(pred_label, true_label)/max(len(pred_label),len(true_label)))
    lexical_dists=np.array(lexical_dists)
    return 1-lexical_dists.min()


# print(Levenshtein.distance(str1, str2))    

if __name__ == "__main__":
    saved_folder_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/soft_match_score/'
    if labels_type==0:
        labels_type_str='golden'
    elif labels_type==1:
        labels_type_str='expert'
    elif labels_type==2:
        labels_type_str='annotated+golden'
    else:
        labels_type_str='annotated+expert'
    file_name='levenshtein_score_dict({}).json'.format(labels_type_str)
    # test lexical (levenshtein distance)
    test_file_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter0_5.json'
    # {song_name|candidate:lexical_soft_match_score}
    Candidates_dict_path='/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/candidate_words_context_embedding_dict_1536_300.npy'
    all_candidates_set=set(list(np.load(Candidates_dict_path, allow_pickle=True).item().keys()))
    song_info_dict=json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/test_song_infos.json','r'))
    lexical_soft_match_score_dict={}
    # todo test
    # str1 = "中国"
    # str2 = "中国"
    # print(Levenshtein.distance(str1, str2))
    data_lst=json.load(fp=open(test_file_path,'r'))
    for data in data_lst:
        song_name=data['song_name']
        if song_name not in song_info_dict:
            continue
        annotated_labels=song_info_dict[song_name]['annotated_labels']
        expert_labels=list(set(data['song_labels'])& all_candidates_set)
        if labels_type==0:
            true_labels=data['song_pseudo_golden_labels']
        elif labels_type==1:
            true_labels=expert_labels
        elif labels_type==2:
            true_labels=list(set(annotated_labels+data['song_pseudo_golden_labels']))
        else:
            true_labels=list(set(annotated_labels+expert_labels))
        # 将 golden \ expert \ annotated+golden 标签加入saong_info_dict
        # golden_labels=data['song_pseudo_golden_labels']
        expert_labels=list(set(data['song_labels'])& all_candidates_set)
        song_candidates=list(data['final_score_dict_sort'].keys())
        print('----------------------')
        print(len(expert_labels),len(set(expert_labels)&set(song_candidates)))
        # annotated_golden_labels=list(set(annotated_labels+data['song_pseudo_golden_labels']))
        # song_info_dict[song_name]['golden_labels']=golden_labels
        # song_info_dict[song_name]['expert_labels']=expert_labels
        # song_info_dict[song_name]['annotated_golden_labels']=annotated_golden_labels
        # song_info_dict[song_name]['annotated_expert_labels']=list(set(annotated_labels+expert_labels))
        
        candiadates_dict=data['final_score_dict_sort']
        for candiadate in candiadates_dict:
            tmp_soft_s=float(round(lexical_soft_match(candiadate, true_labels),5))
            lexical_soft_match_score_dict[song_name+'|'+candiadate]=tmp_soft_s
    # save
    # with open(saved_folder_path+file_name,'w') as f:
    #     json.dump(lexical_soft_match_score_dict,f)
    # save song_info_dict 
    # with open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/test_song_infos.json','w') as f:
    #     json.dump(song_info_dict,f)
    print('lexical soft match score saved')
        
        
        
