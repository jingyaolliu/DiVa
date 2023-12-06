import os, json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import get_song_id_given_song_name
from arguments import args
from utils import ROOT_PATH, compute_negative_sample, convert_list_to_dict,get_recent_time,get_negative_sample
from os.path import join
import math
update_method=args.update_method

candidate_words_embedding_root_path='/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
setup_seed(100)

class CustomDatasetBinary(Dataset):
    def __init__(self,song_id_layer_outputs, label_class,file_version,start_iter,discard_positive,light,data_type="train", process_type="update_match_score"):
        """
        label_class : 'golden' or 'expert'
        file_version : the extension explanation of file version -  basically for data loaded and saved
        """
        self.data_type = data_type
        self.label_class=label_class
        self.iter=-1
        self.process_type = process_type
        self.raw_data_dict = {}
        self.song_id_silver=[]
        self.file_version=file_version
        self.light=light
        self.start_iter=start_iter
        self.discard_positive=discard_positive
        # self.train_song_id_candidate_words_shuffle_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_id_candidate_words_shuffle_ebc_1116.npy"
        # 无用
        self.train_song_id_candidate_words_shuffle_path ='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/train_ebc_shuffle_candidates_1123.npy'
        # self.train_song_candidate_words_embedding_shuffle_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/train_song_candidate_words_embedding_shuffle_1108.npy'
        # self.val_song_candidate_words_embedding_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/val_song_candidate_words_embedding_1108.npy'
        # self.val_song_id_candidate_words_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_id_candidate_words_shuffle_ebc_1116.npy"
        # 无用
        self.val_song_id_candidate_words_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/test_ebc_shuffle_candidates_1123.txt'
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_fasttext_embedding_dict.npy"), allow_pickle=True)
        # obtaining better static word embeddings using contextual embedding models
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_CEM_embedding_dict.npy"), allow_pickle=True)
        self.song_candidate_words_dict = np.load('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/candidate_words_context_embedding_dict_1536_300.npy', allow_pickle=True)
        self.availabel_candidates_set=set(self.song_candidate_words_dict.item().keys())
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)
        #统计每个词的词频
        self.word_frequency_dict = json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/words_tf_info_0322.json','r',encoding='utf-8'))

        if self.data_type == "train":
            # self.song_ids = np.load(self.train_song_id_candidate_words_shuffle_path)
            self.candidate_words_layer_outputs = []#np.load(self.train_song_candidate_words_embedding_shuffle_path)
            # self.song_ids = open(
            #     "/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_id_candidate_words.txt").readlines()
            # self.candidate_words_layer_outputs = np.load(
            #     '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_candidate_words_embedding.npy')
        else:
            self.candidate_words_layer_outputs = []#np.load(self.val_song_candidate_words_embedding_path)
            # if '.npy' in self.val_song_id_candidate_words_path:
            #     self.song_ids = np.load(self.val_song_id_candidate_words_path)
            # elif '.txt' in self.val_song_id_candidate_words_path:
            #     self.song_ids = open(self.val_song_id_candidate_words_path, "r").readlines()
        #self.data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file)))
        #self.val_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_val)))
        #self.test_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_test)))
        self.song_id_layer_outputs = song_id_layer_outputs
        self.golden_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/song_id_candidate_words_embedding/{}_{}_song_id_golden.npy'.format(update_method,self.data_type)
        self.negative_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/song_id_candidate_words_embedding/{}_{}_song_id_negative_iter_{}.npy'.format(update_method,
            self.data_type,self.iter)
        self.silver_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/song_id_candidate_words_embedding/{}_{}_song_id_silver_iter_{}.npy'.format(update_method,
            self.data_type,self.iter)
        # self.process_golden_data()
        # self.process_negative_data()
        # self.process_silver_data()
        # self.load_golden_data()
        # self.load_negative_data()
        # self.load_silver_data()
        # self.train_data = random.shuffle(self.song_id_golden+self.song_id_silver+self.song_id_negative)
        # self.data = self.convert_words_to_idx(self.data_raw, vocab=vocab)
        #self.val_data = self.convert_words_to_idx(self.val_data_raw, vocab=vocab)
        #self.test_data = self.convert_words_to_idx(self.test_data_raw, vocab=vocab)
        # self.train_t0_path = os.path.join(ROOT_PATH, "Preprocess", "train_t0_for_supervise_with_all_baselines_1108_resample.json")
        # self.test_t0_path = os.path.join(ROOT_PATH, "Preprocess", "annotation_t0_for_supervise_with_all_baselines_1108_resample.json")
        #todo iter1 
        # self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_t0_for_supervise_DIVA_1123.json'
        # self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_t0_for_supervise_DIVA_1123.json'
        #todo iter2 
        self.discard_num=0
        # if self.start_iter>0 and file_version!='none':
        #     if label_class=='golden':
        #         self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter{}_{}.json'.format(self.start_iter,self.file_version)
        #         self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter{}_{}.json'.format(self.start_iter,self.file_version)
        #     elif label_class=='expert':
        #         self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter{}_{}_expert.json'.format(self.start_iter,self.file_version)
        #         self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter{}_{}_expert.json'.format(self.start_iter,self.file_version)
        #     print('train file losd from {}'.format(sself.train_t0_path))
        #     print('test file losd from {}'.format(self.test_t0_path))
            
        # else:
        if label_class=='golden':
            self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter_based.json'
            self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter_based.json'
        elif label_class=='expert':
            self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_iter_based_expert.json'
            self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_iter_based_expert.json'
        print('train file losd from {}'.format(self.train_t0_path))
        print('test file losd from {}'.format(self.test_t0_path))
        #{candidate_tuple:[song_id,candidate],sample_rate}
        self.negative_sample_info={}


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        #todo old
        # song_idx, sample_type = self.train_data[idx]
        # song_id, one_candidate_word, is_unk = self.song_ids[song_idx].strip().split("|")
        # todo old
        #todo new
        song_id, one_candidate_word, sample_type = self.train_data[idx].split("|")
        # song_comments_vector = torch.cat([torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_1']), torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_last'])], dim=0)
        song_comments_vector = torch.FloatTensor(self.song_id_layer_outputs[song_id])
        # song_candidate_word_vector = torch.FloatTensor(self.candidate_words_layer_outputs[idx])
        song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        # get fasttext embedding dim from 300 to 300* 1500
        # song_candidate_word_vector = torch.FloatTensor(np.tile(self.song_candidate_words_dict.item()[one_candidate_word],5))
        return song_comments_vector, song_candidate_word_vector, sample_type, song_id

    def process_t0_data(self, re_sample=True, negative_sample_method='random'):
        negative_sample_method=args.negative_sample_method
        if self.data_type == "train":
            # todo here is debug
            data_segmented_list = json.load(open(self.train_t0_path, "r", encoding='utf-8'))
            if re_sample:
                # negative_candidates,negative_sample_ration=get_negative_sample(data_segmented_list)
                # self.negative_sample_info['negative_candidates']=negative_candidates
                # self.negative_sample_info['negative_sample_ration']=negative_sample_ration
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        elif self.data_type == "val":
            data_segmented_list = json.load(open(self.test_t0_path, "r", encoding='utf-8'))
            if re_sample:
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
                # negative_candidates, negative_sample_ration = get_negative_sample(data_segmented_list)
                # self.negative_sample_info['negative_candidates'] = negative_candidates
                # self.negative_sample_info['negative_sample_ration'] = negative_sample_ration
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        else:
            raise ValueError
        #todo 
        # song_ids_new = []
        # for i in range(len(self.song_ids)):
        #     song_id, one_candidate_word, is_unk = self.song_ids[i].strip().split("|")
        #     if song_id in self.raw_data_dict:
        #         song_ids_new.append("|".join([song_id, one_candidate_word, is_unk]))
        # self.song_ids = song_ids_new
    def reprocess_t0_data(self, data_lst,re_sample=True, negative_sample_method='random'):
        negative_sample_method=args.negative_sample_method
        if self.data_type == "train":
            # todo here is debug
            data_segmented_list = data_lst
            if re_sample:
                # negative_candidates,negative_sample_ration=get_negative_sample(data_segmented_list)
                # self.negative_sample_info['negative_candidates']=negative_candidates
                # self.negative_sample_info['negative_sample_ration']=negative_sample_ration
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        elif self.data_type == "val":
            data_segmented_list = data_lst
            if re_sample:
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
                # negative_candidates, negative_sample_ration = get_negative_sample(data_segmented_list)
                # self.negative_sample_info['negative_candidates'] = negative_candidates
                # self.negative_sample_info['negative_sample_ration'] = negative_sample_ration
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        else:
            raise ValueError

    # 只有golden， silver，negative samples才会参与到训练中去
    def process_golden_data(self):
        song_id_golden = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    song_id_golden.append(idx)
        # np.save(self.golden_path, song_id_golden, allow_pickle=False)
        np.savetxt(self.golden_path, song_id_golden,  delimiter=",")

    #todo 加载golden数据
    # 按照概率随机筛出一些频率高的golden
    # 获取被删除的概率
    def get_del_prob(self,word):
        if word not in self.word_frequency_dict:
            return 0
        word_freq=self.word_frequency_dict[word]['word_freq']
        discard_rate=1-math.sqrt(1e-4/word_freq)
        discard_rate=max(discard_rate,0)
        return discard_rate
    
    def process_golden_data_only(self):
        golden_data=[]
        for song_id in self.raw_data_dict:
            if self.label_class=='golden':
                for one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    # if self.discard_positive:
                    #     discard_rate=self.get_del_prob(one_candidate_word)
                    # else:
                    #     discard_rate=0
                    # # 获取0-1间的随机数字
                    # random_num=random.random()
                    # # random_num=2
                    # if random_num>discard_rate:
                    golden_data.append("|".join([song_id, one_candidate_word, "golden"]))
                    # else:
                    #     self.discard_num+=1
            elif self.label_class=='expert':
                for one_candidate_word in self.raw_data_dict[song_id]['song_labels']:
                    if one_candidate_word in self.availabel_candidates_set:
                        if self.discard_positive:
                            discard_rate=self.get_del_prob(one_candidate_word)
                        else:
                            discard_rate=0
                        # 获取0-1间的随机数字
                        random_num=random.random()
                        # random_num=2
                        if random_num>discard_rate:
                            golden_data.append("|".join([song_id, one_candidate_word, "golden"]))
                        else:
                            self.discard_num+=1
        self.golden_data=golden_data

    def load_golden_data(self):
        song_id_golden = np.loadtxt(self.golden_path, delimiter=",", dtype=int)
        self.song_id_golden = [(one_line, "golden") for one_line in song_id_golden]

    def process_negative_data(self):
        song_id_negative = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word in self.raw_data_dict[song_id]['negative_samples']['label']:
                    song_id_negative.append(idx)
        np.savetxt(self.negative_path,song_id_negative, delimiter=",")
    
    #todo 加载negative数据
    def process_negative_data_only(self):
        negative_data=[]
        for song_id in self.raw_data_dict:
            for one_candidate_word in self.raw_data_dict[song_id]['negative_samples']['label']:
                negative_data.append("|".join([song_id, one_candidate_word, "negative"]))
        self.negative_data=negative_data
        
    def process_negative_data_by_ration(self):
        negative_data=[]
        # 根据negative sample ration 筛选negative num 个negative sample
        negative_num=int(len(self.golden_data)+len(self.silver_data))
        negative_data=random.sample(self.negative_sample_info['negative_candidates'],self.negative_sample_info['negative_sample_ration'],negative_num)
        self.negative_data=negative_data

    def load_negative_data(self):
        song_id_negative = np.loadtxt(self.negative_path,delimiter=",",dtype=int)
        self.song_id_negative = [(one_line, "negative") for one_line in song_id_negative]

    def process_silver_data(self,iter=-1):
        self.iter=iter
        song_id_silver = []
        if self.iter != -1:
            print("iter:{}".format(self.iter))
        print("len(self.song_ids):{}".format(len(self.song_ids)))
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            # if self.iter!=-1:
            #     if idx%10000==0:
            #         print('{}/{}'.format(idx,len(self.song_ids)))
            if song_id in self.raw_data_dict:
                if 'song_silver_label_details' in self.raw_data_dict[song_id]:
                    if one_candidate_word in self.raw_data_dict[song_id]['song_silver_label_details'][str(self.iter)]:
                        song_id_silver.append(idx)
                elif 'song_siver_labels' in self.raw_data_dict[song_id]:
                    if one_candidate_word in self.raw_data_dict[song_id]['song_siver_labels']:
                        song_id_silver.append(idx)
        # 获取当前时间 填充到文件名中
        now_time=get_recent_time()
        self.recent_silver_path=self.silver_path.replace(".npy","_{}.npy".format(now_time))
        np.savetxt(self.silver_path, song_id_silver, delimiter=",")
    
    #todo 加载silver数据
    # silver label中包含 ebc_based >=0.95的，这部分数据也包含很多高频错词，需要筛出
    def process_silver_data_only(self):
        silver_data=[]
        for song_id in self.raw_data_dict:
            golden_labels=self.raw_data_dict[song_id]['song_pseudo_golden_labels']
            if self.light==False and 'song_all_silver_labels' in self.raw_data_dict[song_id]:
                for one_candidate_word in self.raw_data_dict[song_id]['song_all_silver_labels']:
                    if one_candidate_word in golden_labels:
                        continue
                    if self.discard_positive:
                        discard_rate=self.get_del_prob(one_candidate_word)
                    else:
                        discard_rate=0
                    # 获取0-1间的随机数字
                    random_num=random.random()
                    # random_num=2
                    if random_num>discard_rate:
                        silver_data.append("|".join([song_id, one_candidate_word, "silver"]))
                    else:
                        self.discard_num+=1
            elif self.light==True and 'song_silver_labels' in self.raw_data_dict[song_id]:
                for one_candidate_word in self.raw_data_dict[song_id]['song_silver_labels'][-1]:
                    if one_candidate_word in golden_labels:
                        continue
                    if self.discard_positive:
                        discard_rate=self.get_del_prob(one_candidate_word)
                    else:
                        discard_rate=0
                    # 获取0-1间的随机数字
                    random_num=random.random()
                    # random_num=2
                    if random_num>discard_rate:
                        silver_data.append("|".join([song_id, one_candidate_word, "silver"]))
                    else:
                        self.discard_num+=1
        self.silver_data=silver_data
        print('silver_len:{}'.format(len(self.silver_data)))

    def load_silver_data(self):
        song_id_silver = np.loadtxt(self.silver_path, delimiter=",", dtype=int)
        self.song_id_silver = [(one_line,"silver") for one_line in song_id_silver]

    def shuffle_train_data(self):
        self.train_data = list(set(self.song_id_golden + self.song_id_silver) )+ self.song_id_negative
        random.shuffle(self.train_data)
    
    # 初始化数据加载器
    def initial_data_loader(self):
        if self.light==False:
            self.process_golden_data_only()
        else:
            self.golden_data=[]
        self.process_negative_data_only()
        self.process_silver_data_only()
        # 保证negative 数量和golden+silever数量一致，若多了则随机筛除
        print("len(self.golden_silver_data):{}".format(len(self.golden_data+self.silver_data)))
        print('discard_num:{}'.format(self.discard_num))
        if len(self.negative_data)>len(self.golden_data)+len(self.silver_data):
            self.negative_data=random.sample(self.negative_data,len(self.golden_data)+len(self.silver_data))
        self.train_data=self.golden_data+self.silver_data+self.negative_data
        random.shuffle(self.train_data)

    def initial_data_loader_process_and_save(self):
        self.process_golden_data()
        self.process_negative_data()
        self.process_silver_data()

    def initial_data_loader_load(self):
        self.load_golden_data()
        self.load_negative_data()
        self.load_silver_data()
        self.shuffle_train_data()
    
    def reset_raw_data(self,dict_data):
        self.raw_data_dict = dict_data


class CustomDatasetBinaryUpdateMatchScore(Dataset):
    def __init__(self, file_content, song_id_layer_outputs, candidate_words_layer_outputs, song_candidate_words_dict,train_data):
        self.raw_data_dict = file_content
        # self.song_ids = song_ids
        self.candidate_words_layer_outputs = candidate_words_layer_outputs
        self.song_id_layer_outputs = song_id_layer_outputs
        self.song_candidate_words_dict = song_candidate_words_dict
        self.train_data=train_data
        # self.get_all_infos()

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        song_id, one_candidate_word, sample_type = self.train_data[idx].split("|")
        # song_id, one_candidate_word, _ = self.song_ids[song_idx].strip().split("|")
        # song_comments_vector = torch.cat([torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_1']), torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_last'])], dim=0)
        song_comments_vector = torch.FloatTensor(self.song_id_layer_outputs[song_id])
        # song_candidate_word_vector = torch.FloatTensor(self.candidate_words_layer_outputs[idx])
        # song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        # song_candidate_word_vector = torch.FloatTensor(np.tile(self.song_candidate_words_dict.item()[one_candidate_word],5))
        song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        return song_comments_vector, one_candidate_word, song_candidate_word_vector, song_id

    def get_all_infos(self):
        all_datas=[]
        for song_name in self.raw_data_dict:
            score_dict=self.raw_data_dict[song_name]['final_score_dict_sort']
            for candidate in score_dict:
                all_datas.append('|'.join([song_name,candidate,'none']))
        self.train_data=all_datas


class PUDataLoader(Dataset):
    def __init__(self, song_id_layer_outputs, label_class,data_type="train", process_type="update_match_score"):
        """
        label class = expert / golden
        """
        self.data_type = data_type
        self.label_class= label_class
        self.iter=-1
        self.process_type = process_type
        self.raw_data_dict = {}
        self.song_id_silver=[]
        # self.train_song_id_candidate_words_shuffle_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_id_candidate_words_shuffle_ebc_1116.npy"
        self.train_song_id_candidate_words_shuffle_path ='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/train_ebc_shuffle_candidates_1123.npy'
        # self.train_song_candidate_words_embedding_shuffle_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/train_song_candidate_words_embedding_shuffle_1108.npy'
        # self.val_song_candidate_words_embedding_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/val_song_candidate_words_embedding_1108.npy'
        # self.val_song_id_candidate_words_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_id_candidate_words_shuffle_ebc_1116.npy"
        self.val_song_id_candidate_words_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/test_ebc_shuffle_candidates_1123.txt'
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_fasttext_embedding_dict.npy"), allow_pickle=True)
        # obtaining better static word embeddings using contextual embedding models
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_CEM_embedding_dict.npy"), allow_pickle=True)
        self.song_candidate_words_dict = np.load('/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/candidate_words_context_embedding_dict_1536_300.npy', allow_pickle=True)
        self.availabel_candidates_set = list(self.song_candidate_words_dict.item().keys())
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)

        if self.data_type == "train":
            self.song_ids = np.load(self.train_song_id_candidate_words_shuffle_path)
            self.candidate_words_layer_outputs = []#np.load(self.train_song_candidate_words_embedding_shuffle_path)
            # self.song_ids = open(
            #     "/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_id_candidate_words.txt").readlines()
            # self.candidate_words_layer_outputs = np.load(
            #     '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_candidate_words_embedding.npy')
        else:
            self.candidate_words_layer_outputs = []#np.load(self.val_song_candidate_words_embedding_path)
            if '.npy' in self.val_song_id_candidate_words_path:
                self.song_ids = np.load(self.val_song_id_candidate_words_path)
            elif '.txt' in self.val_song_id_candidate_words_path:
                self.song_ids = open(self.val_song_id_candidate_words_path, "r").readlines()
        #self.data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file)))
        #self.val_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_val)))
        #self.test_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_test)))
        self.song_id_layer_outputs = song_id_layer_outputs
        # self.process_golden_data()
        # self.process_negative_data()
        # self.process_silver_data()
        # self.load_golden_data()
        # self.load_negative_data()
        # self.load_silver_data()
        # self.train_data = random.shuffle(self.song_id_golden+self.song_id_silver+self.song_id_negative)
        # self.data = self.convert_words_to_idx(self.data_raw, vocab=vocab)
        #self.val_data = self.convert_words_to_idx(self.val_data_raw, vocab=vocab)
        #self.test_data = self.convert_words_to_idx(self.test_data_raw, vocab=vocab)
        # self.train_t0_path = os.path.join(ROOT_PATH, "Preprocess", "train_t0_for_supervise_with_all_baselines_1108_resample.json")
        # self.test_t0_path = os.path.join(ROOT_PATH, "Preprocess", "annotation_t0_for_supervise_with_all_baselines_1108_resample.json")
        self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_t0_for_supervise_DIVA_1123.json'
        self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_t0_for_supervise_DIVA_1123.json'


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # 改为 song_id, candidate_word, sample_type
        # song_idx, sample_type = self.train_data[idx]
        song_id, one_candidate_word, sample_type = self.train_data[idx].strip().split("|")
        # song_comments_vector = torch.cat([torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_1']), torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_last'])], dim=0)
        song_comments_vector = torch.FloatTensor(self.song_id_layer_outputs[song_id])
        # song_candidate_word_vector = torch.FloatTensor(self.candidate_words_layer_outputs[idx])
        song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        # get fasttext embedding dim from 300 to 300* 1500
        # song_candidate_word_vector = torch.FloatTensor(np.tile(self.song_candidate_words_dict.item()[one_candidate_word],5))
        return song_comments_vector, song_candidate_word_vector, sample_type, song_id

    def process_t0_data(self, re_sample=True, negative_sample_method='random'):
        negative_sample_method=args.negative_sample_method
        if self.data_type == "train":
            # todo here debug
            data_segmented_list = json.load(open(self.train_t0_path, "r", encoding='utf-8'))
            
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        elif self.data_type == "val":
            data_segmented_list = json.load(open(self.test_t0_path, "r", encoding='utf-8'))

            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        else:
            raise ValueError
        #todo 
        # song_ids_new = []
        # for i in range(len(self.song_ids)):
        #     song_id, one_candidate_word, is_unk = self.song_ids[i].strip().split("|")
        #     if song_id in self.raw_data_dict:
        #         song_ids_new.append("|".join([song_id, one_candidate_word, is_unk]))
        # self.song_ids = song_ids_new

    # 只有golden， silver，negative samples才会参与到训练中去
    # def process_golden_data(self):
    #     song_id_golden = []
    #     for idx in range(len(self.song_ids)):
    #         song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
    #         if song_id in self.raw_data_dict:
    #             if self.label_class=='golden':
    #                 if one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
    #                     song_id_golden.append(idx)
    #     # np.save(self.golden_path, song_id_golden, allow_pickle=False)
    #     self.song_id_golden = [(item,'golden') for item in song_id_golden]
    
    def process_golden_data(self):
        labeled_data=[]
        for song_id in self.raw_data_dict:
            if self.label_class=='golden':
                for one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    labeled_data.append('|'.join([song_id,one_candidate_word,'golden']))
            elif self.label_class=='expert':
                for one_candidate_word in self.raw_data_dict[song_id]['song_labels']:
                    if one_candidate_word in self.availabel_candidates_set:
                        labeled_data.append('|'.join([song_id,one_candidate_word,'golden']))
        self.song_id_golden = labeled_data
    
    # def process_unlabel_data(self):
    #     song_id_unlabel = []
    #     for idx in range(len(self.song_ids)):
    #         song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
    #         if song_id in self.raw_data_dict:
    #             if one_candidate_word not in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
    #                 song_id_unlabel.append(idx)
    #     self.song_id_unlabel = [(item,'unlabel') for item in song_id_unlabel]
        
    def process_unlabel_data(self):
        unlabeled_data = []
        for song_id in self.raw_data_dict:
            if self.label_class=='golden':
                song_labels=self.raw_data_dict[song_id]['song_pseudo_golden_labels']
            else:
                song_labels=self.raw_data_dict[song_id]['song_labels']
                song_labels=list(set(song_labels).intersection(self.availabel_candidates_set))
            candidates=self.raw_data_dict[song_id]['final_score_dict_sort']
            for candidate_word in candidates:
                if candidate_word not in song_labels:
                    unlabeled_data.append('|'.join([song_id,candidate_word,'unlabel']))
        self.song_id_unlabel = unlabeled_data
            


    def shuffle_train_data(self):
        # self.train_data = self.song_id_golden + self.song_id_silver + self.song_id_negative
        # set song_id_unlabel same size as song_id_golden
        # random.shuffle(self.song_id_unlabel)
        # self.song_id_unlabel = self.song_id_unlabel[:len(self.song_id_golden)]
        self.train_data=self.song_id_golden+self.song_id_unlabel
        random.shuffle(self.train_data)

    def initial_data_loader_load(self):
        self.process_golden_data()
        self.process_unlabel_data()
        self.shuffle_train_data()

    def reset_raw_data(self,dict_data):
        self.raw_data_dict = dict_data

class NSDataLoader(Dataset):
    def __init__(self, song_id_layer_outputs, data_type="train", process_type="update_match_score",alpha=0.5):
        '''
        pram :
        -- alpha : for label popularity calculate
        '''
        self.alpha=alpha
        self.data_type = data_type
        self.iter=-1
        self.process_type = process_type
        self.raw_data_dict = {}
        self.song_id_silver=[]
        # unlabel data 所占比例
        self.unlabel_partition=0
        self.hard_negative_partition=0
        # label_popularity {l:popularity}
        self.label_popularity_dict={}
        # self.train_song_id_candidate_words_shuffle_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_id_candidate_words_shuffle_ebc_1116.npy"
        self.train_song_id_candidate_words_shuffle_path ='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/train_ebc_shuffle_candidates_1123.npy'
        # self.train_song_candidate_words_embedding_shuffle_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/train_song_candidate_words_embedding_shuffle_1108.npy'
        # self.val_song_candidate_words_embedding_path = '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding/val_song_candidate_words_embedding_1108.npy'
        # self.val_song_id_candidate_words_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_id_candidate_words_shuffle_ebc_1116.npy"
        self.val_song_id_candidate_words_path = '/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/test_ebc_shuffle_candidates_1123.txt'
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_fasttext_embedding_dict.npy"), allow_pickle=True)
        # obtaining better static word embeddings using contextual embedding models
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_CEM_embedding_dict.npy"), allow_pickle=True)
        self.song_candidate_words_dict = np.load('/data/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/candidate_words_context_embedding_dict_1536_300.npy', allow_pickle=True)
        # self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_random_embedding_dict.npy"), allow_pickle=True)

        if self.data_type == "train":
            self.song_ids = np.load(self.train_song_id_candidate_words_shuffle_path)
            self.candidate_words_layer_outputs = []#np.load(self.train_song_candidate_words_embedding_shuffle_path)
            # self.song_ids = open(
            #     "/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_id_candidate_words.txt").readlines()
            # self.candidate_words_layer_outputs = np.load(
            #     '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_candidate_words_embedding.npy')
        else:
            self.candidate_words_layer_outputs = []#np.load(self.val_song_candidate_words_embedding_path)
            if '.npy' in self.val_song_id_candidate_words_path:
                self.song_ids = np.load(self.val_song_id_candidate_words_path)
            elif '.txt' in self.val_song_id_candidate_words_path:
                self.song_ids = open(self.val_song_id_candidate_words_path, "r").readlines()
        #self.data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file)))
        #self.val_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_val)))
        #self.test_data_raw = self.extract_pseudo_golden_labels(json.load(open(annotations_file_test)))
        self.song_id_layer_outputs = song_id_layer_outputs
        # self.process_golden_data()
        # self.process_negative_data()
        # self.process_silver_data()
        # self.load_golden_data()
        # self.load_negative_data()
        # self.load_silver_data()
        # self.train_data = random.shuffle(self.song_id_golden+self.song_id_silver+self.song_id_negative)
        # self.data = self.convert_words_to_idx(self.data_raw, vocab=vocab)
        #self.val_data = self.convert_words_to_idx(self.val_data_raw, vocab=vocab)
        #self.test_data = self.convert_words_to_idx(self.test_data_raw, vocab=vocab)
        # self.train_t0_path = os.path.join(ROOT_PATH, "Preprocess", "train_t0_for_supervise_with_all_baselines_1108_resample.json")
        # self.test_t0_path = os.path.join(ROOT_PATH, "Preprocess", "annotation_t0_for_supervise_with_all_baselines_1108_resample.json")
        self.train_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/train_t0_for_supervise_DIVA_1123.json'
        self.test_t0_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/diva_data/test_t0_for_supervise_DIVA_1123.json'


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        song_idx,tmp_label,sample_type = self.train_data[idx]
        if tmp_label in self.label_popularity_dict:
            label_popularity = self.label_popularity_dict[tmp_label]
        else:
            label_popularity=0
        song_id, one_candidate_word, is_unk = self.song_ids[song_idx].strip().split("|")
        # song_comments_vector = torch.cat([torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_1']), torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_last'])], dim=0)
        song_comments_vector = torch.FloatTensor(self.song_id_layer_outputs[song_id])
        # song_candidate_word_vector = torch.FloatTensor(self.candidate_words_layer_outputs[idx])
        song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        # get fasttext embedding dim from 300 to 300* 1500
        # song_candidate_word_vector = torch.FloatTensor(np.tile(self.song_candidate_words_dict.item()[one_candidate_word],5))
        return song_comments_vector, song_candidate_word_vector, sample_type, song_id,label_popularity

    def process_t0_data(self, re_sample=True, negative_sample_method='random'):
        negative_sample_method=args.negative_sample_method
        if self.data_type == "train":
            data_segmented_list = json.load(open(self.train_t0_path, "r", encoding='utf-8'))
            
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        elif self.data_type == "val":
            data_segmented_list = json.load(open(self.test_t0_path, "r", encoding='utf-8'))

            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        else:
            raise ValueError
        #todo 
        # song_ids_new = []
        # for i in range(len(self.song_ids)):
        #     song_id, one_candidate_word, is_unk = self.song_ids[i].strip().split("|")
        #     if song_id in self.raw_data_dict:
        #         song_ids_new.append("|".join([song_id, one_candidate_word, is_unk]))
        # self.song_ids = song_ids_new

    # 只有golden， silver，negative samples才会参与到训练中去
    def process_golden_data(self):
        song_id_golden = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    song_id_golden.append((idx,one_candidate_word))
                    if one_candidate_word not in self.label_popularity_dict:
                        self.label_popularity_dict[one_candidate_word] = 1
                    else:
                        self.label_popularity_dict[one_candidate_word] += 1
                        
        # np.save(self.golden_path, song_id_golden, allow_pickle=False)
        # (idx,golden_label,golden_type)
        self.song_id_golden = [(item[0],item[1],'golden') for item in song_id_golden]
    
    def process_unlabel_data(self):
        song_id_unlabel = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word not in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    song_id_unlabel.append((idx,one_candidate_word))
        self.song_id_unlabel = [(item[0],item[1],'unlabel') for item in song_id_unlabel]


    def shuffle_train_data(self):
        # self.train_data = self.song_id_golden + self.song_id_silver + self.song_id_negative
        # set song_id_unlabel same size as song_id_golden
        # random.shuffle(self.song_id_unlabel)
        # self.song_id_unlabel = self.song_id_unlabel[:len(self.song_id_golden)]
        self.train_data=self.song_id_golden+self.song_id_unlabel
        random.shuffle(self.train_data)

    def initial_data_loader_load(self):
        self.process_golden_data()
        self.process_unlabel_data()
        self.shuffle_train_data()
        self.unlabel_partition=len(self.song_id_unlabel)/(len(self.song_id_golden)+len(self.song_id_unlabel))
        # self.hard_negative_partition=len(self.label_popularity_dict)/(len(self.song_id_unlabel))
        # calculate label priority：
        freq_val_np=np.array(list(self.label_popularity_dict.values()))
        exp_sum=np.power(freq_val_np,self.alpha).sum()
        
        popular_label_size=len(self.label_popularity_dict)
        for key in self.label_popularity_dict:
            self.label_popularity_dict[key]=(np.power(self.label_popularity_dict[key],self.alpha)/exp_sum)*self.unlabel_partition 

    def reset_raw_data(self,dict_data):
        self.raw_data_dict = dict_data
class CustomDatasetBinaryLSTM(Dataset):
    def __init__(self, data_type="train", process_type="update_match_score"):
        self.data_type = data_type
        self.process_type = process_type
        self.raw_data_dict = {}
        self.train_song_id_candidate_words_shuffle_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/train_song_id_candidate_words_shuffle_1108.npy"
        self.val_song_id_candidate_words_path = "/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/val_song_id_candidate_words_1108.txt"
        self.song_candidate_words_dict = np.load(join(candidate_words_embedding_root_path, "candidate_words_fasttext_embedding_dict.npy"), allow_pickle=True)
        self.iter=0
        if self.data_type == "train":
            self.song_ids = np.load(self.train_song_id_candidate_words_shuffle_path)
            self.candidate_words_layer_outputs = []#np.load(self.train_song_candidate_words_embedding_shuffle_path)
            # self.song_ids = open(
            #     "/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_id_candidate_words.txt").readlines()
            # self.candidate_words_layer_outputs = np.load(
            #     '/home/yuanxin/tme_big_data/song_id_candidate_words_embedding_old/train_song_candidate_words_embedding.npy')
        else:
            self.candidate_words_layer_outputs = []#np.load(self.val_song_candidate_words_embedding_path)
            self.song_ids = open(self.val_song_id_candidate_words_path, "r").readlines()
        self.golden_path = '/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/{}_song_id_golden.npy'.format(self.data_type)
        self.negative_path = '/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/{}_song_id_negative_iter_{}.npy'.format(
            self.data_type,self.iter)
        self.silver_path = '/home/yuanxin/yuanxin_data/tme_big_data/song_id_candidate_words_embedding/{}_song_id_silver_iter_{}.npy'.format(
            self.data_type,self.iter)

        self.train_t0_path = os.path.join(ROOT_PATH, "Preprocess", "train_t0_for_supervise_with_all_baselines_1108_resample.json")
        self.test_t0_path = os.path.join(ROOT_PATH, "Preprocess", "annotation_t0_for_supervise_with_all_baselines_1108_resample.json")


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        song_idx, sample_type = self.train_data[idx]
        song_id, one_candidate_word, is_unk = self.song_ids[song_idx].strip().split("|")
        # song_comments_vector = torch.cat([torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_1']), torch.FloatTensor(self.song_id_layer_outputs[song_id]['layer_last'])], dim=0)
        song_comments_vector = torch.FloatTensor(self.song_id_layer_outputs[song_id])
        # song_candidate_word_vector = torch.FloatTensor(self.candidate_words_layer_outputs[idx])
        song_candidate_word_vector = torch.FloatTensor(self.song_candidate_words_dict.item()[one_candidate_word])
        return song_comments_vector, song_candidate_word_vector, sample_type, song_id

    def process_t0_data(self, re_sample=True, negative_sample_method='random'):
        if self.data_type == "train":
            data_segmented_list = json.load(open(self.train_t0_path, "r", encoding='utf-8'))
            if re_sample:
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        elif self.data_type == "val":
            data_segmented_list = json.load(open(self.test_t0_path, "r", encoding='utf-8'))
            if re_sample:
                data_segmented_list = compute_negative_sample(data_segmented_list,
                                                              reset_negative_sample=True,
                                                              sample_type=negative_sample_method)
            self.raw_data_dict = convert_list_to_dict(data_segmented_list)
        else:
            raise ValueError
        song_ids_new = []
        for i in range(len(self.song_ids)):
            song_id, one_candidate_word, is_unk = self.song_ids[i].strip().split("|")
            if song_id in self.raw_data_dict:
                song_ids_new.append("|".join([song_id, one_candidate_word, is_unk]))
        self.song_ids = song_ids_new

    # 只有golden， silver，negative samples才会参与到训练中去
    def process_golden_data(self):
        song_id_golden = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word in self.raw_data_dict[song_id]['song_pseudo_golden_labels']:
                    song_id_golden.append(idx)
        # np.save(self.golden_path, song_id_golden, allow_pickle=False)
        np.savetxt(self.golden_path, song_id_golden,  delimiter=",")

    def load_golden_data(self):
        song_id_golden = np.loadtxt(self.golden_path, delimiter=",", dtype=int)
        self.song_id_golden = [(one_line, "golden") for one_line in song_id_golden]

    def process_negative_data(self):
        song_id_negative = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if one_candidate_word in self.raw_data_dict[song_id]['negative_samples']['label']:
                    song_id_negative.append(idx)
        np.savetxt(self.negative_path,song_id_negative, delimiter=",")

    def load_negative_data(self):
        song_id_negative = np.loadtxt(self.negative_path,delimiter=",",dtype=int)
        self.song_id_negative = [(one_line, "negative") for one_line in song_id_negative]

    def process_silver_data(self,iter=0):
        self.iter = iter
        song_id_silver = []
        for idx in range(len(self.song_ids)):
            song_id, one_candidate_word, _ = self.song_ids[idx].strip().split("|")
            if song_id in self.raw_data_dict:
                if 'song_silver_label_details' in self.raw_data_dict[song_id]:
                    if one_candidate_word in self.raw_data_dict[song_id]['song_silver_label_details'][str(self.iter)]:
                        song_id_silver.append(idx)
                elif 'song_silver_labels' in self.raw_data_dict[song_id]:
                    if one_candidate_word in self.raw_data_dict[song_id]['song_silver_labels']:
                        song_id_silver.append(idx)
        np.savetxt(self.silver_path, song_id_silver, delimiter=",")

    def load_silver_data(self):
        song_id_silver = np.loadtxt(self.silver_path, delimiter=",", dtype=int)
        self.song_id_silver += [(one_line,"silver") for one_line in song_id_silver]

    def shuffle_train_data(self):
        self.train_data = self.song_id_golden + self.song_id_silver + self.song_id_negative
        random.shuffle(self.train_data)

    def initial_data_loader_process_and_save(self):
        self.process_golden_data()
        self.process_negative_data()
        self.process_silver_data()

    def initial_data_loader_load(self):
        self.load_golden_data()
        self.load_negative_data()
        self.load_silver_data()
        self.shuffle_train_data()

