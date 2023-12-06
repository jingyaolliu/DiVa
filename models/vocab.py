import os.path
import torch, json, os
#from gensim.test.utils import datapath
#from gensim.models import KeyedVectors
#from gensim.models.wrappers import FastText
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import numpy as np
from gensim import utils
from utils import ROOT_PATH, CODE_PATH

torch.manual_seed(100)
np.random.seed(100)

#ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
OS_NAME = os.name
LOGIN_NAME = os.getlogin()

class Vocab:
    def __init__(self, data, labels, load_embedding="", word_embedding_size=300):
        if os.path.exists(os.path.join(ROOT_PATH, "vocab.json")) and os.path.exists(os.path.join(ROOT_PATH, "vocab_word_embedding.pt")):
            vocab_dict = json.load(open(os.path.join(ROOT_PATH, "vocab.json"), "r", encoding='utf-8'))
            self.labels = vocab_dict["labels"]
            self.unk_words = vocab_dict["unk_words"]
            self.label2index = vocab_dict["label2index"]
            self.index2label = vocab_dict["index2label"]
            self.word2index = vocab_dict["word2index"]
            self.index2word = vocab_dict["index2word"]
            #self.word_embedding = torch.load(os.path.join(ROOT_PATH, 'vocab_word_embedding.pt'))
        else:
            word_embedding_dictionary = {}
            self.labels = labels
            self.unk_words = []
            self.label2index = {}
            self.index2label = []
            for i in range(len(self.labels)):
                #for j in range(len(self.labels[i])):
                self.label2index[" ".join(self.labels[i])] = i
                self.index2label.append(" ".join(self.labels[i]))

            self.word2index = {"<pad>": 0, "<unk>": 1}
            self.index2word = ["<pad>", "<unk>"]
            self.word_frequency = {}
            #must load from pretrained model
            load_embedding_content = open(load_embedding, "r", encoding='utf-8').readlines()
            for i in range(1, len(load_embedding_content)):
                tmp = load_embedding_content[i].rstrip().split(" ")
                word_vector = tmp[1:]
                if len(word_vector) != 300:
                    if len(word_vector) == 301:
                        word_vector = word_vector[1:]
                    else:
                        continue
                word_embedding_dictionary[tmp[0]] = torch.FloatTensor([float(one_str) for one_str in word_vector])
            word_index_count = len(self.word2index.keys())
            for one_word in data:
                if one_word not in self.word_frequency:
                    self.word_frequency[one_word] = 1
                else:
                    self.word_frequency[one_word] += 1
            self.word_frequency = {k: v for k, v in sorted(self.word_frequency.items(),
                                 key=lambda item: item[1],
                                 reverse=True)}
            for one_word in self.word_frequency:
                if one_word not in self.word2index:
                    self.word2index[one_word] = word_index_count
                    self.index2word.append(one_word)
                    word_index_count += 1
            for one_word in word_embedding_dictionary.keys():
                if one_word not in self.word2index:
                    self.word2index[one_word] = word_index_count
                    self.index2word.append(one_word)
                    word_index_count += 1

            assert len(self.word2index.keys()) == len(self.index2word)
            word_embedding_list = []
            for i in range(len(self.index2word)):
                if self.index2word[i] in word_embedding_dictionary:
                    word_embedding_list.append(word_embedding_dictionary[self.index2word[i]].double())
                else:
                    #raise ValueError
                    self.unk_words.append(self.index2word[i])
                    word_embedding_list.append(torch.randn(size=(1, word_embedding_size), dtype=torch.double).uniform_(-1,1).view(word_embedding_size))
            print("total unk words number is {}, words are: {}".format(str(len(self.unk_words)), " ".join(self.unk_words)))
            self.word_embedding = torch.stack(word_embedding_list).double()
            self.save_vocab()

    def save_vocab(self, file_path=os.path.join(ROOT_PATH, "vocab.json")):
        vocab_dict = {}
        vocab_dict["labels"] = self.labels
        vocab_dict["unk_words"] = self.unk_words
        vocab_dict["label2index"] = self.label2index
        vocab_dict["index2label"] = self.index2label
        vocab_dict["word2index"] = self.word2index
        vocab_dict["index2word"] = self.index2word
        json.dump(vocab_dict, open(file_path, "w", encoding='utf-8'))
        torch.save(self.word_embedding, os.path.join(ROOT_PATH, "vocab_word_embedding.pt"))


def compute_subwords(word, minn=1, maxn=4):
    substrings_list = ["".join(word[i: j]) for i in range(len(word))
         for j in range(i + 1, len(word) + 1)]
    '''
    for i in range(len(word)):
        ngram = ""
        j = i
        for n in range(1, maxn+1):
            if not (j < len(word) and n <= maxn):
                break
            while j < len(word):
                ngram += word[j]
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == len(word))):
                substrings_list.append(ngram)
    '''
    return substrings_list

'''
  for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
  ngram.push_back(word[j++]);
  while (j < word.size() && (word[j] & 0xC0) == 0x80) {
    ngram.push_back(word[j++]);
  }
  if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
    int32_t h = hash(ngram) % args_->bucket;
    pushHash(ngrams, h);
    if (substrings) {
      substrings->push_back(ngram);
'''

def get_all_words(data_segmented_list):
    total_doc_num = len(data_segmented_list)
    words_dict = {}
    for i in range(len(data_segmented_list)):
        print(str(i) + "/" + str(total_doc_num))
        song_name = data_segmented_list[i]["song_name"]
        comments_dict = data_segmented_list[i]['song_comments_detail_final']
        for song_id in comments_dict:
            for sent_id in comments_dict[song_id]:
                if sent_id != 'likecnt_weight':
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

def main():
    '''
    wv_from_text = KeyedVectors.load_word2vec_format(
        datapath('/home/cide/workspace/keyphrase-extraction-generation-classification/merge_sgns_bigram_char300.txt'),
        binary=False)
    train_data_segmented_list = json.load(open("data/train_data_reviews_filtered_2.json"))
    words_list = get_all_words(train_data_segmented_list)
    new_word_vector_dict = {}
    for one_word in words_list:
        word_subword_vectors = []
        if one_word not in wv_from_text.vocab:
            subwords_list = compute_subwords(list(one_word))
            for one_sub_word in subwords_list:
                if one_sub_word in wv_from_text:
                    word_subword_vectors.append(wv_from_text[one_sub_word].tolist())
        if len(word_subword_vectors) > 0:

            one_word_vector = np.array(word_subword_vectors).sum(axis=0) / float(len(word_subword_vectors))
            new_word_vector_dict[one_word] = one_word_vector.tolist()
    json.dump(new_word_vector_dict, open("new_word_vector_dict.json", "w", encoding='utf-8'))
    '''
    original_embedding_path = os.path.join(ROOT_PATH, "merge_sgns_bigram_char300.txt")
    original_list = open(original_embedding_path).readlines()
    new_word_vector_dict = json.load(open('new_word_vector_dict.json'))
    new_list = []
    skip_count = 0
    for one_key in new_word_vector_dict:
        line = " ".join([one_key.strip(), " ".join([str(one_num) for one_num in new_word_vector_dict[one_key]])])
        parts = utils.to_unicode(line.rstrip(), encoding='utf-8', errors='strict').split(" ")
        if len(parts) != 300 + 1:
            #print("debug")
            skip_count += 1
            continue
        #new_list.append(" ".join([one_key, " ".join([str(round(one_num, 6)) for one_num in new_word_vector_dict[one_key]])]))
        new_list.append(line)
        #" ".join(str(new_word_vector_dict[one_key]).split(","))
    list_length, dim = original_list[0].split(" ")
    list_length = int(list_length) + len(new_list)
    original_list[0] = " ".join([str(list_length), dim])
    new_first_line = " ".join([str(len(new_list)+1), dim])
    print("skip count: {}".format(skip_count))
    #open('/home/cide/workspace/keyphrase-extraction-generation-classification/merge_sgns_bigram_char300_2.txt',
    #     "w", encoding='utf-8').write("".join(original_list) + "\n".join(new_list))
    #final_str = original_list[0] + "\n".join(new_list) + "".join(original_list[1:]

    open(os.path.join(ROOT_PATH, 'new_list.txt'),
         "w", encoding='utf-8').write(new_first_line + "\n".join(new_list))
    open(os.path.join(ROOT_PATH, 'merge_sgns_bigram_char300_3.txt'),
         "w", encoding='utf-8').write(original_list[0] + "\n".join(new_list) + "\n" + "".join(original_list[1:]))

def build_customized_embdding(base_embedding, data_list):
    wv_from_text = KeyedVectors.load_word2vec_format(
        datapath(base_embedding),
        binary=False)
    words_list = get_all_words(data_list)
    new_word_vector_dict = {}
    for one_word in words_list:
        word_subword_vectors = []
        if one_word not in wv_from_text:
            subwords_list = compute_subwords(list(one_word))
            for one_sub_word in subwords_list:
                if one_sub_word in wv_from_text:
                    word_subword_vectors.append(wv_from_text[one_sub_word].tolist())
        if len(word_subword_vectors) > 0:
            one_word_vector = np.array(word_subword_vectors).sum(axis=0) / float(len(word_subword_vectors))
            new_word_vector_dict[one_word] = one_word_vector.tolist()
    original_list = open(base_embedding, "r", encoding='utf-8').readlines()
    #original_list = open(
    #    '/home/cide/workspace/keyphrase-extraction-generation-classification/merge_sgns_bigram_char300.txt').readlines()
    new_list = []
    skip_count = 0
    for one_key in new_word_vector_dict:
        line = " ".join([one_key.strip(), " ".join([str(one_num) for one_num in new_word_vector_dict[one_key]])])
        parts = utils.to_unicode(line.rstrip(), encoding='utf-8', errors='strict').split(" ")
        if len(parts) != 300 + 1:
            # print("debug")
            skip_count += 1
            continue
        # new_list.append(" ".join([one_key, " ".join([str(round(one_num, 6)) for one_num in new_word_vector_dict[one_key]])]))
        new_list.append(line)
        # " ".join(str(new_word_vector_dict[one_key]).split(","))
    list_length, dim = original_list[0].split(" ")
    print("original embedding head: {}".format(original_list[0]))
    print("original embedding has lines: {}".format(str(len(original_list[1:]))))
    list_length = int(list_length) + len(new_list)
    original_list[0] = " ".join([str(list_length), dim])
    new_first_line = " ".join([str(len(new_list) + 1), dim])
    print("skip count: {}".format(skip_count))
    # open('/home/cide/workspace/keyphrase-extraction-generation-classification/merge_sgns_bigram_char300_2.txt',
    #     "w", encoding='utf-8').write("".join(original_list) + "\n".join(new_list))
    # final_str = original_list[0] + "\n".join(new_list) + "".join(original_list[1:]
    #open('/home/cide/workspace/keyphrase-extraction-generation-classification/new_list.txt',
    #     "w", encoding='utf8').write(new_first_line + "\n".join(new_list))
    print("new embedding head: {}".format(original_list[0]))
    print("new embedding has lines: {}".format(str(len(new_list) + len(original_list[1:]))))

    open(os.path.join(ROOT_PATH, "merge_sgns_bigram_char300_3.txt"),
         "w", encoding='utf-8').write(original_list[0] + "\n".join(new_list) + "\n" + "".join(original_list[1:]))


if __name__ == '__main__':
    main()
