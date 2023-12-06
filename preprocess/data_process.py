from ltp import LTP
import torch
from utils import *


# Set up argparse to handle command line arguments
parser = argparse.ArgumentParser(description='Process some arguments...')
parser.add_argument('--eval', type=bool, default=True, help='whether to use the model to tokenize')
parser.add_argument('--batch_size',type=int,default=512,help="the batch size of getting comments tokenized")
parser.add_argument('--train_file_name', type=str, default='train', help='the name of your training data file')
parser.add_argument('--test1_file_name', type=str, default='test1', help='the name of your testing(1) data file')
parser.add_argument('--test2_file_name', type=str, default='test2', help='the name of your testing(2) data file')

args = parser.parse_args()

# Define file paths
TrainFilePath = '../data/{}.json'.format(args.train_file_name)
Test1FilePath = '../data/{}.json'.format(args.test1_file_name)
Test2FilePath = '../data/{}.json'.format(args.test2_file_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up file paths for saving data
WordCountPath='../data/word_count.json'


def comment_split_batch(data_dict, save_path, batch_size=args.batch_size):
    """
    Tokenizes comments in a batch and saves the segmented words to a file.

    Args:
        data_dict (dict): A dictionary representing data items.
        save_path (str): The file path to save the tokenized words to.
        batch_size (int): The batch size for tokenizing comments.

    Returns:
        data_dict (dict): The input data with tokenized words added.
    """
    pat = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    length = len(data_dict)
    for i in range(0, length, batch_size):
        batch = data_dict[i:i+batch_size]
        sentences = []
        for data_key in batch:
            data_item = batch[data_key]
            song_comments = ''.join(data_item['user_comments'])
            sentences.append(song_comments)
        segmented_sentences = ltp.pipeline_batch(sentences, tasks=['cws'], return_list=True)
        for j, data_key in enumerate(batch):
            data_dict[data_key]['tokenized_words'] = segmented_sentences[j]
    save_json(save_path, data_dict)     
    return data_dict

def statistic_word_count(train_data_dict, test_data_dict, word_counts_file_path=WordCountPath):
    """
    Counts the frequency of each word in the segmented words of the training and validation data.

    Args:
        train_data_dict (dict): A dictionary representing training data items.
        test_data_dict (dict): A dictionary representing test data items.
        word_counts_file_path (str): The file path to save the word count data to.

    Returns:
        None
    """
    # Combine the segmented words from the training and validation data
    splited_w_lst_all = []
    print('statistic for word count train...')
    for data_key in train_data_dict:
        splited_w_lst_all += train_data_dict[data_key]['tokenized_words']
    print('statistic for word count test...')
    for data_key in test_data_dict:
        splited_w_lst_all += test_data_dict[data_key]['tokenized_words']
    # Count the frequency of each word
    seg_word_dict = Counter(splited_w_lst_all)
    # Save the word count data to a new file
    print('saving data..')
    save_json(word_counts_file_path, seg_word_dict)

def main():
    print('loading data...')
    test_data_1_dict = json.load(open(Test1FilePath))
    # test_data_2_dict = json.load(open(Test2FilePath))
    train_data_dict = json.load(open(TrainFilePath))

    if args.eval:
        # Load LTP model and move it to the device (CPU or GPU)
        print('loading model...')
        ltp = LTP("LTP/base1").to(device)
        # Update LTP vocabulary with <mask> and all labels in the training and test data
        ltp.add_word(word="<mask>", freq=2)
        length = len(train_data_dict) + len(test_data_1_dict)
        index = 0
        for data_dict in [train_data_dict, test_data_1_dict]:
            for data_item in data_dict.values():
                index += 1
                if index % 100 == 0:
                    print('-------------{}/{}------------'.format(index, length))
                song_labels = data_item['labels']
                for song_label in song_labels:
                    ltp.add_word(word=song_label, freq=2)
    train_data_dict = comment_split_batch(train_data_dict, TrainFilePath)
    test_data_1_dict = comment_split_batch(test_data_1_dict, Test1FilePath)
    statistic_word_count(train_data_dict, test_data_1_dict)

#main
if __name__ == '__main__':
    main()