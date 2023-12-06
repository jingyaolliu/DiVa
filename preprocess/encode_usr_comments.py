import torch
from transformers import XLNetModel, XLNetTokenizer
from utils import *


def encode_user_comments_xlnet(data_dict, comment_vdict={}):
    """
    Encodes the user comments in each data item in the data dictionary using XLNet.

    Args:
        data_dict (dict): A dictionary of data items.

    Returns:
        comment_vdict (dict): A dictionary containing the XLNet encodings for all user comments.
    """
    # Load the XLNet model and tokenizer
    model_name = 'hfl/chinese-xlnet-base'
    model = XLNetModel.from_pretrained(model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    # Set the device to use for encoding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the layers to use for encoding
    layers = [-1, 0]

    # Encode the user comments for each data item in the data dictionary
    for data_item_key in data_dict:
        data_item = data_dict[data_item_key]
        user_comments = data_item['user_comments']
        # Join the user comments into a single string separated by [SEP]
        user_comments_str = "[SEP]".join(user_comments)
        # Tokenize the user comments string
        input_ids = torch.tensor([tokenizer.encode(user_comments_str, add_special_tokens=True)])
        # Move the input_ids tensor to the device
        input_ids = input_ids.to(device)
        # Encode the input_ids with XLNet
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            encoded_layers = [outputs.hidden_states[i] for i in layers]
        # Concatenate the first and last layers and move the result to the CPU
        encoded_layers = [layer.cpu().numpy() for layer in encoded_layers]
        encoded_layers_concat = np.concatenate((encoded_layers[0], encoded_layers[-1]), axis=1)
        # Update the comment_vdict with the XLNet encoding
        comment_vdict[data_item_key] = encoded_layers_concat

    return comment_vdict


if __name__=='__main__':
    # Load test_dict and train_dict
    data_dir = '../data'
    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        train_dict = json.load(f)
    with open(os.path.join(data_dir, 'test1.json'), 'r', encoding='utf-8') as f:
        test_dict = json.load(f)

    # Encode user comments in test_dict and train_dict and return a commentvdict
    commentvdict = encode_user_comments_xlnet(train_dict)
    commentvdict.update(encode_user_comments_xlnet(test_dict))

    np.save(os.path.join(data_dir, 'commentvdict.npy'), commentvdict)