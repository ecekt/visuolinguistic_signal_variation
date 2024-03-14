import torch
import os
from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import json

with open('data/didec/didec_image_specificity_bertjecls.json', 'r') as f:
    bertje_cls = json.load(f)

with open('data/didec/didec_image_specificity_bertjepooler.json', 'r') as f:
    bertje_pool = json.load(f)

cls = []
pool = []

for im in bertje_cls:
    cls.append(bertje_cls[im])
    pool.append(bertje_pool[im])

corr, pvalue = spearmanr(cls, pool)
print('correlation img spec, bertje cls vs. bertje pooled')
print(round(corr, 3), round(pvalue, 3))
# 0.981 0.0



import csv
import json
import math
import pickle
from collections import defaultdict
import numpy as np
import torch
import argparse
import os

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed, get_scheduler

from tqdm.auto import tqdm
import datetime


class CustomCLIPOnsetModel(nn.Module):
    def __init__(self, img_dim, hidden_dim):
        super(CustomCLIPOnsetModel, self).__init__()
        self.dropout = nn.Dropout(0.1)

        # project image embeddings to hidden dimensions
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.linear_map = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_single = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights()  # initialize layers

    def init_weights(self):
        for ll in [self.linear_map, self.linear_single]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_ids, attention_mask, img_vectors):

        batch_size = input_ids.shape[0]

        image_prefix = self.linear_map(img_vectors)
        image_prefix = self.tanh(image_prefix)
        logits = self.linear_single(image_prefix)
        logits = self.relu(logits)

        return_dict = {'logits': logits}

        return return_dict



class CustomCLIPOnsetCategoryModel(nn.Module):
    def __init__(self, img_dim, hidden_dim, category_dim):
        super(CustomCLIPOnsetCategoryModel, self).__init__()
        self.dropout = nn.Dropout(0.1)

        # project image embeddings to hidden dimensions
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim
        self.linear_map = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_category = nn.Linear(self.hidden_dim, self.category_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights()  # initialize layers

    def init_weights(self):
        for ll in [self.linear_map, self.linear_category]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_ids, attention_mask, img_vectors):

        batch_size = input_ids.shape[0]

        image_prefix = self.linear_map(img_vectors)
        image_prefix = self.tanh(image_prefix)
        logits = self.linear_category(image_prefix)
        logits = self.relu(logits)

        return_dict = {'logits': logits}

        return return_dict


def eval_predict_onset(model, split_dataloader, model_type, data_length, split_name):

    loss_eval = 0.0
    loss_eval_sqrt = 0.0
    eval_acc = 0.0
    count_sys2_targets = 0
    count_sys2_preds = 0
    pred_min = math.inf
    pred_max = -math.inf

    list_bertje = []
    list_onset_target = []
    list_onset_pred = []

    dict_bertje = defaultdict(list)
    dict_onset_target = defaultdict(list)
    dict_onset_pred = defaultdict(list)

    for bt in split_dataloader:
        bt = {k: v.to(device) for k, v in bt.items()}

        bertje_score = bt['bertje_cls']

        with torch.no_grad():
            outputs = model(input_ids=bt['input_ids'], attention_mask=bt['attention_mask'],
                            img_vectors=bt['img_vector'])

            preds = outputs['logits']

            if model_type == 'regression':

                cur_pred_min = min(preds)
                cur_pred_max = max(preds)

                if cur_pred_min < pred_min:
                    pred_min = cur_pred_min

                if cur_pred_max > pred_max:
                    pred_max = cur_pred_max

                target = bt['speech_onset']
                loss_b = criterion(preds, target)
                losses_b_indiv = [math.sqrt(lsb.item()) for lsb in loss_b]
                loss_eval_sqrt += sum(losses_b_indiv)
                loss_eval += sum(loss_b).item()

                list_bertje.extend([btj.item() for btj in bertje_score])
                list_onset_target.extend([tgt.item() for tgt in target])
                list_onset_pred.extend([pds.item() for pds in preds])

                # bs 1
                dict_bertje[bt['img_id'].item()].append(bertje_score.item())
                dict_onset_target[bt['img_id'].item()].append(target.item())
                dict_onset_pred[bt['img_id'].item()].append(preds.item())

                if str(bt['img_id'].item()) == '498039':
                    print(target.item(), preds.item())

                # print('pred:', preds)
                # print('target:', target)

            elif model_type == 'categorical':
                target = bt['speech_cat']
                loss_b = criterion_categorical(preds, target)

                eval_acc += sum(torch.argmax(preds, dim=-1) == target).item()
                loss_eval += loss_b.item()

                count_sys2_targets += sum(target)
                count_sys2_preds += sum(torch.argmax(preds, dim=-1))

    print(split_name, ' loss total raw', loss_eval)

    if model_type == 'regression':
        loss_ret = round(loss_eval_sqrt / data_length, 3)
        print(split_name, ' ONSET interpretable loss - avg sqrt', loss_ret)
        print('min', pred_min.item())
        print('max', pred_max.item())

    elif model_type == 'categorical':
        loss_ret = round(eval_acc / data_length, 4)
        print(split_name, ' category system accuracy', loss_ret)
        print('sys2 target', count_sys2_targets)
        print('sys2 preds', count_sys2_preds)

    print()

    print(split_name, 'correlation specificity vs. target onset')
    corr, pvalue = spearmanr(list_bertje, list_onset_target)
    print(round(corr, 3), round(pvalue, 3))

    print(split_name, 'correlation specificity vs. predicted onset')
    corr, pvalue = spearmanr(list_bertje, list_onset_pred)
    print(round(corr, 3), round(pvalue, 3))

    print(split_name, 'correlation predicted vs. target onset')
    corr, pvalue = spearmanr(list_onset_pred, list_onset_target)
    print(round(corr, 3), round(pvalue, 3))

    print('\n')

    list_bertje_avg = []
    list_onset_target_avg = []
    list_onset_pred_avg = []

    for i in dict_bertje:
        list_bertje_avg.append(np.mean(dict_bertje[i]))
        list_onset_target_avg.append(np.mean(dict_onset_target[i]))
        list_onset_pred_avg.append(np.mean(dict_onset_pred[i]))

    print()

    print(split_name, 'AVG correlation specificity vs. target onset')
    corr, pvalue = spearmanr(list_bertje_avg, list_onset_target_avg)
    print(round(corr, 3), round(pvalue, 3))

    print(split_name, 'AVG correlation specificity vs. predicted onset')
    corr, pvalue = spearmanr(list_bertje_avg, list_onset_pred_avg)
    print(round(corr, 3), round(pvalue, 3))

    print(split_name, 'AVG correlation predicted vs. target onset')
    corr, pvalue = spearmanr(list_onset_pred_avg, list_onset_target_avg)
    print(round(corr, 3), round(pvalue, 3))

    print('\n')

    return loss_ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=str)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-bs', type=int, default=4)
    parser.add_argument('-subset_size',type=int, default=-1)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-shuffle', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-hidden_dim', type=int, default=768)
    parser.add_argument('-category_dim', type=int, default=2)
    parser.add_argument('-threshold', type=float, default=3.42)  # train set mean or median onset

    # print('TODO threshold is to be replaced by empirical category threshold (mean, median etc. of the actual onsets)')
    # print('check the systems')

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)
    print('code starts', timestamp)

    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    print('DIDEC ANALYSIS onset ' + model_type)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    subset_size = args.subset_size

    debug = args.debug

    batch_size = args.bs

    hidden_dim = args.hidden_dim
    category_dim = args.category_dim
    threshold = args.threshold

    args_dev = args.device

    # use the one obtained on CPU as it produces dtype float32 tensors
    # compatible with linear layers etc.
    with open('data/didec/clip_preprocessed_didec_images_cpu.pickle', 'rb') as f:
        clip_preprocessed_images = pickle.load(f)

    image_dim = [clip_preprocessed_images[c].shape[1] for c in clip_preprocessed_images][0]
    # 512 for CLIP

    datapaths = {'train': 'data/didec/split_train.json',
                 'val': 'data/didec/split_val.json',
                 'test': 'data/didec/split_test.json'}

    original_data = dict()

    for split in datapaths:

        print(split)
        path = datapaths[split]

        dataset_param = defaultdict(dict)

        with open(path, 'r') as f:
            datasplit = json.load(f)

        count_aligned = 0

        for im in datasplit:
            for ppn in datasplit[im]:

                alignment_file = 'data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + im + '.json'

                with open(alignment_file, 'r') as j:
                    lines = json.load(j)

                    delay = 0
                    count_line = 0
                    text = []

                    if len(lines) == 0:
                        print('empty', f)
                    else:
                        count_aligned += 1
                        print(count_aligned)

                        for line in lines:
                            if count_line == 0:
                                delay = line['start']

                            count_line += 1
                            text.append(line['text'])

                print('speech onset', delay, '\n')
                delay = torch.tensor([delay])
                if delay < threshold:  # 0.5 < 3.42
                    category = 0  # fast
                else:
                    category = 1  # slow
                print('system', str(category + 1), '\n')
                text_str = ' '.join(text)
                dataset_param[im][ppn] = {'text': text_str,
                                          'image': clip_preprocessed_images[im].squeeze(0),
                                          'delay': delay,
                                          'category': category}

        print(split, count_aligned)
        original_data[split] = dataset_param

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch")

    # https://github.com/huggingface/transformers/issues/3311#issuecomment-693719190
    # It's not really a bug because the default behavior of GPT2 is ' \
    # 'to just not add bos or eos tokens. GPT2 is mainly used to generate' \
    # ' text so it would not make a lot of sense to add a EOS of a input prompt. ' \
    # 'If one wants he could just manually add gpt2_tokenizer.eos_token to the input and ' \
    # 'the eos_token_id will be added

    sos = tokenizer.convert_tokens_to_ids('<s>')
    eos = tokenizer.convert_tokens_to_ids('</s>')
    pad = tokenizer.convert_tokens_to_ids('<pad>') # unk 0 according to GPT-2
    # https://github.com/huggingface/transformers/issues/2630

    processed_data = dict()

    for split in original_data:

        ds = original_data[split]

        dataset_param_new = []

        sentence_count = 0

        for im in ds:

            for ppn in ds[im]:

                print(sentence_count)
                sentence_count += 1

                tokenized = tokenizer('<s> ' + ds[im][ppn]['text'] + ' </s>', add_special_tokens=False)

                input_ids = tokenized['input_ids']
                attention_masks = tokenized['attention_mask']

                # GPT-2 context size is 1024, much shorter than the descriptions we have, so no truncation
                assert len(input_ids) < 1024

                pad_size = 1024 - len(input_ids)

                input_ids += [pad] * pad_size
                attention_masks += [0] * pad_size

                dataset_param_new.append({'input_ids': torch.tensor(input_ids),
                                          'attention_mask': torch.tensor(attention_masks),
                                          'img_vector': ds[im][ppn]['image'],
                                          'speech_onset': ds[im][ppn]['delay'],
                                          'speech_cat': ds[im][ppn]['category'],
                                          'bertje_cls': torch.tensor(bertje_cls[im]),
                                          'img_id': torch.tensor(int(im))})

        print(split, len(dataset_param_new))
        processed_data[split] = dataset_param_new

    if subset_size == -1:
        train_dataloader = DataLoader(processed_data['train'], shuffle=args.shuffle, batch_size=batch_size)
        val_dataloader = DataLoader(processed_data['val'], shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(processed_data['test'], shuffle=False, batch_size=batch_size)
        train_dataloader_debug = DataLoader(processed_data['train'], shuffle=args.shuffle, batch_size=batch_size)

        data_length_train = len(processed_data['train'])
        data_length_val = len(processed_data['val'])
        data_length_test = len(processed_data['test'])

    else:
        train_dataloader = DataLoader(processed_data['train'][:subset_size], shuffle=args.shuffle, batch_size=batch_size)
        val_dataloader = DataLoader(processed_data['val'][:subset_size], shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(processed_data['test'][:subset_size], shuffle=False, batch_size=batch_size)
        train_dataloader_debug = DataLoader(processed_data['train'][:subset_size], shuffle=args.shuffle,
                                      batch_size=batch_size)

        data_length_train = len(processed_data['train'][:subset_size])
        data_length_val = len(processed_data['val'][:subset_size])
        data_length_test = len(processed_data['test'][:subset_size])

    # training
    if args_dev == 'cpu':
        device = torch.device("cpu")
    elif args_dev == 'cuda':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model_type == 'regression':
        model = CustomCLIPOnsetModel(img_dim=image_dim, hidden_dim=hidden_dim)
    elif model_type == 'categorical':
        model = CustomCLIPOnsetCategoryModel(img_dim=image_dim, hidden_dim=hidden_dim, category_dim=category_dim)

    checkpoint = 'models_onset/predict_gaze_onset_regression_0.0001_32_42_2023-06-21-12-28-4.pt'
    checkpoint = torch.load(checkpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction='none')
    criterion_categorical = nn.CrossEntropyLoss(reduction='sum')

    loss_train = eval_predict_onset(model, train_dataloader, model_type, data_length_train, "train")

    loss_val = eval_predict_onset(model, val_dataloader, model_type, data_length_val, "val")

    loss_test = eval_predict_onset(model, test_dataloader, model_type, data_length_test, "test")



