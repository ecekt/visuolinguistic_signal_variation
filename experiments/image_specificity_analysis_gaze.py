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

from transformers import AutoTokenizer, GPT2Model, GPT2Config, set_seed, get_scheduler

from tqdm.auto import tqdm
import datetime


class CustomCLIP_FixOnsetModel(nn.Module):
    def __init__(self, img_dim, hidden_dim, fix_dim):
        super(CustomCLIP_FixOnsetModel, self).__init__()
        self.dropout = nn.Dropout(0.1)

        # project image embeddings to hidden dimensions
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.fix_dim = fix_dim

        self.linear_map = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_fix2hid = nn.Linear(self.fix_dim, self.hidden_dim)
        self.linear_single = nn.Linear(2 * self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights()  # initialize layers

    def init_weights(self):
        for ll in [self.linear_map, self.linear_single, self.linear_fix2hid]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, attention_mask, img_vectors,  fix_vectors, scanpath_lens):

        batch_size = img_vectors.shape[0]

        img_vectors = self.dropout(img_vectors)
        image_prefix = self.linear_map(img_vectors)
        image_prefix = self.tanh(image_prefix)

        fix_vectors = self.dropout(fix_vectors)
        fix_prefix = self.linear_fix2hid(fix_vectors)

        fix_prefix_sum = torch.sum(fix_prefix, dim=1)
        fix_prefix_avg = fix_prefix_sum / scanpath_lens.unsqueeze(1)
        fix_prefix_avg = self.tanh(fix_prefix_avg)

        full_prefix = torch.cat([image_prefix, fix_prefix_avg], dim=-1)

        logits = self.linear_single(full_prefix)
        logits = self.relu(logits)

        return_dict = {'logits': logits}

        return return_dict


class CustomCLIP_GPT_FixOnsetModel(nn.Module):
    def __init__(self, img_dim, hidden_dim, fix_dim, no_layers):
        super(CustomCLIP_GPT_FixOnsetModel, self).__init__()
        # custom untrained GPT2-style model
        config = GPT2Config(n_layer=no_layers)
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)

        # project image embeddings to hidden dimensions
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.fix_dim = fix_dim

        self.linear_map = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_fix2hid = nn.Linear(self.fix_dim, self.hidden_dim)
        self.linear_single = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights()  # initialize layers

    def init_weights(self):
        for ll in [self.linear_map, self.linear_single, self.linear_fix2hid]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, attention_mask, img_vectors, fix_vectors, scanpath_lens):

        batch_size = img_vectors.shape[0]

        img_vectors = self.dropout(img_vectors)
        image_prefix = self.linear_map(img_vectors)
        image_prefix = self.tanh(image_prefix)

        fix_vectors = self.dropout(fix_vectors)
        fix_prefix = self.linear_fix2hid(fix_vectors)
        fix_prefix = self.tanh(fix_prefix)

        full_prefix = torch.cat([image_prefix.unsqueeze(1), fix_prefix], dim=1)

        outputs_gpt = self.gpt2(inputs_embeds=full_prefix, attention_mask=attention_mask)

        # max timestep last
        # last_hidden = outputs_gpt.last_hidden_state[:, -1]

        # actual last based on scanpath lengths (since I give the length, it is +1 as CLIP at 0)
        last_hidden = torch.stack([outputs_gpt.last_hidden_state[s, scanpath_lens[s]] for s in range(len(scanpath_lens))])

        logits = self.linear_single(last_hidden)
        logits = self.relu(logits)

        return_dict = {'logits': logits}

        return return_dict


class CustomCLIP_GPTBOW_FixOnsetModel(nn.Module):
    def __init__(self, img_dim, hidden_dim, fix_dim, no_layers):
        super(CustomCLIP_GPTBOW_FixOnsetModel, self).__init__()
        # custom untrained GPT2-style model
        config = GPT2Config(n_layer=no_layers)
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)

        # project image embeddings to hidden dimensions
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.fix_dim = fix_dim

        self.linear_map = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_fix2hid = nn.Linear(self.fix_dim, self.hidden_dim)
        self.linear_single = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights()  # initialize layers

    def init_weights(self):
        for ll in [self.linear_map, self.linear_single, self.linear_fix2hid]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, attention_mask, img_vectors, fix_vectors, scanpath_lens):

        batch_size = img_vectors.shape[0]

        img_vectors = self.dropout(img_vectors)
        image_prefix = self.linear_map(img_vectors)
        image_prefix = self.tanh(image_prefix)

        fix_vectors = self.dropout(fix_vectors)
        fix_prefix = self.linear_fix2hid(fix_vectors)

        fix_prefix_sum = torch.sum(fix_prefix, dim=1)
        fix_prefix_avg = fix_prefix_sum / scanpath_lens.unsqueeze(1)
        fix_prefix_avg = self.tanh(fix_prefix_avg)

        full_prefix = torch.cat([image_prefix.unsqueeze(1), fix_prefix_avg.unsqueeze(1)], dim=1)
        # only 2 items
        attention_mask = attention_mask[:, :2]
        outputs_gpt = self.gpt2(inputs_embeds=full_prefix, attention_mask=attention_mask)

        # max timestep last
        # last_hidden = outputs_gpt.last_hidden_state[:, -1]

        # 2 items
        last_hidden = outputs_gpt.last_hidden_state[:, -1]

        logits = self.linear_single(last_hidden)
        logits = self.relu(logits)

        return_dict = {'logits': logits}

        return return_dict


def eval_predict_onset(model, split_dataloader, model_type, data_length, split_name):

    loss_eval = 0.0
    loss_eval_sqrt = 0.0
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
            outputs = model(attention_mask=bt['attention_mask'], img_vectors=bt['img_vector'],
                            fix_vectors=bt['fix_vectors'], scanpath_lens=bt['scanpath_lens'])

            preds = outputs['logits']

            cur_pred_min = min(preds)
            cur_pred_max = max(preds)

            if cur_pred_min < pred_min:
                pred_min = cur_pred_min

            if cur_pred_max > pred_max:
                pred_max = cur_pred_max

            target = bt['speech_onset']
            loss_b = criterion(preds, target)
            losses_b_indiv = [math.sqrt(lsb.item()) for lsb in loss_b]
            # print('pred:', preds)
            # print('target:', target)

            loss_eval += sum(loss_b).item()
            loss_eval_sqrt += sum(losses_b_indiv)

            list_bertje.extend([btj.item() for btj in bertje_score])
            list_onset_target.extend([tgt.item() for tgt in target])
            list_onset_pred.extend([pds.item() for pds in preds])

            # bs 1
            dict_bertje[bt['img_id'].item()].append(bertje_score.item())
            dict_onset_target[bt['img_id'].item()].append(target.item())
            dict_onset_pred[bt['img_id'].item()].append(preds.item())

            if str(bt['img_id'].item()) == '498039':
                print(target.item(), preds.item())

    print(split_name, ' loss total raw', loss_eval)
    loss_ret = round(loss_eval_sqrt / data_length, 3)
    print(split_name, ' ONSET interpretable loss - avg sqrt', loss_ret)
    print('min', pred_min.item())
    print('max', pred_max.item())
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
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-bs', type=int, default=4)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-subset_size',type=int, default=-1)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-shuffle', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-hidden_dim', type=int, default=768)
    parser.add_argument('-reps', type=str)
    parser.add_argument('-nlayers', type=int, default=12)

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)
    print('code starts', timestamp)

    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    print('DIDEC ANALYSIS onset with Gaze input ' + model_type)

    learning_rate = args.lr
    batch_size = args.bs
    num_epochs = args.epoch

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    subset_size = args.subset_size

    debug = args.debug

    hidden_dim = args.hidden_dim

    no_layers = args.nlayers

    args_dev = args.device

    reps = args.reps

    # use the one obtained on CPU as it produces dtype float32 tensors
    # compatible with linear layers etc.
    with open('data/didec/clip_preprocessed_didec_images_cpu.pickle', 'rb') as f:
        clip_preprocessed_images = pickle.load(f)

    if reps == 'l14':
        with open('data/didec/clip_preprocessed_didec_fx2SAMcropped_L14_cpu.pickle', 'rb') as f:
            clip_preprocessed_fixations = pickle.load(f)
    elif reps == 'b32':
        with open('data/didec/clip_preprocessed_didec_fx2SAMcropped_B32_cpu.pickle', 'rb') as f:
            clip_preprocessed_fixations = pickle.load(f)

    image_dim = [clip_preprocessed_images[c].shape[1] for c in clip_preprocessed_images][0]

    if reps == 'l14':
        fixation_dim = 768
    elif reps == 'b32':
        fixation_dim = 512

    # 512 for CLIP ViT B/32
    # 768 for CLIP ViT L/14

    with open('data/didec/filtered_fixation_counts_BEFOREgist1000_DS_2023.pickle', 'rb') as f:
        filtered_count = pickle.load(f)

    max_scanpath_len = 7  # from the filtered file above
    pad_embedding = torch.zeros(1, fixation_dim)

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

                text_str = ' '.join(text)
                dataset_param[im][ppn] = {'text': text_str,
                                          'image': clip_preprocessed_images[im].squeeze(0),
                                          'fixations': clip_preprocessed_fixations[(ppn, im)],
                                          'fixation_counts_beforeonset': filtered_count[(ppn, im)],
                                          'delay': delay}

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

                fixations = ds[im][ppn]['fixations']

                len_fixations = len(fixations)
                len_fixations_filtered = ds[im][ppn]['fixation_counts_beforeonset']

                fix_padded = []
                for ind in range(len_fixations_filtered):
                    fix_padded.append(fixations[str(ind)])

                to_pad = max_scanpath_len - len_fixations_filtered
                fix_padded += [pad_embedding] * to_pad

                # CLIP + FIXATIONS + PADS
                attention_masks = [1] + len_fixations_filtered * [1] + to_pad * [0]

                if len_fixations_filtered == 0:
                    len_fixations_filtered += 1
                    # to avoid nan in calculations

                dataset_param_new.append({'attention_mask': torch.tensor(attention_masks),
                                          'img_vector': ds[im][ppn]['image'],
                                          'speech_onset': ds[im][ppn]['delay'],
                                          'fix_vectors': torch.stack(fix_padded).squeeze(1),
                                          'scanpath_lens': torch.tensor(len_fixations_filtered),
                                          'bertje_cls': torch.tensor(bertje_cls[im]),
                                          'img_id': torch.tensor(int(im))
                                          })

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

    if model_type == 'clip':
        model = CustomCLIP_FixOnsetModel(img_dim=image_dim, hidden_dim=hidden_dim, fix_dim=fixation_dim)
    elif model_type == 'gpt':
        model = CustomCLIP_GPT_FixOnsetModel(img_dim=image_dim, hidden_dim=hidden_dim, fix_dim=fixation_dim, no_layers=no_layers)
    elif model_type == 'gptbow':
        model = CustomCLIP_GPTBOW_FixOnsetModel(img_dim=image_dim, hidden_dim=hidden_dim, fix_dim=fixation_dim,
                                             no_layers=no_layers)

    checkpoint = 'models_onset_gazeinput_gpt/predict_onset_gazeinput_gpt_0.0001_16_42_2023-06-21-15-52-42.pt'
    checkpoint = torch.load(checkpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction='none')

    model.eval()

    loss_train = eval_predict_onset(model, train_dataloader, model_type, data_length_train, "train")
    loss_val = eval_predict_onset(model, val_dataloader, model_type, data_length_val, "val")

    loss_test = eval_predict_onset(model, test_dataloader, model_type, data_length_test, "test")

