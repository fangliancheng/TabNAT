import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils_train import preprocess
from tabnat.model import TabNAT

warnings.filterwarnings('ignore')

def main(args):

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
        
    dataname = args.dataname
    embed_dim = args.embed_dim
    buffer_size = args.buffer_size
    depth = args.depth
    data_dir = f'data/{dataname}'

    
    device =  args.device
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    column_names = [info['column_names'][i] for i in num_col_idx]

    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(data_dir, task_type = task_type, inverse = True, concat = True)
    n_cat = len(categories)
    len_ori_cat = len(categories)
    if args.n_num != -1:
        n_num = args.n_num
    if args.n_cat != -1:
        n_cat = args.n_cat
    
    has_num = n_num > 0 
    has_cat = n_cat > 0
    print(f'has_num: {has_num}, has_cat: {has_cat}')

    if has_num:
        X_train_num, X_test_num = X_num
    else:
        X_train_num, X_test_num = None, None
    if has_cat:
        X_train_cat, X_test_cat = X_cat
    else:
        X_train_cat, X_test_cat = None, None

    if has_num:
        X_train_num = X_train_num[:, :n_num]
        X_test_num = X_test_num[:, :n_num]
        X_train_num = torch.tensor(X_train_num).float()
        X_test_num = torch.tensor(X_test_num).float()

        mean, std = X_train_num.mean(0), X_train_num.std(0)
        X_train_num = (X_train_num - mean) / std / 2
        X_test_num = (X_test_num - mean) / std / 2

        X_train_num = X_train_num.float()

    if has_cat:
        X_train_cat = X_train_cat[:, :n_cat]
        X_test_cat = X_test_cat[:, :n_cat]
        X_train_cat = torch.tensor(X_train_cat)
        X_test_cat = torch.tensor(X_test_cat)
        categories = categories[:n_cat]

    n_cat_model = n_cat

    model = TabNAT(
                n_num=n_num, n_cat=n_cat_model, categories=categories, 
                embed_dim = embed_dim, 
                buffer_size = buffer_size, 
                depth = depth, 
                norm_layer = nn.LayerNorm, 
                dropout_rate = args.dropout_rate,
                device = device).to(device)

    # load model checkpoint
    model.load_state_dict(torch.load(f'checkpoints/{dataname}/model.pt'))

    model.eval()

    # Sampling
    start_time = time.time()
    with torch.no_grad():
        
        B = X_train_cat.shape[0]
        ret_num = []
        ret_cat = []
        for cls in range(categories[0]):
            cur_bsz = (X_train_cat == cls).sum()
            syn_num, syn_cat = model.sample(cur_bsz, torch.full((cur_bsz,), cls, device=device), device=device)
            ret_num.append(syn_num)
            ret_cat.append(torch.full((cur_bsz, 1), cls, device=device))
        syn_num = torch.cat(ret_num, dim=0)
        syn_cat = torch.cat(ret_cat, dim=0)

        if syn_num is not None:
            syn_num = (syn_num.cpu() * 2 * std) + mean
            syn_num = syn_num.numpy()
        
            syn_num = num_inverse(syn_num)
        
        if syn_cat is not None:
            syn_cat = syn_cat.cpu().numpy()
            pad_syn_cat = np.zeros((B, len_ori_cat - X_train_cat.shape[1]))
            pad_syn_cat = np.concatenate([syn_cat, pad_syn_cat], axis = 1)

            syn_cat = cat_inverse(pad_syn_cat) 
    

    syn_df = pd.DataFrame()
    
    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    # num_col_idx = info['target_col_idx'] + info['num_col_idx'] if info['task_type'] == 'regression' else info['num_col_idx']
    # cat_col_idx = info['target_col_idx'] + info['cat_col_idx'] if info['task_type'] == 'binclass' else info['cat_col_idx']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_set = set(num_col_idx)
    cat_set = set(cat_col_idx)

    for i in range(n_num + n_cat):

        if info['task_type'] == 'regression':
            if i in num_set and syn_num is not None:
                syn_df[i] = syn_num[:, idx_mapping[i] + 1]
            elif i in cat_set and syn_cat is not None:
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_set)]
            elif i in info['target_col_idx']:
                syn_df[i] = syn_num[:, 0]
        else:
            if i in num_set and syn_num is not None:
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in cat_set and syn_cat is not None:
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_set) + 1]
            elif i in info['target_col_idx']:
                syn_df[i] = syn_cat[:, 0]

    column_names = info['column_names']
    try:
        syn_df.columns = column_names
    except ValueError: # if mismatch, it is due to not concat X with target column.
        syn_df.columns = column_names[:-1]
    
    end_time = time.time()
    print(f'Sampling time: {end_time - start_time:.2f} seconds')

    save_path = f'synthetic/{dataname}/tabdar.csv'
    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))
        
        