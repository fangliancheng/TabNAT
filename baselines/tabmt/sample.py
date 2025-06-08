import os
import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer

from utils_train import preprocess
from baselines.tabmt.models.model import TabMAR

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
    enable_padding = True if args.padding == 1 else False
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = f'{script_path}/../../data/{dataname}'

    device =  args.device
    info_path = f'{script_path}/../../data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    column_names = [info['column_names'][i] for i in num_col_idx]

    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(data_dir, task_type = task_type, inverse = True, concat = True)
    len_ori_cat = len(categories)
    
    X_cat_train = X_cat[0]

    if X_num is not None:
        X_num = X_num[0]
        # discretize numerical features
        enc = KBinsDiscretizer(n_bins=[100 for _ in range(n_num)], encode='ordinal', strategy='uniform', subsample=None)
        X_num_disc = enc.fit_transform(X_num).astype(np.int32)

        X_cat = np.concatenate([X_num_disc, X_cat_train], axis=1)
        categories = [100 for _ in range(n_num)] + categories

    n_cat = len(categories)
    n_num = 0

    X_train_num = None
    X_train_cat = torch.tensor(X_cat_train)

    model = TabMAR(n_num, n_cat, categories, 
                    embed_dim = embed_dim, 
                    buffer_size = buffer_size, 
                    depth = depth, 
                    norm_layer = nn.LayerNorm, 
                    dropout_rate = args.dropout_rate,
                    padding = enable_padding,
                    device = device).to(device)

    # load model checkpoint
    model.load_state_dict(torch.load(f'TabMT/checkpoints/{dataname}/model_{embed_dim}_{buffer_size}_{depth}_{args.padding}.pt'))

    model.eval()

    # Sampling
    start_time = time.time()
    with torch.no_grad():
        B = X_cat_train.shape[0]
        syn_num, syn_cat = model.sample(B, device=device)
        
        assert syn_num is None
        syn_cat = syn_cat.cpu().numpy()
        
        if X_num is not None:
            # reverse KBinsDiscretizer
            syn_num_disc = enc.inverse_transform(syn_cat[:, :-len_ori_cat]).astype(np.float64)
            syn_num = num_inverse(syn_num_disc)
    
        # pad_syn_cat = np.zeros((B, len_ori_cat - X_train_cat.shape[1]))
        # pad_syn_cat = np.concatenate([syn_cat, pad_syn_cat], axis = 1)

        #syn_cat = cat_inverse(pad_syn_cat)
        syn_cat = cat_inverse(syn_cat[:,-len_ori_cat:])
    
    syn_df = pd.DataFrame()
    
    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_set = set(num_col_idx)
    cat_set = set(cat_col_idx)

    for i in range(n_cat):

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
    
    save_path = f'{script_path}/../../synthetic/{dataname}/tabmt_{embed_dim}_{buffer_size}_{depth}_{args.padding}.csv'
    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))
        
        