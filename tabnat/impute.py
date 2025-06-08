import warnings
import json

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from utils_train import preprocess, get_eval
from tabnat.model import TabNAT

warnings.filterwarnings('ignore')

def main(args):

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    device =  args.device
    dataname = args.dataname
    embed_dim = args.embed_dim
    buffer_size = args.buffer_size
    depth = args.depth
    one_step = True if args.one_step == 1 else False
    
    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    column_names = [info['column_names'][i] for i in num_col_idx]

    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(data_dir, task_type=task_type, inverse=True, concat=True)
    n_cat = len(categories)
    if args.n_num != -1:
        n_num = args.n_num
    if args.n_cat != -1:
        n_cat = args.n_cat

    B = X_cat[1].shape[0]
    # load mask from .npy file
    mask_path = f'data/{dataname}/rate30/mask.npy'
    impute_mask = np.load(mask_path)
    impute_mask = torch.tensor(impute_mask)
    
    has_num = n_num > 0 
    has_cat = n_cat > 0
    
    print(f'has_num: {has_num}, has_cat: {has_cat}')

    if has_num:
        X_train_num, X_test_num = X_num

        X_train_num = X_train_num[:, :n_num]
        X_test_num = X_test_num[:, :n_num]

        X_train_num_ori = num_inverse(X_train_num)
        X_test_num_ori = num_inverse(X_test_num)

        X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
        X_train_num_ori, X_test_num_ori = torch.tensor(X_train_num_ori).float(), torch.tensor(X_test_num_ori).float()

        mean, std = X_train_num.mean(0), X_train_num.std(0)
        mean_ori, std_ori = X_train_num_ori.mean(0), X_train_num_ori.std(0)

        gt_test_num_ori = ((X_test_num_ori - mean_ori) / std_ori).numpy() # gt test num for eval
        x_miss_num = (X_test_num - mean) / std / 2  # masked num for impute

        x_miss_num = x_miss_num * (1-impute_mask[:, :n_num])

    else:
        X_train_num, X_test_num = None, None
        x_miss_num = None
        gt_test_num_ori = None

    if has_cat:
        X_train_cat, X_test_cat = X_cat
        
        X_train_cat = X_train_cat[:, :n_cat]
        X_test_cat = X_test_cat[:, :n_cat]
        categories = categories[:n_cat]

        X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

        gt_test_cat = X_test_cat.clone().numpy() # gt test cat for eval
        x_miss_cat = X_test_cat.clone() # masked cat for impute

        x_miss_cat = x_miss_cat * (1-impute_mask[:, -n_cat:])
    else:
        X_train_cat, X_test_cat = None, None
        x_miss_cat = None
        gt_test_cat = None

    rec_Xs_num = []
    rec_Xs_cat = []

    # posterior sampling
    with torch.no_grad():
        for trail in range(args.num_trail):
            print(f'Trail {trail}')
            n_cat_model = n_cat
                
            model = TabNAT(n_num=n_num, n_cat=n_cat_model, categories=categories, 
                    embed_dim = embed_dim, 
                    buffer_size = buffer_size, 
                    depth = depth, 
                    norm_layer = nn.LayerNorm, 
                    dropout_rate = args.dropout_rate,
                    device = device).to(device)

            model.load_state_dict(torch.load(f'checkpoints/{dataname}/model.pt'))

            model.eval()
            
            syn_num, syn_cat = model.impute(x_miss_num, x_miss_cat, impute_mask, one_step=one_step, device=device)
            
            if syn_num is not None:
                syn_num = (syn_num.cpu() * 2 * std) + mean
                syn_num = syn_num.numpy()
            
                syn_num = num_inverse(syn_num)
                
                # normalize again for evaluation
                syn_num = (syn_num - mean_ori.cpu().numpy()) / std_ori.cpu().numpy() 

                rec_Xs_num.append(syn_num)
            
            if syn_cat is not None:
                syn_cat = syn_cat.cpu().numpy()
                rec_Xs_cat.append(syn_cat)
    
    # Expectation A Posteriori
    if len(rec_Xs_num) > 0:
        pred_X_num =  np.stack(rec_Xs_num, axis = 0).mean(0) 
    else:
        pred_X_num = None

    # majority vote for categorical data
    if len(rec_Xs_cat) > 0:
        pred_X_cat = np.stack(rec_Xs_cat, axis = 0)
        pred_X_cat = stats.mode(pred_X_cat, axis=0, keepdims=False)[0]
    else:
        pred_X_cat = None

    mae, rmse, acc = get_eval(pred_X_num, gt_test_num_ori, pred_X_cat, gt_test_cat, impute_mask[:,:n_num], impute_mask[:,-n_cat:])

    print(f'EAP result: MAE: {mae}, RMSE: {rmse}, ACC: {acc}')

    for i in range(args.num_trail):
        if has_num:
            pred_X_num = rec_Xs_num[i]
        if has_cat:
            pred_X_cat = rec_Xs_cat[i]
            pred_X_cat = np.round(pred_X_cat)

        mae, rmse, acc = get_eval(pred_X_num, gt_test_num_ori, pred_X_cat, gt_test_cat, impute_mask[:,:n_num], impute_mask[:,-n_cat:])
        print(f'Sample {i} result: MAE: {mae}, RMSE: {rmse}, ACC: {acc}')

    


        
        