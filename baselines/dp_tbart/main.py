import os
import time
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import KBinsDiscretizer

from utils_train import preprocess, TabularDataset
from baselines.dp_tbart.models.model import TabMAR

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
    batch_size = args.batch_size
    enable_padding = True if args.padding == 1 else False
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = f'{script_path}/../../data/{dataname}'
    device =  args.device
    info_path = f'{script_path}/../../data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    
    X_num, X_cat, categories, n_num, _, _ = preprocess(data_dir, task_type=info['task_type'], inverse=True, concat=True)

    X_cat = X_cat[0]
    
    if X_num is not None:
        X_num = X_num[0]
        # discretize numerical features
        enc = KBinsDiscretizer(n_bins=[100 for _ in range(n_num)], encode='ordinal', strategy='kmeans', subsample=None)
        X_num_disc = enc.fit_transform(X_num).astype(np.int64)
    
        X_cat = np.concatenate([X_num_disc, X_cat], axis=1)
        categories = [100 for _ in range(n_num)] + categories
        X_num = None
    
    n_cat = len(categories)
    n_num = 0

    X_train_num =  None
    X_train_cat = torch.tensor(X_cat)

    train_data = TabularDataset(X_train_num, X_train_cat)
    
    model = TabMAR(n_num, n_cat, categories, 
                embed_dim = embed_dim, 
                buffer_size = buffer_size, 
                depth = depth, 
                norm_layer = nn.LayerNorm, 
                dropout_rate = args.dropout_rate,
                padding = enable_padding,
                device = device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02, betas=(0.9, 0.95))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=200, verbose=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_loss = np.inf
    patience = 1000
    if not os.path.exists(f'TabART/checkpoints/{dataname}'):
        os.makedirs(f'TabART/checkpoints/{dataname}')
    
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        epoch_loss_num = 0
        epoch_loss_cat = 0
        epoch_num = 0

        for batch in train_loader:
            x_cat = batch
            x_num = None

            # =================================
            if n_num >= 1:
                x_num = x_num.to(device)
            else:
                x_num = None
            if n_cat >= 1:
                x_cat = x_cat.to(device)
            else:
                x_cat = None

            repeat = 1
        
            optimizer.zero_grad()

            if n_num >= 1:
                x_num = x_num.repeat(repeat, 1)
            if n_cat >= 1:
                x_cat = x_cat.repeat(repeat, 1)
            
            loss, loss_num, loss_cat = model(x_num, x_cat)

            loss.backward()
            optimizer.step()

            batch_len = x_cat.shape[0]

            epoch_loss += loss.item() * batch_len
            # epoch_loss_num += loss_num.item() * batch_len
            epoch_loss_cat += loss_cat.item() * batch_len

            epoch_num += batch_len

            # =================================

        scheduler.step(epoch_loss)
        epoch_loss /= epoch_num
        # epoch_loss_num /= epoch_num
        epoch_loss_cat /= epoch_num

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0

        else:
            patience += 1
        
        # if patience > 500:
        #     break
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Epoch Loss: {epoch_loss:.4f}, Epoch Loss Cat: {epoch_loss_cat:.4f}, Best Loss: {best_loss:.4f}, patience: {patience}')

        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f'TabART/checkpoints/{dataname}/model_{args.embed_dim}_{args.buffer_size}_{args.depth}_{args.padding}.pt')
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    print(f'Training finished. Best Loss: {best_loss}')