import os
import time
import json
import warnings
    
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils_train import preprocess, TabularDataset
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
    batch_size = args.batch_size
    data_dir = f'data/{dataname}'
    device =  args.device
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    
    X_num, X_cat, categories, n_num, _, _ = preprocess(data_dir, task_type=info['task_type'], inverse=True, concat=True)

    n_cat = len(categories)
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

    train_data = TabularDataset(X_train_num, X_train_cat)
    
    n_cat_model = n_cat
   
    model = TabNAT(n_num=n_num, 
                   n_cat=n_cat_model, 
                   categories=categories, 
                   embed_dim=embed_dim, 
                   buffer_size=buffer_size, 
                   depth = depth, 
                   norm_layer = nn.LayerNorm, 
                   dropout_rate = args.dropout_rate,
                   device = device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_loss = np.inf
    if not os.path.exists(f'checkpoints/{dataname}'):
        os.makedirs(f'checkpoints/{dataname}')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        epoch_loss_num = 0
        epoch_loss_cat = 0
        epoch_num = 0

        for batch in train_loader:
            if not has_cat:
                x_num = batch 
                x_cat = None
            elif not has_num:
                x_cat = batch
                x_num = None
            else:
                x_num, x_cat = batch

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

            batch_len = x_cat.shape[0] if has_cat else x_num.shape[0]

            epoch_loss += loss.item() * batch_len
            if has_cat:
                epoch_loss_cat += loss_cat.item() * batch_len
            if has_num:
                epoch_loss_num += loss_num.item() * batch_len

            epoch_num += batch_len
            # =================================

        scheduler.step(epoch_loss)
        epoch_loss /= epoch_num
        epoch_loss_cat /= epoch_num
        epoch_loss_num /= epoch_num

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{args.epochs}], Epoch Loss: {epoch_loss:.4f}, Epoch Loss Cat: {epoch_loss_cat:.4f}, Epoch Loss Num: {epoch_loss_num:.4f}, Best Loss: {best_loss:.4f}, patience: {patience}')

        if epoch % 1000 == 0:
           torch.save(model.state_dict(), f'checkpoints/{dataname}/model_{epoch}.pt')
    
    torch.save(model.state_dict(), f'checkpoints/{dataname}/model.pt')
    
    end_time = time.time()
        
    print(f'Training time: {end_time - start_time:.2f} seconds')
    
    print(f'Training finished. Best Loss: {best_loss}')