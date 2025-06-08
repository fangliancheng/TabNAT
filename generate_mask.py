import numpy as np
import json
import argparse
import os

from utils_train import preprocess, generate_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mask')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='Mask ratio.')
    args = parser.parse_args()

    data_dir = f'data/{args.dataname}'
    info_path = f'data/{args.dataname}/info.json'
    save_path = f'data/{args.dataname}/rate{int(100*args.mask_ratio)}'

    with open(info_path, 'r') as f:
        info = json.load(f)

    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(data_dir, task_type = info['task_type'], inverse = True, concat = True)
    n_cat = len(categories)

    B = X_cat[1].shape[0]
    
    mask = generate_mask(B, n_num+n_cat, args.mask_ratio)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(f'{save_path}/mask.npy', mask)
