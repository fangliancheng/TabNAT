import os
import json
import time
from baselines.ctgan.data import read_csv
from baselines.ctgan.models.tvae import TVAE


def main(args):
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f'data/{dataname}/train.csv'
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    if info['task_type'] == 'regression':
        discrete = [info['column_names'][i] for i in info['cat_col_idx']]
    else:
        discrete = [info['column_names'][i] for i in (info['cat_col_idx'] + info['target_col_idx'])]


    syn_path = f'synthetic/{dataname}'
    ckpt_path = f'{curr_dir}/ckpt/{dataname}/TVAE'
    
    if not os.path.exists(syn_path):
        os.makedirs(syn_path)
    data, discrete_columns = read_csv(data_path, discrete = discrete)

    model = TVAE.load(f'{ckpt_path}/model.pt')
    print(f'Loading saved model from {ckpt_path}/model.pt')
 
    start_time = time.time()
    num_samples = len(data)

    sampled = model.sample(num_samples)
    sampled.to_csv(save_path, index = False)

    end_time = time.time()
    print(f'Sampling time = {end_time - start_time}')

    print('Saving sampled data to {}'.format(save_path))