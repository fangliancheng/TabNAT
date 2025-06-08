import os
import json
import time
from baselines.ctgan.data import read_csv
from baselines.ctgan.models.tvae import TVAE


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    data_path = f'{curr_dir}/../../data/{dataname}/train.csv'
    info_path = f'{curr_dir}/../../data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    if info['task_type'] == 'regression':
        discrete = [info['column_names'][i] for i in info['cat_col_idx']]
    else:
        discrete = [info['column_names'][i] for i in (info['cat_col_idx'] + info['target_col_idx'])]
    
    ckpt_path = f'{curr_dir}/ckpt/{dataname}/TVAE'
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    data, discrete_columns = read_csv(data_path, discrete = discrete)
    
    start_time = time.time()
    model = TVAE()
    model.fit(data, discrete_columns)

    end_time = time.time()

    print(f'Training time = {end_time - start_time}')
    model.save(f'{ckpt_path}/model.pt')