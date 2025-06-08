from baselines.ctgan.data import read_csv
from baselines.ctgan.models.ctgan import CTGAN
import os
import json
import time

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

    ckpt_path = f'{curr_dir}/ckpt/{dataname}/CTGAN'

    data, discrete_columns = read_csv(data_path, discrete = discrete)

    model = CTGAN.load(f'{ckpt_path}/model.pt')
    print(f'Loading saved model from {ckpt_path}/model.pt')
 
    num_samples = len(data)

    start_time = time.time()
    sampled = model.sample(
        num_samples,
        args.sample_condition_column,
        args.sample_condition_column_value)

    sampled.to_csv(save_path, index = False)

    end_time = time.time()
    print(f'Sampling time = {end_time - start_time}')

    print('Saving sampled data to {}'.format(save_path))