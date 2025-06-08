import os
import json
import time
from baselines.ctgan.data import read_csv
from baselines.ctgan.models.ctgan import CTGAN


def main(args):
    dataname = args.dataname
    
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
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    data, discrete_columns = read_csv(data_path, discrete = discrete)

    generator_dim = [int(x) for x in args.generator_dim.split(',')]
    discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
    
    start_time = time.time()
    model = CTGAN(
        embedding_dim=args.embedding_dim, generator_dim=generator_dim,
        discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
        generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
        discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
        epochs=args.epochs)
    model.fit(data, discrete_columns)

    end_time = time.time()
    print(f'Training time = {end_time - start_time}')

    model.save(f'{ckpt_path}/model.pt')