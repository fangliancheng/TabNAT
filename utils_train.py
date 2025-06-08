import numpy as np
import os

import src
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat
        self.num_only = X_cat is None 
        self.cat_only = X_num is None

    def __getitem__(self, index):
        if self.num_only:
            this_num = self.X_num[index]
            this_cat = None
            return this_num
        
        elif self.cat_only:
            this_num = None
            this_cat = self.X_cat[index]
            return this_cat
        else:
            this_num = self.X_num[index]
            this_cat = self.X_cat[index]

            return (this_num, this_cat)

    def __len__(self):
        if self.num_only:
            return self.X_num.shape[0]
        elif self.cat_only:
            return self.X_cat.shape[0]
        else:
            return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat
        
        d_numerical = 0
        categories = []
        num_inverse = None
        cat_inverse = None

        if X_num is not None:
            X_train_num, X_test_num = X_num['train'], X_num['test']
            X_num = (X_train_num, X_test_num)

            d_numerical = X_train_num.shape[1]

        if X_cat is not None:
            X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
            X_cat = (X_train_cat, X_test_cat)
        
            categories = src.get_categories(X_train_cat)

        if inverse:
            if X_num is not None:
                num_inverse = dataset.num_transform.inverse_transform
            if X_cat is not None:
                cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))
    
    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)


def generate_mask(bsz, seq_len, mask_ratio):
    mask = np.random.choice([0, 1], size=(bsz, seq_len), p=[1-mask_ratio, mask_ratio])
    print('generated mask ratio:', mask.sum() / mask.size)
    return torch.tensor(mask)

def generate_fix_mask(bsz, seq_len, mask_ratio):
    masks = np.zeros((bsz, seq_len))
    for i in range(bsz):
        #randomly select 5 columns
        cols = np.random.choice(seq_len, size=5, replace=False)
        masks[i, cols] = 1
    print('generated mask ratio:', masks.sum() / masks.size)
    return torch.tensor(masks, dtype=torch.int64)

def get_eval(num_pred, num_true, cat_pred, cat_true, num_mask, cat_mask):
        mae, rmse = 0.0, 0.0
        if num_pred is not None:
            mae = np.abs(num_pred[num_mask==1] - num_true[num_mask==1]).mean()
            rmse = np.sqrt(((num_pred[num_mask==1] - num_true[num_mask==1]) ** 2).mean())

        acc = 0.0
        if cat_true is not None:
            if cat_true[cat_mask].shape[0] > 0:
                # compute accuracy at cat_mask==1
                acc = (cat_pred[cat_mask==1] == cat_true[cat_mask==1]).sum() / cat_mask.sum()

        return mae, rmse, acc
