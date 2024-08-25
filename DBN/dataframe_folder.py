#!/usr/bin/env python
# coding: utf-8

import warnings
import pandas as pd
import numpy as np
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import pickle
import re

df = pd.read_csv('./cycle_dataset.csv')


with open('./cycle_datetimes', 'rb') as cdts:
    cycle_datetimes = pickle.load(cdts)


deleted_features = ['XXXXXXXXX']
translation_dict = {
    'XXXXXXXXXXXXX' 
}

def get_train_df(n_cycles):
    n_cycles = int(n_cycles)
    chunk_list = []

    for cycle_idx in list(cycle_datetimes.keys())[:n_cycles]:
        df_slice = df[df.columns[df.columns.str.startswith(f'{cycle_idx}_')].tolist()]
        df_slice = df_slice[(~df_slice[df_slice.columns[0]].isna())]
        df_slice.columns =[re.split(r'c\d+_', x)[1] for x in df_slice.columns]

        df_slice = df_slice[[x for x in df_slice.columns if x not in deleted_features]]
        df_slice.columns =[translation_dict[x] for x in df_slice.columns]
            
        chunk_list.append(df_slice)

    straighted_df = pd.concat(chunk_list, axis=0)
    straighted_df = straighted_df.reset_index(drop=True)
    del chunk_list
    return straighted_df

def fold_dataframe(size, n_cycles):
    size = int(size)
    n_cycles = int(n_cycles)
    chunk_list = []
    for cycle_idx in list(cycle_datetimes.keys())[:n_cycles]:
        df_slice = df[df.columns[df.columns.str.startswith(f'{cycle_idx}_')].tolist()]
        df_slice.columns =[re.split(r'c\d+_', x)[1] for x in df_slice.columns]

        df_slice = df_slice[[x for x in df_slice.columns if x not in deleted_features]]
        chunkie_list = []
        for window_size in range(0, size):
            lag = df_slice[(~df_slice[df_slice.columns[0]].isna())].shift(window_size)
            lag.columns =[translation_dict[x] + f'_t_{window_size}' for x in lag.columns]
            chunkie_list.append(lag)

        cycle_lag_df = pd.concat(chunkie_list, axis=1).iloc[window_size:]
        del chunkie_list
        chunk_list.append(cycle_lag_df)

    folded_df = pd.concat(chunk_list, axis=0)

    folded_df = folded_df.reset_index(drop=True)
    del chunk_list
    
    return folded_df

