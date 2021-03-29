#!/usr/bin/env python3
'''
Join main Sentivent with SentiFM dataset.
`sentivent-plus-sentifm.csv` maintains original train-test-dev split labels for instances.
`sentivent-plus-sentifm-as-train.csv` labels all SentiFM as train and only uses Sentivent dev and test.

join_sets.py in sentivent-implicit-economic-sentiment
3/16/21 Copyright (c) Gilles Jacobs
'''
import pandas as pd

if __name__ == '__main__':

    sentivent_fp = '../data/sentivent_implicit.csv'
    sentifm_fp = '../data/sentifm-fine-en.csv'

    df_vent = pd.read_csv(sentivent_fp, sep='\t')
    df_fm = pd.read_csv(sentifm_fp, sep='\t')

    # concat
    df_all = pd.concat([df_vent, df_fm])
    df_fm['split'] = 'train' # make SentiFM train
    df_all_fmtrain = pd.concat([df_vent, df_fm])

    # write
    df_all.to_csv('../data/sentivent-plus-sentifm.csv', sep='\t', index=False)
    df_all_fmtrain.to_csv('../data/sentivent-plus-sentifm-as-train.csv', sep='\t', index=False)