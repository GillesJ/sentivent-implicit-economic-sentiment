#!/usr/bin/env python3
'''
Explain purpose of script here

format_berticon.py in sentivent-implicit-economic-sentiment
3/23/21 Copyright (c) Gilles Jacobs
'''
import pandas as pd

if __name__ == '__main__':

    # Preparing train data
    sentivent_fp = '../data/sentivent_implicit.csv'
    df = pd.read_csv(sentivent_fp, sep='\t')
    polarity_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    # make unique int ids
    df['id_orig'] = df['id']
    df['id'] = df.index.astype(int)
    # select text to clf
    df['text'] = df['polex+targets']
    # df_all = df_all.rename(columns={'polex': 'text'})
    df['label'] = df['polarity'].map(polarity_map)

    # split + write
    opt_fp_base = '../src/BERTICON/Data/sentivent-implicit_'
    for split in ['train', 'dev', 'test']:
        df_split = df[df['split'] == split]
        # only keep id, text, label
        df_split = df_split[['id', 'text', 'label']]
        # write
        opt_fp = opt_fp_base + f'{split}.tsv'
        df_split.to_csv(opt_fp, sep='\t', index=False)
        print(f'{opt_fp} split written with size {df_split.shape}.')