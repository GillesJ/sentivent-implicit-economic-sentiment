#!/usr/bin/env python3
'''
Determine good max seq length

determine_max_seq_length.py in sentivent-implicit-economic-sentiment
4/15/21 Copyright (c) Gilles Jacobs
'''
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from scipy import stats
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/gilles/.cache/huggingface/transformers/'

# data_fp = '../data/sentivent-plus-sentifm-as-train.csv'
# data_fp = '../data/sentivent_implicit.csv' # gold-polarexpr experiments
data_fp = '../data/sentivent_implicit_clauses.csv'# clause experiments
# text_col = 'polex+targets' # gold-polarexpr experiments
text_col = 'clause_text' # clause experiments
df_all = pd.read_csv(data_fp, sep='\t')
df_all = df_all.rename(columns={text_col: 'text'}) # polex+target = input
# df_all = df_all.rename(columns={'polex': 'text'})
df_dev = df_all[(df_all['split'] == 'train') | (df_all['split'] == 'dev')] # split off devset

# tokenizer length settings:
for model_name in ['roberta-base', 'roberta-large', 'bert-large-cased']:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(df_dev['text'].to_list()).input_ids
    lens = np.array([len(t) for t in tokens])
    print(f'{model_name.upper()} seq. len max:\t{lens.max()}\tmin:\t{lens.min()}\tmean: {lens.mean()}\tmedian;\t{np.median(lens)}')
    for p in [99, 98, 95, 90]:
        p_len = np.percentile(lens, p)
        print(f'{int(p_len)} max_seq_length will not truncate {p}% of input text.')

    for l in [16, 24, 32, 64, 128, 256, 512]:
        ps = stats.percentileofscore(lens, l)
        print(f"{l} max_seq_len will not truncate {ps}% of input text.")
        if ps == 100:
            break
    print('-------------------------------------------------------')
pass
