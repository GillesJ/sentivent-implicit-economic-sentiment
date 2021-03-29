#!/usr/bin/env python3
'''
Get sentiment lexicon features from text.
Loads and applies scoring for several sentiment lexicons.

- Input: tsv with pre-tokenized space-split text column + lexicons.tsv with a lex-key column
- Output: tsv with word-level and sequence-level matches.

lexicon.py in sentivent-implicit-economic-sentiment
3/23/21 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from itertools import groupby, combinations
from sklearn.preprocessing import MinMaxScaler

lex_dir = Path('../lexicons')
UNILEX_FPS = { # paths to single-token (i.e. no MWE, or n>1gram matching)
    'henry': 'henry2008.tsv',
    'lm': 'loughranmcdonald-v2019.tsv',
    'ntusd': 'ntusdfinword-v1.tsv'
}
UNILEX_FPS = {k: lex_dir / v for k,v in UNILEX_FPS.items()}

def discretize_polarity(v):
    if v > 0.5:
        return 1
    elif -0.5 <= v <= 0.5:
        return 0
    else:
        return -1

def match_unigrams(tokens, wordlist):

    return [wordlist.get(t, False) for t in tokens]

def add_unilex(texts, lexicons):

    df_res = texts.to_frame()
    for name_lex, df_lex in lexicons.items():
        if 'lex-uncased' in df_lex.columns: # set uncased flag
            txt = texts.str.lower()
        else:
            txt = texts
        txt = txt.str.split() # text is pretokenized 'tok1 tok2'
        key_col = [c for c in df_lex.columns if 'lex-' in c][0] # key token column starts with lex- by convention formatting
        wordlist_cols = [c for c in df_lex.columns if 'lex-' not in c]
        for c in wordlist_cols:
            feat_name = name_lex + '_' + c
            wordlist_dict = dict(zip(df_lex[key_col], df_lex[c]))

            df_word_match = txt.apply(match_unigrams, args=(wordlist_dict,))
            df_res[feat_name] = df_word_match

    return df_res

def sum_norm_wordlist_score(word_match):
    # sum of polarity matched tokens
    name = word_match.name
    seq_sum = word_match.apply(sum).rename(name + '-seqsum')
    seq_normlen = word_match.apply(lambda x: sum(x) / len(x)).rename(name + '-seqnorm')

    match_only = word_match.apply(lambda x: [i for i in x if i !=False])
    match_norm = match_only.apply(lambda x: sum(x) / len(x) if len(x) else 0).rename(name + '-matchnorm')

    return pd.concat([seq_sum, seq_normlen, match_norm], axis=1)

def process_henry(word_match):
    return sum_norm_wordlist_score(word_match)


def process_ntusd(word_match):
    return sum_norm_wordlist_score(word_match)

def process_lm(df_match):
    df_feats = pd.DataFrame()
    for list_name, list_match in df_match.items():
    # every 0 value in LM is no-match (artifact of wordlist) -> should be converted to False.
        list_match = list_match.apply(lambda x: [i if i != 0 else False for i in x])
        df_feats = pd.concat([df_feats, sum_norm_wordlist_score(list_match)], axis=1)

    # LM specifies uncertainty, litiguous, constraining list with negative connotation, join these together
    neg_cols_combo = ['negative', 'uncertainty', 'litigious', 'constraining']
    metric_key = lambda x: x.split('-')[-1]
    for n, cols in groupby(sorted([c for c in df_feats if c.split('_')[1].split('-')[0] in neg_cols_combo],
                             key=metric_key), metric_key):
        df_feats['lm_negativecombo-'+n] = df_feats[list(cols)].sum(axis=1)

    # make a single polarity column polarity = pos - neg sentiment words
    pos_cols = sorted([c for c in df_feats.columns if 'positive' in c])
    neg_cols = sorted([c for c in df_feats.columns if c.split('-')[0] == 'lm_negative'])
    neg_cols_combo = sorted([c for c in df_feats.columns if c.split('-')[0] == 'lm_negativecombo'])
    for pos, neg in zip(pos_cols, neg_cols):
        polarity_n = 'lm_polarity-' + pos.split('-')[-1]
        df_feats[polarity_n] = df_feats[pos] - df_feats[neg]
    for pos, neg in zip(pos_cols, neg_cols_combo):
        polarity_n = 'lm_polaritycombo-' + pos.split('-')[-1]
        df_feats[polarity_n] = df_feats[pos] - df_feats[neg]

    return df_feats

def scale_minmax(df, feature_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def save_object(obj, fpath):
	"""
	Pickle an object and save it to file
	"""
	with open(fpath, 'wb') as o:
		pickle.dump(obj, o)

def load_object(fpath):
	"""
	Load a pickled object from file
	"""
	with open(fpath, 'rb') as i:
		return pickle.load(i)

if __name__ == '__main__':

    # load base data for adding lexicon features
    sentivent_fp = '../data/sentivent_implicit.csv'
    # sentivent_fp = './BERTICON/Data/sentivent-implicit_train.tsv'
    df = pd.read_csv(sentivent_fp, sep='\t', quoting=3) # quoting NEEDS TO BE TURNED OFF with 3
    text_col = 'polex+targets'
    # text_col = 'text'

    # load lexicons
    unilex_dfs = {k: pd.read_csv(v, sep='\t') for k, v in UNILEX_FPS.items()}

    # get token-wise lexicon-values for single token wordlist, False = no match
    df_match = add_unilex(df[text_col], unilex_dfs)

    # process matches
    henry_feats = process_henry(df_match['henry_polarity'])
    # henry_debug = pd.concat([df[text_col], df_match['henry_polarity'], henry_feats, df['polarity']], axis=1)
    ntusd_feats = process_ntusd(df_match['ntusd_market_sentiment'])
    # ntusd_debug = pd.concat([df[text_col], df_match['ntusd_market_sentiment'], ntusd_feats, df['polarity']], axis=1)
    lm_feats = process_lm(df_match[[c for c in df_match.columns if 'lm_' in c]])
    # lm_debug = pd.concat([df[text_col], lm_feats, df['polarity']], axis=1)
    # matchnorm corresponds worse to gold-standard polarity than lennorm -> don't use it

    df_feats = pd.concat([henry_feats, ntusd_feats, lm_feats], axis=1)

    norm_methods = set(n.split('-')[-1] for n in df_feats.columns)
    for norm_method in norm_methods:
        df_pol_norm = df_feats.filter(regex=f'(polarity|market_sentiment)-{norm_method}')
        df_pol_norm = scale_minmax(df_pol_norm, feature_range=(-1,1))
        df_feats[f'all_polarity-{norm_method}sum'] = df_pol_norm.sum(axis=1)
        df_feats[f'all_polarity-{norm_method}mean'] = df_pol_norm.mean(axis=1)

    # investigate correlation of features
    # normalization
    # invert neg cols
    neg_cats = ['negative', 'negativecombo', 'uncertainty', 'litigious', 'constraining']
    neg_cols = [c for c in df_feats.columns if c.split('_')[-1].split('-')[0] in neg_cats]
    df_feats[neg_cols] = -df_feats[neg_cols]
    # discretize polarity
    df_discr = df_feats.applymap(discretize_polarity)
    # minmaxscale polarity scores, this will have neutral bias
    df_scale = scale_minmax(df_feats.filter(like='polarity'), feature_range=(-1,1))

    # test correlation with gold polarity
    # df_goldpol = df['polarity'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    df_goldpol = df['polarity'].map({0: 1, 1: -1, 2: 0})
    corr = df_feats.corrwith(df_goldpol, method='spearman')
    corr_scaled_pol = df_scale.corrwith(df_goldpol, method='spearman')
    corr_discr = df_discr.corrwith(df_goldpol, method='spearman')
    # simple > scaled > discretized
    # manual review of corrs: seqnorm > seqsum > matchnorm superior => do not use matchnorm
    # do not use lm_interesting, lm_constraining, lm_modal, lm_litigious, lm_positive
    # best is all_polarity-seqnormmean = all_polarity-seqnormsum (combination of all 3 economic is best correlation)
    df_feats_keep = df_feats.filter(
        regex='(positive|negative(combo)?|market_sentiment|polarity(combo)?)-(seqsum|seqnorm)(mean)?')
    # df_debug = pd.concat([df_feats_keep, df['polarity']], axis=1)

    # pickle featdict like original pipeline
    sentiment_feat_dict = {text: feat for text, feat in zip(df[text_col].to_list(), df_feats_keep.to_numpy())}
    save_object(sentiment_feat_dict, './BERTICON/feat_dict_sentiment.pkl')
    loaded_f = load_object('./BERTICON/feat_dict_sentiment.pkl')
    model_key = '"AMD said they would "" lay the foundation for growth and profitability"'
    loaded_f[model_key]
    pass