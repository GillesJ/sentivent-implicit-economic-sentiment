#!/usr/bin/env python3
'''
Get sentiment lexicon features from text.
Loads and applies scoring for several sentiment lexicons.

- Input: tsv with pre-tokenized space-split text column + lexicons.tsv with a lex-key column
- Output: tsv with word-level and sequence-level matches.

SPECIAL CASE = SentiEcon lexicon which is slow, to speed up: precompute and store features using ../util/lingmotif_prep.py

featurize_lexicon.py in sentivent-implicit-economic-sentiment
3/23/21 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import json
from itertools import groupby, combinations
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sentiwordnet import SentiwordnetAnalysis
import spacy

lex_dir = Path('../lexicons')
UNILEX_FPS = { # paths to single-token (i.e. no MWE, or n>1gram matching)
    'henry': 'henry2008.tsv',
    'lm': 'loughranmcdonald-v2019.tsv',
    'ntusd': 'ntusdfinword-v1.tsv', # no lemmatizing needed direct token match ok
}
UNILEX_FPS = {k: lex_dir / v for k,v in UNILEX_FPS.items()}

FEAT_FILTERS = [ # 2-tup (name, regex Dataframe column filter
    ('posneg-pol-seqsumseqnorm', '(positive|negative(combo)?|market_sentiment|polarity(combo)?)-(seqsum|seqnorm)(mean)?'), # exp2
    ('posneg-pol-all', '(positive|negative(combo)?|market_sentiment|polarity(combo)?)'),
    ('pol-all', '(market_sentiment|polarity(combo)?)'),
    ('pol-all-seqsumseqnorm', '(market_sentiment|polarity(combo)?)-(seqsum|seqnorm)(mean)?'),
    ('no-lexicon-features', '^\b$'), # won't match any column
    ('all-lexicon-features', '.*'), # will match every column
]

# setup spacy nlp pipeline for our pre-tokenized and presentence-split text
@spacy.Language.component('prevent-sbd')
def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection -> our data was manually sentence-split
        token.is_sent_start = False
    return doc

def pretokenized_split(text):
    # this will use simple split on presplit text
    return spacy.tokens.Doc(nlp.vocab, text.split())

nlp = spacy.load("en_core_web_trf", disable=['ner'])
nlp.tokenizer = pretokenized_split
nlp.add_pipe('prevent-sbd', before="parser")

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

    match_only = word_match.apply(lambda x: [i for i in x if i != False])
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

def build_trie(lexicon, include_only=False):
    """
    Build a character-trie from the plain pattern_string -> categories_list
    mapping provided by `lexicon`.
    Some LIWC patterns end with a `*` to indicate a wildcard match.

    """
    trie = {}
    for pattern, category_names in lexicon.items():
        if include_only:
            category_names = [c for c in category_names if c in include_only]
        if category_names:
            cursor = trie
            for char in pattern:
                if char == "*":
                    cursor["*"] = category_names
                    break
                if char not in cursor:
                    cursor[char] = {}
                cursor = cursor[char]
            cursor["$"] = category_names
    return trie

def search_trie(trie, token, token_i=0):
    """
    Search the given character-trie for paths that match the `token` string.
    """
    if "*" in trie:
        return trie["*"]
    if "$" in trie and token_i == len(token):
        return trie["$"]
    if token_i < len(token):
        char = token[token_i]
        if char in trie:
            return search_trie(trie[char], token, token_i + 1)
    return []

def process_liwc(df_text, liwc_fp='../lexicons/liwc2007_en.json'):

    cats_keep = ['Anger', 'Anx', 'Achiev', 'Affect', 'Cause', 'Certain', 'CogMech', 'Discrep', 'Excl',
             'Insight', 'Money', 'Negemo', 'Negate', 'Posemo', 'Relativ', 'Sad', 'Tentat', 'Quant', 'Work']
    # cats_keep = False
    with open(liwc_fp, 'rt') as liwc_dict_in:
        liwc_dict = json.load(liwc_dict_in)
    trie = build_trie(liwc_dict, include_only=cats_keep)

    liwc_matches = []
    lens = df_text.str.split().apply(len)
    for tokens in df_text.str.split():
        liwc_c = dict(Counter('liwc_'+cat.lower()+'-seqsum' for t in tokens for cat in search_trie(trie, t)))
        liwc_matches.append(liwc_c)

    df_feats = pd.DataFrame(liwc_matches).fillna(0)
    for list_name, matches in df_feats.items():
        df_feats[list_name.replace('-seqsum', '-seqnorm')] = matches / lens

    # make matchnorm for PosEma and NegEmo, we do not match across all Liwc categories
    # because that would dilute polarity which is only based on these columns
    n_posneg_matches = df_feats[['liwc_posemo-seqsum', 'liwc_negemo-seqsum']].sum(axis=1)
    df_feats['liwc_posemo-matchnorm'] = (df_feats['liwc_posemo-seqsum'] / n_posneg_matches).fillna(0)
    df_feats['liwc_negemo-matchnorm'] = (df_feats['liwc_negemo-seqsum'] / n_posneg_matches).fillna(0)

    pos_cols = sorted([c for c in df_feats.columns if 'posemo' in c])
    neg_cols = sorted([c for c in df_feats.columns if 'negemo' in c])
    for pos, neg in zip(pos_cols, neg_cols):
        df_feats[f'liwc_polarity-{pos.split("-")[-1]}'] = df_feats[pos] - df_feats[neg]

    return df_feats

def scale_polarity(df, test_idc):
    # fit on devset then transform to whole set to prevent test leakage
    feature_range = (-1,1) # for polarity between -1 and 1
    scaler = MinMaxScaler(feature_range=feature_range)
    # clip outliers to {05-95} quantile and scale
    df_dev = df[~df.index.isin(test_idc)]
    lower = df_dev.apply(lambda x: np.quantile(x[x<0], 0.05), axis=0) # set lower bound on negatives
    upper = df_dev.apply(lambda x: np.quantile(x[x>0], 0.95), axis=0) # upper clip on positives
    df_dev_clipped = df_dev.clip(lower=lower, upper=upper, axis=1)
    df_clipped = df.clip(lower=lower, upper=upper, axis=1)
    scaler = scaler.fit(df_dev_clipped)
    df_scaled = pd.DataFrame(scaler.transform(df_clipped), columns=df.columns, index=df.index)
    return df_scaled
        

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

def process_sentiwordnet(df_text, swn_fp='../lexicons/SentiWordNet_3.0.0.txt'):

    df_feats = pd.DataFrame()
    df_toks = df_text.str.split()
    # geometric = w2 in "Guerini et al 2013: Sentiment Analysis: How to Derive Prior Polarities from SWN"
    # w2 in paper was best for clf' available: ['geometric', 'average', 'harmonic']
    sense_weighting = ['geometric']
    for w in sense_weighting:
        swn3 = SentiwordnetAnalysis(filename=swn_fp, weighting=w)
        df_match = df_toks.map(swn3.score_tokens) # get lexicon scores for words, we do not retrieve neg pos scores because of the multiple senses
        df_match.name = f'swn3{w}_polarity'
        df_pol = sum_norm_wordlist_score(df_match) # compute polarity
        df_feats = pd.concat([df_feats, df_pol], axis=1)
    return df_feats

def make_phrase_matcher(df_lexicon):

    df_lexicon['lex'] = df_lexicon['lex'].str.replace('_', ' ')
    mwes = defaultdict(list)
    for pol, mwe in zip(df_lexicon.pol, df_lexicon.lex):
        mwes[pol].append(nlp.make_doc(mwe))
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr='LOWER')
    for label, patterns in mwes.items():
        matcher.add(label, patterns)
    return matcher


def process_msol(df_text, msol_fp):

    df_msol = pd.read_csv(msol_fp, sep=' ', header=None, names=['lex', 'pol'], converters={0: str})
    # converter is needed so literal string 'null' is not turned in float
    matcher = make_phrase_matcher(df_msol)
    df_match = df_text.apply(lambda x: dict(Counter(nlp.vocab.strings[m_id]
                                                    for m_id, _, _, in matcher(nlp.make_doc(x)))))
    df_feats = pd.DataFrame(df_match.to_list()).rename(
        columns={'positive': 'msol_negative-seqsum', 'negative': 'msol_positive-seqsum'}).fillna(0)

    # normalize
    lens = df_text.str.split().apply(len)
    matches = df_feats.sum(axis=1)
    for c in df_feats:
        c_stem = c.replace('-seqsum', '')
        df_feats[f'{c_stem}-seqnorm'] = df_feats[c] / lens
        df_feats[f'{c_stem}-matchnorm'] = df_feats[c] / matches
    df_feats = df_feats.fillna(0)

    pos_cols = sorted([c for c in df_feats if 'positive' in c])
    neg_cols = sorted([c for c in df_feats if 'negative' in c])
    for pos, neg in zip(pos_cols, neg_cols):
        norm_method = pos.split('-')[-1]
        df_feats[f'msol_polarity-{norm_method}'] = df_feats[pos] - df_feats[neg]

    return df_feats

if __name__ == '__main__':

    # load base data for adding lexicon features
    sentivent_fp = '../data/sentivent_implicit.csv'
    # sentivent_fp = './BERTICON/Data/sentivent-implicit_train.tsv'
    df = pd.read_csv(sentivent_fp, sep='\t', quoting=3) # quoting NEEDS TO BE TURNED OFF with 3
    text_col = 'polex+targets'
    # text_col = 'text'

    # A. Non single token lexicons: need individual approach with mwe/lemma/pos/synset matching + prior-to-posterior polarity calc
    # MSOL large-scale general domain
    msol_feats = process_msol(df[text_col], '../lexicons/MSOL-June15-09.txt')
    # LIWC (needs lemmatization)
    liwc_feats = process_liwc(df[text_col], liwc_fp='../lexicons/liwc2007_en.json')
    # Sentiecon Lexicon is preprocess with ./lingmotig_prep
    secon_feats = pd.read_csv('../util/sentiecon_feats.csv')
    # Sentiwordnet3 general domain
    swn3_feats = process_sentiwordnet(df[text_col], swn_fp='../lexicons/SentiWordNet_3.0.0.txt')

    # B. WORDLIST-BASED LEXICONS: match token-wise lexicon-values for single token wordlist (False = no match)
    unilex_dfs = {k: pd.read_csv(v, sep='\t') for k, v in UNILEX_FPS.items()} # load lexicons
    df_match = add_unilex(df[text_col], unilex_dfs) # then compute scores over matches
    # process matches
    henry_feats = process_henry(df_match['henry_polarity'])
    # henry_debug = pd.concat([df[text_col], df_match['henry_polarity'], henry_feats, df['polarity']], axis=1)
    ntusd_feats = process_ntusd(df_match['ntusd_market_sentiment'])
    # ntusd_debug = pd.concat([df[text_col], df_match['ntusd_market_sentiment'], ntusd_feats, df['polarity']], axis=1)
    lm_feats = process_lm(df_match[[c for c in df_match.columns if 'lm_' in c]])
    # lm_debug = pd.concat([df[text_col], lm_feats, df['polarity']], axis=1)
    # matchnorm corresponds worse to gold-standard polarity than lennorm -> don't use it

    # C. JOIN THEM ALL
    df_feats = pd.concat([henry_feats, ntusd_feats, lm_feats, liwc_feats, swn3_feats, secon_feats, msol_feats], axis=1)
    # combine all polarity features into one allcombo-combined polarity and economic in econcombo
    norm_methods = set(n.split('-')[-1] for n in df_feats.columns)
    for norm_method in norm_methods:
        df_pol_norm = df_feats.filter(regex=f'(polarity|market_sentiment)-{norm_method}')
        df_pol_scaled = scale_polarity(df_pol_norm, df[df['split']=='test'].index)
        df_feats[f'allcombo_polarity-{norm_method}sum'] = df_pol_scaled.sum(axis=1)
        df_feats[f'allcombo_polarity-{norm_method}mean'] = df_pol_scaled.mean(axis=1)
        df_pol_econ = df_feats.filter(regex=f'(ntusd|lm|secon|henry)_(polarity|market_sentiment)-{norm_method}')
        df_pol_econ_scaled = scale_polarity(df_pol_econ, df[df['split']=='test'].index)
        df_feats[f'econcombo_polarity-{norm_method}sum'] = df_pol_econ_scaled.sum(axis=1)
        df_feats[f'econcombo_polarity-{norm_method}mean'] = df_pol_econ_scaled.mean(axis=1)
    # sanity check polarity combination against label
    df_all_pol_debug = pd.concat([df['polarity'], df_feats.filter(regex=f'(all|combo)_polarity')], axis=1)

    # # simple > scaled > discretized
    # # manual review of corrs: seqnorm > seqsum > matchnorm superior => do not use matchnorm
    # # do not use lm_interesting, lm_constraining, lm_modal, lm_litigious, lm_positive
    # # best is all_polarity-seqnormmean = all_polarity-seqnormsum (combination of all 3 economic is best correlation)
    df_feats.to_csv('sentivent-implicit-lexiconfeats-all.csv', index=False)
    df_polarity_only = df_feats.filter(regex='(polarity|market_sentiment)')
    df_polarity_only.to_csv('sentivent-implicit-lexiconfeats-polarities.csv', index=False)

    # Write pickled DICTS for BERTICON pipeline
    # dirp_feat_dict = Path('../src/BERTICON')
    # for filt_name, filt_regex in FEAT_FILTERS:
    #     df_feats_filt = df_feats.filter(regex=filt_regex)
    #     print(f'Selected {filt_name} {df_feats_filt.columns.to_list()} with re: \'{filt_regex}\'')
    #     print(f'Lexicon feat dims:\t{df_feats_filt.shape[1]}')
    #     # pickle featdict like original pipeline
    #     fp_feat_dict = dirp_feat_dict / f'lexfeatdict_{filt_name}.pkl'
    #     feat_dict = {text: feat for text, feat in zip(df[text_col].to_list(), df_feats_filt.to_numpy())}
    #     save_object(feat_dict, fp_feat_dict)
    #     print(f'Pickled featdict to {fp_feat_dict}')
    #     # load test
    #     loaded_f = load_object(fp_feat_dict)
    #     model_key = '"AMD said they would "" lay the foundation for growth and profitability"'
    #     loaded_f[model_key]
    #     # df_debug = pd.concat([df_feats_keep, df['polarity']], axis=1)


    # 3. Investigate correlation of features
    # normalization
    # invert neg cols
    neg_cats = ['negative', 'negativecombo', 'uncertainty', 'litigious', 'constraining', # lm + nstud +
                'anger', 'anx', 'death', 'inhib', 'negate', 'negemo', 'sad' # liwc
                ]
    neg_cols = [c for c in df_feats.columns if c.split('_')[-1].split('-')[0] in neg_cats]
    df_feats[neg_cols] = -df_feats[neg_cols]
    # discretize polarity
    df_discr = df_feats.applymap(discretize_polarity)

    # test correlation with gold polarity
    df_goldpol = df['polarity'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    # df_goldpol = df['polarity'].map({0: 1, 1: -1, 2: 0})
    corr = df_feats.corrwith(df_goldpol, method='spearman').abs()
    corr_discr = df_discr.corrwith(df_goldpol, method='spearman').abs()
    pass
