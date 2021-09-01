#!/usr/bin/env python3
'''
Explain purpose of script here

lingmotif_prep.py in sentivent-implicit-economic-sentiment
4/20/21 Copyright (c) Gilles Jacobs
'''
import multiprocessing

import pandas as pd
import spacy

from ast import literal_eval
from spacy.util import filter_spans
from spacy.matcher import Matcher
from spacy import Language
from spacy.tokens import Doc
from spacy.symbols import ADJ, NOUN, VERB
from pathlib import Path
import numpy as np

from collections import defaultdict
# setup spacy nlp pipeline for our pre-tokenized and presentence-split text
@Language.component('prevent-sbd')
def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection -> our data was manually sentence-split
        token.is_sent_start = False
    return doc

def pretokenized_split(text):
    # this will use simple split on presplit text
    return Doc(nlp.vocab, text.split())

nlp = spacy.load("en_core_web_trf", disable=['ner'])
nlp.tokenizer = pretokenized_split
nlp.add_pipe('prevent-sbd', before="parser")

lemma_rule_lookup = nlp.get_pipe('lemmatizer').lookups.get_table("lemma_rules")

def lemma_pos_lookup(word, pos):
    token = spacy.tokens.Doc(nlp.vocab, words=[word])[0]
    pos_orig = spacy.tokens.Doc(nlp.vocab, words=[word])[0].pos_
    if pos == 'vb':
        token.pos_ = 'VERB'
        token.pos = VERB
    elif pos == 'nn':
        token.pos_ = 'NOUN'
        token.pos = NOUN
    elif pos == 'jj':
        token.pos_ = 'ADJ'
        token.pos = ADJ
    else:
        token.pos_ = ''
        token.pos = 0

    lemmas_lookup = nlp.get_pipe('lemmatizer').lookup_lemmatize(token) # use lookup table for language
    lemmas_rule = nlp.get_pipe('lemmatizer').rule_lemmatize(token) # use rule-based lemmatizer
    lemmas = list(set(lemmas_lookup + lemmas_rule))

    if len(lemmas) == 1:
        return lemmas[0]
    else:
        return {'IN': lemmas}

def lex_to_patterns(lex_entry):

    mwe, pos, pol, intens = lex_entry[0], lex_entry[1], lex_entry[2], lex_entry[3]
    parts = mwe.split('_')
    parts_without_wildcard = [p for p in parts if not p.isdigit()]
    parts_without_wildcard_idc = [i for i, p in enumerate(parts) if not p.isdigit()]
    pattern = []
    pattern_text = []
    pattern_lemma = []
    # if len(parts) > 1: # ignore pos for now
    # if len(parts_without_wildcard) > 1: # if parts is MWE: ignore the POS
    # # these are not derivable or correct in Sentiecon, head of const. does not correspond.
    #     toks = nlp(' '.join(parts_without_wildcard))
    #     lemmas = [t.lemma_ for t in toks]
    #     pos_spacy = [t.pos_ for t in toks]
    #     pass
    # else:
    for p in parts:
        if p[0] == '<': # <lemma> match
            p = p[1:-1] # strip off lemma tags
            pattern.append({'LEMMA': lemma_pos_lookup(p, pos)}) # this is for adapting the lexicon lemmatization to our model
            pattern_text.append({'LOWER': p})
            pattern_lemma.append({'LEMMA': lemma_pos_lookup(p, pos)})
        elif p.isdigit():
            wildcards = [{"OP": "*"}]
            pattern.extend(wildcards)
            pattern_text.extend(wildcards)
            pattern_lemma.extend(wildcards)
        else: # word or multiword phrase
            pattern.append({'LOWER': p})
            pattern_text.append({'LOWER': p})
            pattern_lemma.append({'LEMMA': lemma_pos_lookup(p, pos)})
    patterns = [pattern, pattern_lemma]
    if pattern_text not in patterns:
        patterns.append(pattern_text)
    match = (f'{mwe}:{pos}:{pol}:{intens}', patterns)
    return match

# def lex_to_patterns_lemma(lex_entry):
#     mwe, pos, pol, intens = lex_entry[0], lex_entry[1], lex_entry[2], lex_entry[3]
#     parts = mwe.split('_')
#     pattern = []
#     # if len(parts) > 1: # ignore pos for now
#     for p in parts:
#         if p.isdigit():
#             i = int(p)
#             pattern.extend([{}] * i)
#         else:
#             if p[0] == '<': # <lemma> match
#                 p = p[1:-1] # fast strip of tags
#             pattern.append({'LEMMA': lemma_pos_lookup(p, pos)})
#
#     match = (f'{mwe}:{pos}:{pol}:{intens}', [pattern])
#     return match


def parse_mwe_to_matcher(df_lex):

    matcher = Matcher(nlp.vocab)
    for p in df_lex.apply(lex_to_patterns, axis=1):
        matcher.add(*p, greedy="LONGEST")
    # for p in df_lex.apply(lex_to_patterns_lemma, axis=1):
    #     matcher.add(*p)
    return matcher

def lexicon_match(text):

    print(f'{text}')
    m = defaultdict(list)

    doc = nlp(text)
    matches = matcher(doc)

    m['n_match'] = len(matches)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        lex, pos, pol, intens = string_id.split(':')
        span = doc[start:end]
        m['lex'].append(lex)
        m['pos'].append(pos)
        m['polarity'].append(pol)
        m['intens'].append(intens)
        m['matched_spans'].append(str(span))
        m['matched_spans_idc'].append((start,end))
        print('\t' + string_id, span)
    return m

def sum_norm_wordlist_score(word_match):
    # sum of polarity matched tokens
    name = word_match.name
    seq_sum = word_match.apply(sum).rename(name + '-seqsum')
    seq_normlen = word_match.apply(lambda x: sum(x) / len(x)).rename(name + '-seqnorm')

    match_only = word_match.apply(lambda x: [i for i in x if i != False])
    match_norm = match_only.apply(lambda x: sum(x) / len(x) if len(x) else 0).rename(name + '-matchnorm')

    return pd.concat([seq_sum, seq_normlen, match_norm], axis=1)

def merge_intervals(intervals):
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for  t in s:
        if t[0] > s[m][1]:
            m += 1
            s[m] = t
        else:
            s[m] = (s[m][0], t[1])
    return s[:m+1]

def process_sentiecon(row, intensity_weight=True):
    '''
    Make neg, pos, neu match list features and combine in polarity.
    Intensity weighing or mean for overlap and polarity calc available.
    :param normalize: If overlapping lex matches: mean
    :return:
    '''
    feats = {}
    INTENSITY_WEIGHT = {0: 0.8, 1: 0.9, 2: 1, 3: 1.1}
    tok_len = len(row[0].split())
    toks_neg = [0] * tok_len
    toks_pos = [0] * tok_len
    toks_neu = [0] * tok_len
    scores = {'pos': 0, 'neg': 0, 'neu': 0}
    if pd.notnull(row.matched_spans_idc):
        pols, intens, idcs = literal_eval(row[1]), literal_eval(row[2]), literal_eval(row[3])
        n_match = len(pols)
        matched_idc = merge_intervals(idcs)
        matched_span_len = sum(x[1]-x[0] for x in matched_idc)
        matched_proportion = matched_span_len / tok_len
        for pol, itns, idc in zip(pols, intens, idcs):
            score = 1 if not intensity_weight else INTENSITY_WEIGHT[int(itns)]
            scores[pol] += score
            # for i in range(*idc): # for when you want to get fancy with token-level prior polarity derivation
            #     if pol == 'pos':
            #         toks_pos[i] += score
            #     elif pol == 'neg':
            #         toks_neg[i] += score
            #     elif pol == 'neu':
            #         toks_neu[i] += score
    else: # provide div by zero for non-matches
        matched_proportion = 1
        n_match = 1
    feats['secon_negative-seqsum'] = scores['neg']
    feats['secon_positive-seqsum'] = scores['pos']
    feats['secon_neutral-seqsum'] = scores['neu']
    # feats['secon_negative-seqall'] = sum(toks_neg) # multi- not uni-word lexicon matching, compute like uni inflates scores
    # feats['secon_positive-seqall'] = sum(toks_pos)
    # feats['secon_neutral-seqall'] = sum(toks_neu)
    feats['secon_negative-seqnorm'] = scores['neg'] * matched_proportion
    feats['secon_positive-seqnorm'] = scores['pos'] * matched_proportion
    feats['secon_neutral-seqnorm'] = scores['neu'] * matched_proportion
    feats['secon_negative-matchnorm'] = scores['neg'] / n_match
    feats['secon_positive-matchnorm'] = scores['pos'] / n_match
    feats['secon_neutral-matchnorm'] = scores['neu'] / n_match
    for n in ['seqsum', 'seqnorm', 'matchnorm']:
        feats[f'secon_polarity-{n}'] = feats[f'secon_positive-{n}'] - feats[f'secon_negative-{n}']
    return feats

if __name__ == "__main__":

    # text_col = 'polex+targets' # gold-polarexpr experiments
    # dataset_fp = '../data/sentivent_implicit.csv'
    # match_fp = 'sentiecon_match.tsv'
    # opt_fp = 'sentiecon_feats.csv'

    text_col = 'clause_text' # clause experiments
    dataset_fp = '../data/sentivent_implicit_clauses.csv'
    match_fp = 'clause_sentiecon_match.tsv'
    opt_fp = 'clause_sentiecon_feats.csv'

    if not Path(match_fp).is_file(): # avoid recomputing matches which is costly when they have been calced
        fp_lex = '../lexicons/Sentiecon_Tecnolengua_v1.0.csv'
        df_lex = pd.read_csv(fp_lex, sep='\t', skiprows=4, header=None, names=['mwe', 'pos', 'polarity', 'intensity']) # first 4 lines are comments

        matcher = parse_mwe_to_matcher(df_lex)

        df_dataset = pd.read_csv(dataset_fp, sep='\t')
        # from pandarallel import pandarallel
        # pandarallel.initialize(
        #     nb_workers=16,
        #     progress_bar=True
        # )
        # matches = df[text_col].parallel_apply(lexicon_match).to_list()
        matches = df_dataset[text_col].apply(lexicon_match).to_list()
        df_matches = pd.DataFrame().from_records(matches)
        df_c = pd.concat([df_dataset, df_matches], axis=1)
        df_c.to_csv(match_fp, sep='\t', index=False) # write
        df_d = pd.read_csv(match_fp, sep='\t',) # test reload

    # compute sentiment scores from matches
    df_matches = pd.read_csv(match_fp, sep='\t')
    scores = df_matches[[text_col, 'polarity.1', 'intens', 'matched_spans_idc']].apply(process_sentiecon, axis=1)
    scores_df = pd.DataFrame(scores.tolist())
    scores_df.to_csv(opt_fp, index=False)
    # df_secon_feats = pd.concat([df, scores_df], axis=1)




    pass

    # for text in df[text_col]: # needs to be lowered with Sentiecon
    #     print(text)
    #     doc = nlp(text)
    #     matches = matcher(doc)
    #
    #     for match_id, start, end in matches:
    #         string_id = nlp.vocab.strings[match_id]  # Get string representation
    #         span = doc[start:end]
    #         match_com[text].append((string_id, span))
    #         print(f'\t{string_id}\t{span}\t{[t.lemma_ for t in span]}')
    #
    #         # todo pick longest growing debt pil
    #
    #
    #     # if len(matches) % n_matchers != 0: # manually check difference in matchers: decide just combine both are good
    #     #     print(text)
    #     #     for string_id, span in match_com[text]:
    #     #         if 'lemma' in string_id:
    #     #             print(f'\t{string_id}\t{span}\t{[t.lemma_ for t in span]}')
    #     #         else:
    #     #             print(f'\t{string_id}\t\t{span}')
    # pass
