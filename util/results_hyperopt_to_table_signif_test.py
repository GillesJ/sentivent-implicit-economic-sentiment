#!/usr/bin/env python3
'''
Given prediction .tsv files produced by model_selection script hyperopt_model*.py for different architectures,
compute McNemar's significance test on holdout test set.

For paper experiments, preds were logged at wandb.com and downloaded to ../data/predictions/

results_hyperopt_to_table_signif_test.py in sentivent-implicit-economic-sentiment
4/19/21 Copyright (c) Gilles Jacobs
'''
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from mlxtend.evaluate import mcnemar_table, mcnemar

def test_mcnemar(y_target, y_model1, y_model2):
    '''
    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    '''
    assert y_target.shape == y_model1.shape == y_model2.shape

    cont_table = mcnemar_table(
        y_target=y_target,
        y_model1=y_model1,
        y_model2=y_model2,
    )
    print(pd.DataFrame(cont_table, columns=['model2 correct', 'model2 wrong'], index=['model1 correct', 'model1 wrong']))
    chi2, p = mcnemar(ary=cont_table, corrected=True)

    # APA (American Psychological Association) style, which shows three digits but omits the leading zero (.123).
    # P values less than 0.001 shown as "< .001". All P values less than 0.001 are summarized with three asterisks,
    # with no possibility of four asterisks.
    print('chi-squared:', chi2)
    print('p-value:', p)

if __name__ == '__main__':

    # A. Read all hyperparameter search results by architecture and select winning model.
    hyperopt_dirp = Path('../data/results-hyperparameter-search')
    selection_metric = 'f1_macro_best_epoch_dev'
    overfit_metric = 'f1_macro_best_epoch_test'
    overfit_threshold = 0.05 # 5% performance diff threshold
    # A.1. Read in results from wandb.ai exports.
    all_runs = {}
    df_win = pd.DataFrame()
    for arch_fp in hyperopt_dirp.rglob('*.csv'):
        df = pd.read_csv(arch_fp)
        arch_name = arch_fp.name.split('-wandb')[0].replace('senti-', '')
        df['arch_name'] = arch_name
        df['lexicon_name'] = arch_name.split('-')[0]
        df['model_name'] = '-'.join(arch_name.split('-')[1:])
        all_runs[arch_name] = df
        # 2. select winning model: highest devset macro-F1 that does not overfit on holdout testset (within 5% performance)
        df['diff_dev_test'] = 1 - df[selection_metric]/df[overfit_metric]
        df_notoverfit = df.loc[df['diff_dev_test'].abs() <= overfit_threshold]
        winner = df_notoverfit.loc[df_notoverfit[selection_metric].idxmax()]
        df_win = df_win.append(winner, ignore_index=True)
        print(f'{arch_name} winner:\tf1_dev:\t{winner[selection_metric]}\tf1_macro_test:\t{winner[overfit_metric]}')
        print(f'\tdiff: {winner["diff_dev_test"]}')
        pass

    pred_dirp = Path('../data/predictions')
    pred_meta = { # architecture_name: (wandb_url, test_preds_filename)
        'roberta-base_no_lexicon': (
            'https://wandb.ai/gillesjacobs/senti-roberta-base/runs/8x9c1i9s',
            'dauntless-sweep-93-wandb_export_2021-04-19T11 47 38.519+02 00.csv',
        ),
        'roberta-base_lexicon-stable': (
            'https://wandb.ai/gillesjacobs/senti-roberta-base/runs/5md7h9l9/',
            'easy-sweep-17-wandb_export_2021-04-19T11 23 53.492+02 00.csv',
        ),}

    dfs_testpred = {name: pd.read_csv(pred_dirp / t[1]) for name, t in pred_meta.items()}
    y_target = list(dfs_testpred.values())[0]['labels'].to_numpy() # get gold_standard from random pred file

    # check correspondence with model selection run
    for name, df_preds in dfs_testpred.items():
        print(name.upper() + ':\t' + pred_meta[name][0])
        y_pred = df_preds['pred'].to_numpy()
        print(f1_score(y_target, y_pred, average='macro')) # manual check passed with full precision
        print(classification_report(y_target, y_pred))

    test_mcnemar(
        y_target,
        dfs_testpred['roberta-base_no_lexicon']['pred'],
        dfs_testpred['roberta-base_lexicon']['pred'],
    )
    pass

