#!/usr/bin/env python3
'''
Model train calling using modified simpletransformers classes + weightsandbiases(wandb.com) for experiment management/logging.
Hyperparameter space partially constrained in previous iterations.

hyperopt_model_train.py in sentivent-implicit-economic-sentiment
3/16/21 Copyright (c) Gilles Jacobs
'''
from simpletransformers.classification import ClassificationArgs
from custom_classification_model import CustomClassificationModel
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from mlxtend.evaluate import mcnemar_table, mcnemar
import wandb

if __name__ == '__main__':

    # SET quick choices
    parser = argparse.ArgumentParser(description='SENTiVENT coarse implicit sentiment.')
    parser.add_argument(
        '--lexfeat', '-l',
        help='Set lexicon feature group',
        default='nolex',
        type=str,
        choices=['nolex', 'lexall', 'lexecon'],
    )
    parser.add_argument(
        '--model', '-m',
        help='Set transformer model',
        default='roberta-base',
        type=str,
        choices=['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased'],
    )
    args = parser.parse_args()

    # Preparing train data
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    label_map_inv = {v: k for k,v in label_map.items()}
    # data_fp = '../data/sentivent-plus-sentifm-as-train.csv'
    data_fp = '../data/sentivent_implicit.csv'
    df_all = pd.read_csv(data_fp, sep='\t')
    df_all['text'] = df_all['polex+targets']
    df_all['labels'] = df_all['polarity']

    # load lexicon features and append feature vector columns
    fp_lexfeats = 'sentivent-implicit-lexiconfeats-all.csv'
    df_lexfeats = pd.read_csv(fp_lexfeats)
    # define features using pandas regex filters per architecture
    LEXALL = {
         # all features: subwordlist matches + polarity all norm methods + polarity combination across lexicons
        'all': '.*',
        # pos+neg wordlist + polarity all norm methods + polarity combination across lexicons
        'posnegneupol': '(posemo|negemo|positive|negative|neutral|market_sentiment|polarity)',
        # only polarity (separate lexicons + combined)
        'pol': '(market_sentiment|polarity)',
    }
    LEXECON = {
         # all features: subwordlist matches + polarity all norm methods + polarity combination across lexicons
        'all': '(ntusd|henry|lm|secon|econcombo)_',
        # pos+neg wordlist + polarity all norm methods + polarity combination across lexicons
        'posnegneupol': '(ntusd|henry|lm|secon|econcombo)_(posemo|negemo|positive|negative|neutral|market_sentiment|polarity)',
        # only polarity (separate lexicons + combined)
        'pol': '(ntusd|henry|lm|secon|econcombo)_(market_sentiment|polarity)',
    }
    NOLEX = {'nolex': '^\b$',} # No lexicons: won't match anything
    LEXFEAT = {'lexall': LEXALL, 'lexecon': LEXECON, 'nolex': NOLEX}

    ARCH_FEAT = args.lexfeat
    ARCH_MODEL = args.model
    ARCH_MODEL_BASE = args.model.split('-')[0] + "-lexicon" # roberta|bert-lexicon are our custom classes
    LEXFEAT = LEXFEAT[ARCH_FEAT]
    # testing feat_filter by manually checking cols -> ok
    feats_unittest = {k: df_lexfeats.filter(regex=re).columns.tolist() for k, re in LEXFEAT.items()}
    feats_unittest_excl = {k: set(df_lexfeats.columns.tolist()) - set(df_lexfeats.filter(regex=re).columns.tolist()) for k, re in LEXFEAT.items()}

    # Sweep configuration WANDB hyperoptim
    arch_name = f'senti-{ARCH_FEAT}-{ARCH_MODEL}'
    if 'bert-base-cased' in ARCH_MODEL: # signal fix on new sweep project runs
        arch_name += '-fix'
    if '-large' in arch_name:
        lr_param = {"min": 4e-5, "max": 8e-5} # 4e-5 -> 7e-5 works good for roberta|bert large w batch sizes 32, 64
        bs_param = {'values': [32, 64]} # 32-64 works best with large, 16 seems to be way worse across board
    if '-base' in arch_name:
        lr_param = {"min": 4e-5, "max": 1e-4} # 4e-5 -> 7e-5 works best for roberta|bert_base
        bs_param = {'values': [16, 32]} # 16, 32 best for roberta|bert-base

    sweep_config = {
        "name": "paper-experiments",
        "method": "bayes",
        "metric": {"name": "f1_macro", "goal": "maximize"},
        "parameters": { # SET for each model arch
            "learning_rate": lr_param,
            "train_batch_size": bs_param,
            "lexicon_features": {'values': list(LEXFEAT.keys())},
            },
        "early_terminate": {"type": "hyperband", "min_iter": 6,},
    }
    sweep_id = wandb.sweep(sweep_config, project=arch_name)

    def train():
        # Initialize a new wandb run
        wandb.init()

        # load lexicon features
        lex_feat_group = wandb.config['lexicon_features']
        re = LEXFEAT[lex_feat_group]
        df_lexfeats_filter = df_lexfeats.filter(regex=re)
        df_all['lexicon_features'] = df_lexfeats_filter.values.tolist()
        lexicon_feat_dim = len(df_all['lexicon_features'][0])
        print(f'{lex_feat_group}: {df_lexfeats_filter.shape[1]} lexicon features selected by filter {re}')

        # split in train-dev-test
        df_train = df_all[df_all['split'] == 'train']
        df_dev = df_all[df_all['split'] == 'dev']
        df_test = df_all[df_all['split'] == 'test']

        # Default model config, these will be default unless overwritten by sweep_config
        output_dir = f'./outputs/{arch_name}'
        model_args = ClassificationArgs()

        model_args.lexicon_feat_dim = lexicon_feat_dim
        model_args.labels_map = label_map
        model_args.max_seq_length = 64 # no truncate for 99.9% (99.8% for cased) cf. ../util/determine_max_seq_length.py

        model_args.num_train_epochs = 8 # best 8 or 16
        model_args.train_batch_size = 16 # 32 ok
        model_args.learning_rate = 4e-5 # 4e-5 = default and works well
        model_args.use_early_stopping = True
        model_args.early_stopping_consider_epochs = True
        model_args.early_stopping_delta = 0
        model_args.early_stopping_patience = 3
        model_args.early_stopping_metric = 'f1_macro'
        model_args.early_stopping_metric_minimize = False

        model_args.output_dir = output_dir
        model_args.best_model_dir = f'{output_dir}/best_model/'
        model_args.save_best_model = True
        model_args.save_eval_checkpoints = False
        model_args.save_model_every_epoch = False
        model_args.save_optimizer_and_scheduler = False
        model_args.save_steps = -1
        model_args.no_cache = True
        # model_args.no_save = True # DO NOT SET THIS TO TRUE IF YOU WANT BEST EPOCH EVAL

        model_args.evaluate_during_training = True
        model_args.evaluate_during_training_verbose = False # this doesnt actually print metrics
        model_args.evaluate_during_training_steps = -1
        model_args.evaluate_each_epoch = True
        model_args.eval_batch_size = 32
        model_args.evaluate_during_training_silent = True
        model_args.overwrite_output_dir = True
        model_args.manual_seed = 1992
        model_args.reprocess_input_data = True
        # model_args.use_cached_eval_features = False

        # Create a ClassificationModel
        model = CustomClassificationModel(
            ARCH_MODEL_BASE,
            ARCH_MODEL,
            num_labels=3,
            args=model_args,
            sweep_config=wandb.config,
        )

        # Train the model
        eval_metrics = {
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
            'p_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
            'r_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro', zero_division=0),
            'p_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro', zero_division=0),
            'r_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro', zero_division=0),
        }

        model.train_model(df_train,
                          eval_df=df_dev,
                          **eval_metrics
                          )

        # eval the model at best epoch for preds
        best_model = CustomClassificationModel(
            ARCH_MODEL_BASE,
            f'{output_dir}/best_model/',
            num_labels=3,
            args=model_args,
            sweep_config=wandb.config,
        )

        # eval and log best epoch predictions in wandb
        result, model_outputs, wrong_predictions = best_model.eval_model(
            df_dev,
            **eval_metrics)
        print(f'Best epoch dev set: {result}')
        wandb.log({f'{k}_best_epoch_dev': v for k, v in result.items()})
        preds_dev = np.argmax(model_outputs, axis=1)
        result, model_outputs, wrong_predictions = best_model.eval_model(
            df_test,
            **eval_metrics)
        print(f'Best epoch test set: {result}')
        wandb.log({f'{k}_best_epoch_test': v for k, v in result.items()})
        preds_test = np.argmax(model_outputs, axis=1)

        labels_dev = [label_map_inv[i] for i in preds_dev]
        labels_test = [label_map_inv[i] for i in preds_test]
        df_dev['pred'] = labels_dev
        df_test['pred'] = labels_test
        wandb.log({'dev_preds_best_epoch': wandb.Table(dataframe=df_dev[['id', 'text', 'labels', 'pred']])})
        wandb.log({'test_preds_best_epoch': wandb.Table(dataframe=df_test[['id', 'text', 'labels', 'pred']])})

        df_dev.to_csv(Path(output_dir) / 'dev_preds_best_epoch.tsv', sep='\t', index=False)
        df_test.to_csv(Path(output_dir) / 'test_preds_best_epoch.tsv', sep='\t', index=False)

        # Sync wandb
        wandb.join()

    wandb.agent(sweep_id, train)