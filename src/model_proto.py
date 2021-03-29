#!/usr/bin/env python3
'''
Prototype model using simpletransformers roberta classification to check viability of task.

model_proto.py in sentivent-implicit-economic-sentiment
3/16/21 Copyright (c) Gilles Jacobs
'''
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if __name__ == '__main__':

    # Preparing train data
    polarity_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    # data_fp = '../data/sentivent-plus-sentifm-as-train.csv'
    data_fp = '../data/sentivent_implicit.csv'
    df_all = pd.read_csv(data_fp, sep='\t')
    df_all = df_all.rename(columns={'polex+targets': 'text'})
    # df_all = df_all.rename(columns={'polex': 'text'})
    df_all['labels'] = df_all['polarity'].map(polarity_map)

    df_train = df_all[df_all['split'] == 'train']
    df_dev = df_all[df_all['split'] == 'dev']
    df_test = df_all[df_all['split'] == 'test']

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=8,
                                    train_batch_size=32,
                                    overwrite_output_dir=True,
                                    use_early_stopping=True,
                                    evaluate_during_training=True,
                                    )

    # Create a ClassificationModel
    model = ClassificationModel(
        'roberta',
        'roberta-base',
        num_labels=3,
        args=model_args
    )

    # Train the model
    f1_micro = lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
    f1_macro = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
    model.train_model(df_train,
                      eval_df=df_dev,
                      f1_macro=f1_macro,
                      )

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        df_dev,
        f1_micro=f1_micro, f1_macro=f1_macro)
    print(f'Dev set: {result}')
    result, model_outputs, wrong_predictions = model.eval_model(
        df_test,
        f1_micro=f1_micro, f1_macro=f1_macro)
    print(f'Test set: {result}')

