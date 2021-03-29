#!/usr/bin/env python3
'''
Explain purpose of script here

hyperoptim_sweep_layerwise.py in sentivent-implicit-economic-sentiment
3/16/21 Copyright (c) Gilles Jacobs
'''

import logging
from statistics import mean

import pandas as pd
import wandb
from sklearn.metrics import f1_score

from simpletransformers.classification import ClassificationArgs, ClassificationModel

layer_parameters = {f"layer_{i}-{i + 6}": {"min": 0.0, "max": 5e-5} for i in range(0, 24, 6)}

sweep_config = {
    "name": "layerwise-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "f1_macro", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 40},
        "params_classifier-dense-weight": {"min": 0, "max": 1e-3},
        "params_classifier-dense-bias": {"min": 0, "max": 1e-3},
        "params_classifier-out_proj-weight": {"min": 0, "max": 1e-3},
        "params_classifier-out_proj-bias": {"min": 0, "max": 1e-3},
        **layer_parameters,
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6,},
}

sweep_id = wandb.sweep(sweep_config, project="Sentivent - Hyperparameter Optimization")

# logging.basicConfig(level=logging.INFO)
# wandb_logger = logging.getLogger("wandb")
# wandb_logger.setLevel(logging.ERROR)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

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

model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-5
model_args.manual_seed = 4
model_args.max_seq_length = 128
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 8
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 32
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.wandb_project = "Sentivent - Hyperparameter Optimization"


def train():
    # Initialize a new wandb run
    wandb.init()

    # Get sweep hyperparameters
    args = {key: value["value"] for key, value in wandb.config.as_dict().items() if key != "_wandb"}
    print(args)
    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in args.items():
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay if "bias" not in params_key else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups
    model_args.update_from_dict(cleaned_args)

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=3,
        use_cuda=True,
        args=model_args)

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

    # Sync wandb
    wandb.join()

# train() # this works fine
wandb.agent(sweep_id, train)