#!/usr/bin/env python3
"""
Given prediction .tsv files produced by model_selection script hyperopt_model*.py for different architectures,
compute McNemar's significance test on holdout test set.

For paper experiments, preds were logged at wandb.com and downloaded to ../data/predictions/

results_hyperopt_to_table_signif_test.py in sentivent-implicit-economic-sentiment
4/19/21 Copyright (c) Gilles Jacobs
"""
import wandb
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
)
from mlxtend.evaluate import mcnemar_table, mcnemar, cochrans_q
from ast import literal_eval

def post_process_latex(tab, n_step=3, len_header=7, len_footer=3):
    '''

    :param tab: table as DataFrame
    :param n_step: amount of ablations per model arx
    :param len_header: len lines of header in latex tabkle, 7 with label and caption
    :param len_footer: len line of footer in latex table, 3 is default table footer
    :return: str with cleaned latex table
    '''
    lines = tab.splitlines()
    markup_remove = [
        "\\centering",
        #  "\\toprule",
        "\\midrule",  # midrule gets placed wrongly
    ]
    lines = [l for l in lines if l not in markup_remove]  # remove some markup
    # add grouping hspace every third line of content
    n = n_step  # amount steps to vertical space n ablations per arx
    len_header = len_header - len(
        markup_remove
    )
    for i in range(n + len_header, len(lines) - len_footer - n, n):
        lines[i] = lines[i].replace("\\\\", "\\\\[5pt]")
    tab_proc = "\n".join(lines)

    return tab_proc

def test_mcnemar(y_target, y_model1, y_model2):
    """
    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    """
    assert y_target.shape == y_model1.shape == y_model2.shape

    cont_table = mcnemar_table(
        y_target=y_target,
        y_model1=y_model1,
        y_model2=y_model2,
    )
    # print(pd.DataFrame(cont_table, columns=['model2 correct', 'model2 wrong'], index=['model1 correct', 'model1 wrong']))
    chi2, p = mcnemar(ary=cont_table, corrected=True)

    # APA (American Psychological Association) style, which shows three digits but omits the leading zero (.123).
    # P values less than 0.001 shown as "< .001". All P values less than 0.001 are summarized with three asterisks,
    # with no possibility of four asterisks.
    # print('chi-squared:', chi2)
    # print('p-value:', p)
    return chi2, p


def format_pval(p):
    # APA (American Psychological Association) style, which shows three digits but omits the leading zero (.123).
    # P values less than 0.001 shown as "< .001". All P values less than 0.001 are summarized with three asterisks,
    # with no possibility of four asterisks.
    if 0.01 < p <= 0.05:
        return f"${str(round(p, 3)).strip('0')}^{{*}}$"
    elif 0.001 < p <= 0.01:
        return f"${str(round(p, 3)).strip('0')}^{{**}}$"
    elif 0.0 < p < 0.001:
        return "$<.001^{{***}}$"
    elif p == 0.0:
        return "-"
    else:
        return str(round(p, 3)).strip("0")




def export_wandb(projects, entity="gillesjacobs"):

    wandb_api = wandb.Api()
    for proj in projects:
        print(f"WandB.ai export download: {proj}")
        runs = wandb_api.runs(f"{entity}/{proj}")
        summary_list = []
        config_list = []
        name_list = []
        for run in runs:
            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)
            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            # run.name and run.id is the name/id of the run.
            name_list.append({"run_name": run.name, "run_id": run.id})
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame.from_records(name_list)
        all_df = pd.concat([name_df, config_df, summary_df], axis=1)
        all_df.to_csv(
            hyperopt_dirp / f"{proj}-{pd.Timestamp.now()}.csv", index=False
        )  # write for recordkeeping


def load_preds(fp, return_target=False):
    # print(f'Loading preds from {dataset_fp}')
    df = pd.read_csv(fp)
    y_pred = df["pred"].to_numpy()
    if return_target:
        y_true = df["labels"].to_numpy()
        return y_pred, y_true
    else:
        return y_pred


def format_float(f, fmt="{:,.1f}", remove_trailing_zero=True, percent=True):

    if isinstance(f, float):
        if percent:
            f = 100.0 * f
        f_fmt = fmt.format(f)
        if remove_trailing_zero:
            f_fmt = f_fmt.lstrip("0")
        return f_fmt
    else:
        return f


def bold_extreme_values(data, max=-1, float_fmt=format_float):

    if data == float(max):
        data_str = (
            float_fmt.format(data) if isinstance(float_fmt, str) else float_fmt(data)
        )
        return "\\textbf{%s}" % data_str

    return data


def col_sort(c):
    if c.split("_")[-1] == "dev":
        return 0
    elif c.split("_")[-1] == "test":
        return 1
    elif "p-vs" in c:
        return 2


def format_metric(val, tag):
    if "{}" in tag:
        return tag.replace("}", f"{val}}}")
    else:
        return f"{tag} {val}"


def format_abl_names(arx, abl):
    arx_rename = {
        "ProsusAI-finbert": "FinBERT-SST$_{Base}$ \citep{araci2019finbert}",
        "abhilash1910-financial_roberta": "FinRoBERTa$_{Base}$ \citep{abhilash-finroberta}",
        "bert-base-cased-fix": "BERT$_{Base}$ \citep{devlin-etal-2019-bert}",
        "bert-large-cased": "BERT$_{Large}$ \citep{devlin-etal-2019-bert}",
        "finbert-finvocab-uncased": "FinBERT$_{Base}$ \citep{yang2020finbert}",
        "microsoft-deberta-base": "DeBERTa$_{Base}$ \citep{he2021deberta}",
        "roberta-base": "RoBERTa$_{Base}$ \citep{liu2019roberta}",
        "roberta-large": "RoBERTa$_{Large}$ \citep{liu2019roberta}",
    }
    if abl == "":
        arx = arx_rename.get(arx, arx)
        return arx
    else:
        return f"\\hspace{{5pt}} {abl}"


if __name__ == "__main__":

    exp_name = "impliclaus" # senti = gold, impliclaus = clause-based
    EXPORT_WANDB = False  # export overview from wandb API, else load from local storage, first run requires retrieval
    overfit_threshold = (
        1.00  # 0.05 = 5% performance diff threshold, 1.0 = 100% disables overfit check
    )
    selection_metric = "f1_macro_best_epoch_dev"
    overfit_metric = "f1_macro_best_epoch_test"

    hyperopt_dirp = Path(f"../data/{exp_name}-results-hyperparameter-search")
    winpred_dirp = hyperopt_dirp / "winpreds/"
    winpred_dirp.mkdir(parents=True, exist_ok=True)
    # 0. Export all results from wandb first, can be skipped if data is collected
    # list hyperopt architecture projects to include here
    exp_projects = {
        "senti": [
            "senti-nolex-roberta-large",
            "senti-lexall-roberta-large",
            "senti-lexecon-roberta-large",
            "senti-nolex-roberta-base",
            "senti-lexall-roberta-base",
            "senti-lexecon-roberta-base",
            "senti-nolex-bert-large-cased",
            "senti-lexall-bert-large-cased",
            "senti-lexecon-bert-large-cased",
            # "senti-nolex-bert-base-cased", "senti-lexall-bert-base-cased", "senti-lexecon-bert-base-cased", # accidental large lr and batchsize param space
            "senti-nolex-bert-base-cased-fix",
            "senti-lexall-bert-base-cased-fix",
            "senti-lexecon-bert-base-cased-fix",  # fix
            "senti-nolex-microsoft-deberta-base",
            "senti-lexall-microsoft-deberta-base",
            "senti-lexecon-microsoft-deberta-base",
            "senti-nolex-abhilash1910-financial_roberta",
            "senti-lexall-abhilash1910-financial_roberta",
            "senti-lexecon-abhilash1910-financial_roberta",
            "senti-nolex-finbert-finvocab-uncased",
            "senti-lexall-finbert-finvocab-uncased",
            "senti-lexecon-finbert-finvocab-uncased",
            "senti-nolex-ProsusAI-finbert",
            "senti-lexall-ProsusAI-finbert",
            "senti-lexecon-ProsusAI-finbert",
        ],
        "impliclaus": [
            "impliclaus-lexall-ProsusAI-finbert",
            "impliclaus-lexecon-ProsusAI-finbert",
            "impliclaus-nolex-ProsusAI-finbert",
            "impliclaus-lexall-roberta-large",
            "impliclaus-lexecon-roberta-large",
            "impliclaus-nolex-roberta-large",
            "impliclaus-lexall-roberta-base",
            "impliclaus-lexecon-roberta-base",
            "impliclaus-nolex-roberta-base",
            "impliclaus-lexall-microsoft-deberta-base",
            "impliclaus-lexecon-microsoft-deberta-base",
            "impliclaus-nolex-microsoft-deberta-base",
            "impliclaus-lexall-bert-large-cased",
            "impliclaus-lexecon-bert-large-cased",
            "impliclaus-nolex-bert-large-cased",
            "impliclaus-lexall-bert-base-cased",
            "impliclaus-lexecon-bert-base-cased",
            "impliclaus-nolex-bert-base-cased",
            "impliclaus-lexall-finbert-finvocab-uncased",
            "impliclaus-lexecon-finbert-finvocab-uncased",
            "impliclaus-nolex-finbert-finvocab-uncased",
        ],
    }
    projects = exp_projects[exp_name]

    if EXPORT_WANDB:
        export_wandb(projects)

    # A. Read all hyperparameter search results by architecture and select winning model.

    # A.1. Read in results from wandb.ai exports.
    all_runs = {}
    df_win = pd.DataFrame()
    for arch_fp in hyperopt_dirp.glob("*.csv"):
        df = pd.read_csv(arch_fp)
        project_name = arch_fp.name.split("-202")[0]
        arch_name = project_name.replace(f"{exp_name}-", "")
        df["arch_name"] = arch_name
        df["lexicon_name"] = arch_name.split("-")[0]
        df["model_name"] = "-".join(arch_name.split("-")[1:])
        all_runs[arch_name] = df
        # A.2. select winning model: highest devset macro-F1 that does not overfit on holdout testset (within 5% performance)
        df["diff_dev_test"] = 1 - df[selection_metric] / df[overfit_metric]
        df_notoverfit = df.loc[df["diff_dev_test"].abs() <= overfit_threshold]
        winner = df_notoverfit.loc[df_notoverfit[selection_metric].idxmax()].copy()
        print(f"{arch_name}, non-overfit winner {overfit_threshold}:")
        print(
            f"\t{selection_metric}:\t{winner[selection_metric]}\t{overfit_metric}:\t{winner[overfit_metric]}"
        )
        print(f'\tdev. eval - holdout test diff: {winner["diff_dev_test"]}.')

        # A.3. Get dev and test pred files for winning run
        print("\tdownloading predictions.")
        wandb_api = wandb.Api()
        temp_dir = winpred_dirp / "tmp/"
        run_win = wandb_api.run(f"gillesjacobs/{project_name}/{winner['run_id']}")
        for split in ["dev", "test"]:
            win_preds_fp = winpred_dirp / f"{arch_name}-{split}-{winner['run_id']}.csv"
            if not win_preds_fp.exists():  # download preds from WANDB API
                fp_preds = literal_eval(winner[f"{split}_preds_best_epoch"])["path"]
                fp_preds = run_win.file(fp_preds).download(
                    temp_dir, replace=True
                )  # downloading to file is hacky but the only way currently
                df_preds = pd.DataFrame(**json.load(fp_preds))
                df_preds.to_csv(win_preds_fp, index=False)
            winner[f"{split}_preds_fp"] = win_preds_fp
        df_win = df_win.append(winner, ignore_index=True)
        temp_dir.unlink  # clean temporary downloads

    # B.1 Load dev + test preds and apply McNemar's significance test
    df_win = df_win.sort_values(by=selection_metric, ascending=False).reset_index()
    df_win["test_preds"] = df_win["test_preds_fp"].apply(load_preds)
    df_win["dev_preds"] = df_win["dev_preds_fp"].apply(load_preds)
    best = df_win.iloc[0]
    dev_pred_best, dev_target = load_preds(best["dev_preds_fp"], return_target=True)
    # dev_pred_best[dev_pred_best == 4] = -1
    # dev_target[dev_target == 4] = -1
    test_pred_best, test_target = load_preds(best["test_preds_fp"], return_target=True)

    # print("Computing Cochran's test across all predictions...") # this can take a while for a lot of runs
    # q, p_value = cochrans_q(test_target, *df_win["test_preds"].to_list())
    # print('\tCochran\'s Q: %.3f' % q)
    # print(f"\tp-value: {p_value}") # if p-value < 0,05 -> h0 that there is no difference in clf accuracies rejected

    for i, r in df_win.iterrows():
        print(f"{r['arch_name']} pairwise t-test with best preds")
        test_pred = r["test_preds"]
        dev_pred = r["dev_preds"]

        print("DEV CLF REPORT")
        print(classification_report(dev_target, dev_pred))
        print("TEST CLF REPORT")
        print(classification_report(test_target, test_pred))

        print(f"{r['arch_name']} vs {best['arch_name']}")
        chi2, p = test_mcnemar(
            test_target,
            test_pred,
            test_pred_best,
        )
        df_win.loc[i, f"chi2-vs-{best['arch_name']}"] = chi2
        df_win.loc[i, f"p-vs-{best['arch_name']}"] = p

    # Format table
    float_fmt = "{:,.1f}"
    df_win["lexicon_name"] = df_win["lexicon_name"].map(
        {"nolex": "", "lexecon": "+ econ.", "lexall": "+ econ.+general"}
    )  # better sort+naming for end result
    df_win = df_win.set_index(["model_name", "lexicon_name"])
    cols_to_keep = [
        f"p-vs-{best['arch_name']}",
    ]
    for m in ["p_macro", "r_macro", "f1_macro", "f1_micro"]:
        cols_to_keep.append(f"{m}_best_epoch_dev")
        cols_to_keep.append(f"{m}_best_epoch_test")

    cols_to_keep.sort(key=lambda x: col_sort(x))
    df_table = df_win[cols_to_keep]
    df_table = df_table.sort_index()
    df_table.columns = df_table.columns.str.replace("_best_epoch", "")
    df_table = df_table.rename(
        columns={"f1_micro_dev": "A_dev", "f1_micro_test": "A_test"}
    )

    # find global and local arch max indices
    metric_cols = [c for c in df_table.columns if "p-vs" not in c]
    max_glob_ixs = [[df_table[c].idxmax(), c] for c in metric_cols]
    max_loc_ixs = []
    for arx, df_arx in df_table.groupby(level=0):
        max_loc_ixs.extend([[df_arx[c].idxmax(), c] for c in metric_cols])

    # FORMATTING (yeah that's a lot of code for table markup)
    # format the floats to percent
    df_table[metric_cols] = (df_table[metric_cols] * 100).round(decimals=1)
    # format p-values
    df_table[f"p-vs-{best['arch_name']}"] = df_table[f"p-vs-{best['arch_name']}"].apply(lambda x: format_pval(x))

    # format the score highlights
    for idx, c in max_glob_ixs:
        df_table.loc[idx, c] = format_metric(df_table.loc[idx, c], "\\textbf{}")
    for idx, c in max_loc_ixs:
        df_table.loc[idx, c] = format_metric(df_table.loc[idx, c], "\\underline{}")

    df_table = df_table.rename(columns={f"p-vs-{best['arch_name']}": "$p$"})
    df_table.columns = df_table.columns.str.replace("_", "-")  # remove underscores
    df_table = df_table.replace("_", "-", regex=True)
    df_table["model w\ lexicons"] = [
        format_abl_names(m, l) for m, l in df_table.index.to_flat_index()
    ]
    cols_in_order = ["model w\ lexicons"] + list(df_table.columns[:-1])  # order columns
    df_table = df_table[cols_in_order]

    print(f"\n\section{{{exp_name} DEV+TEST results table}}\n")
    caption_devtest = f"Coarse-grained implicit sentiment results on development set and holdout test set for winning models after optimising hyperparameters for each architecture.  Precision (P), recall (R), $F_1$-score ($F_1$) percentages are macro-averaged. Accuracy (A) is reported with the p-value of McNemar's significance test of predictions w.r.t. the best model ({best['arch_name']})"
    cols_devtest = sorted(
        [tuple(c.rsplit("-", 1)) for c in df_table.columns],
        key=lambda x: ['model w\\ lexicons', 'p-macro', 'r-macro', 'f1-macro', 'A', '$p$'].index(x[0]),
    )
    col_sort = sorted([c for c in df_table.columns], key=lambda x: ['model w\\ lexicons', 'p-macro', 'r-macro', 'f1-macro', 'A', '$p$'].index(x.rsplit('-', 1)[0]))
    df_devtest = df_table[col_sort].copy()
    df_devtest.columns = pd.MultiIndex.from_tuples(cols_devtest)
    print(
        post_process_latex(
            df_devtest.to_latex(
                escape=False,
                index=False,
                caption=caption_devtest,
                label="tab:coarse-result-dev+test",
            ),
            len_header=8,
        )
    )

    print(best["arch_name"], best["lexicon_name"], " classification report:")
    print("DEV CLF REPORT")
    print(pd.DataFrame(classification_report(dev_target, dev_pred_best, output_dict=True)).transpose().round(3).to_latex())
    print("TEST CLF REPORT")
    print(pd.DataFrame(classification_report(test_target, test_pred_best, output_dict=True)).transpose().round(3).to_latex())

    # split test and dev for space in manuscript
    print(f"\n\section{{{exp_name} DEV ONLY results table}}\n")
    caption_dev = (
        "Coarse-grained implicit sentiment results on development set for winning models after optimising "
        "hyperparameters for each architecture.  Precision (P), recall (R), $F_1$-score ($F_1$) percentages "
        "are macro-averaged, accuracy (A)."
    )
    df_dev_table = df_table[[c for c in df_table.columns if "dev" in c or "model" in c]]
    df_dev_table = df_dev_table.rename(
        columns={c: c.split("-")[0] for c in df_dev_table.columns}
    )
    print(
        post_process_latex(
            df_dev_table.to_latex(
                escape=False,
                index=False,
                caption=caption_dev,
                label="tab:coarse-result-dev",
            ),
        )
    )

    print(f"\n\section{{{exp_name} TEST ONLY results table}}\n")
    df_test_table = df_table[
        [c for c in df_table.columns if "test" in c or "model" in c or "$p$" in c]
    ]
    caption_test = (
        f"Holdout testset results for coarse-grained implicit polarity classification.  Precision (P), recall"
        f" (R), $F_1$-score ($F_1$) percentages are macro-averaged. Accuracy (A) is reported with the p-value"
        f" of McNemar's significance test of predictions w.r.t. the best model ({best['arch_name']})."
    )
    df_test_table = df_test_table.rename(
        columns={c: c.split("-")[0] for c in df_test_table.columns}
    )
    print(
        post_process_latex(
            df_test_table.to_latex(
                escape=False,
                index=False,
                caption=caption_test,
                label="coarse-result-test",
            )
        )
    )

