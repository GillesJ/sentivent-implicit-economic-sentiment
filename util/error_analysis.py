#!/usr/bin/env python3
'''
1. Determine winning model with results_hyperopt_to_table_signif_test.py and copy path to winning pred file that is downloaded.
2. Run this and generate confusion matrix + error analysis annotation file.
3. Analyze annotation file manual error category columns if exists.

error_analysis.py in sentivent-implicit-economic-sentiment
7/14/21 Copyright (c) Gilles Jacobs
'''
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

import itertools
from pathlib import Path
import matplotlib as mpl

mpl.use("pgf")

# I make my own newfig and savefig functions
def figsize(scale):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,  # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ],
}

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm, decimals=3) * 100
        print("Normalized confusion matrix")
        print(cm_norm)

        plt.imshow(cm_norm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            thresh = cm_norm.max() / 2
            plt.text(
                j,
                i,
                "{}%\n($n={}$)".format(round(cm_norm[i, j], 1), cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    else:
        print("Confusion matrix, without normalization")
        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            thresh = cm.max() / 2
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def savefig_confusion_matrix(y_true, y_pred, fp, title):
    # plot confusion matrix
    cm_norm_plot_fp = str(fp).replace(".pdf", "_norm.pdf")

    # Compute confusion matrix
    order = ["positive", "neutral", "negative", "none"]
    class_names = sorted(np.unique(df.labels), key=lambda x: order.index(x))
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    plt.tight_layout()
    plt.savefig(fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=class_names, normalize=True, title=title,
    )
    plt.tight_layout()
    plt.savefig(cm_norm_plot_fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    print("Saved confusion matrix plot to {}".format(fp))

if __name__ == '__main__':

    FROM_SCRATCH = False # set True to overwrite existing output files.
    data_dir = Path("../data/")
    pred_fs = {
        "gold": "senti-results-hyperparameter-search/winpreds/lexall-roberta-large-test-7ddd8foi.csv", # winner gold polar fact experiments
        "clause": "impliclaus-results-hyperparameter-search/winpreds/lexecon-roberta-large-test-go2f5zir.csv", # winner clause experiments
    }
    df_errors = pd.DataFrame()

    for exp_name, pred_fn in pred_fs.items():

        pred_fp = data_dir / pred_fn
        df = pd.read_csv(pred_fp)

        fp_error = Path(f"{pred_fn.split('-')[0]}-error-analysis.tsv")
        fp_error_manual = Path(f"{pred_fn.split('-')[0]}-error-analysis-manual.csv") # after manual analysis in spreadsheet
        fp_matrix_fig = Path(f"{pred_fn.split('-')[0]}_test_confusion_matrix.pdf")

        # Print P, R, F1-scores by label
        # print(classification_report(df.labels, df.pred)) # sanity check
        order = ["positive", "neutral", "negative", "none"]
        labels = sorted(np.unique(df.labels), key=lambda x: order.index(x))
        prfs = precision_recall_fscore_support(df.labels, df.pred, average=None, labels=labels)
        data = [[metric] + list(scores) for metric, scores in zip(["P", "R", "F$_1$", "support"], prfs)]
        df_scores_label = pd.DataFrame(data).set_index(0).T
        df_scores_label.index = labels
        df_scores_label[["P", "R", "F$_1$"]] = (df_scores_label[["P", "R", "F$_1$"]] * 100).round(1).astype(str)
        df_scores_label['support'] = df_scores_label["support"].astype(int)
        print(df_scores_label.to_latex())

        # write preds file for manual error analysis if it doesn't exist.
        if not fp_error.exists() or FROM_SCRATCH:
            df_error = df[df.labels != df.pred]
            df_error = df_error.reindex(df_error.columns.tolist() + ['strong_lexical_indicator', 'unusual_language', 'in_context', 'plausible_ambiguous'], axis=1)
            df_error.to_csv(fp_error, sep="\t", index=False)

        # Handle manual error analysis: load manually annotated file and print overview.
        if fp_error_manual.exists():
            df_errman = pd.read_csv(fp_error_manual, sep="\t")
            n_total = df_errman.shape[0]
            err_cols = [c for c in df_errman.columns if c.lower() not in
                        ["id", "text", "labels", "pred", "comment", "lexical_ambi", "strong_lexical_indicator"]]
            print(err_cols)
            df_errman = df_errman.dropna(how='all', subset=err_cols)
            n_analysed = df_errman.shape[0]
            print(f'Manually analysed {round(100*n_analysed/n_total, 2)}% ({n_analysed}/{n_total}) errors.')
            df_errc = df_errman[err_cols].count().apply(lambda x: f'{round(100*x/n_analysed,1)}% (n={x})')
            df_errors[exp_name] = df_errc

        # 2. Plot and write Confusion matrix
        if not fp_matrix_fig.exists() or FROM_SCRATCH:
            savefig_confusion_matrix(
               df.labels, df.pred, fp_matrix_fig, ""
            )

    print(df_errors.to_latex(
        caption="Manual error category frequencies for gold polar fact polarity and clause-based polarity testset errors.",
        label='tab:error_analysis'
    ))