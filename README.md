# Coarse-grained implicit sentiment polarity classification
This repos contains replication data in for the paper "Fine-Grained Implicit Sentiment in Financial News: Uncovering Hidden Bulls and Bears." by Gilles Jacobs and Véronique Hoste.
The scientific repository for all experiments and data in the paper can be found at: https://osf.io/cbjta/wiki/home/

The '/data' subfolder and hyperoptim. `src` contains training, testing, validation source code.

Several utility scripts for the pre-processing of lexicons are also included, but the lexicons cannot be included due to copyright issues. https://github.com/GillesJ/sentivent-implicit-economic-sentiment

Use:
    Model training and tokenization code in custom_model.py and custom_classification_model.py, run hyperoptim search with python hyperopt_model_train.py
    /data/: contains coarse-grained datasets in json format.
    /utils/: various utility scripts for viz., lexicon pre-processing, and EDA.
