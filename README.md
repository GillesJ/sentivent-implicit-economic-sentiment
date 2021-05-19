# 

## Exploratory Experiment notes

- Experiment avenues:
Research goal: check feasibility of annotated sentiment data in classification.
  Current setup: feed sentiment expression + target tokens to transformer classifier (+/- lexicon features)
  - current model selection insights: performance with lexicon features is signif. better,
    economic lexicon features only seem to be more stable and outperform econ+general domain lexicon feats. 
    But more model selection is needed + potentially more lexicons.
    
-> Lexicon features avenues:
    - Model selection insight:
        - Roberta-large: all-lexicon and econlexicon-pol and no-lexicon (but too little runs for consistency)
        - Roberta-base: econlexicon-pol and all-lexicon
        -> right now focus on Roberta-base: clear improvement with lexicon, possible with econlex 
    - General domain: Currently, LIWC and SentiWordnet3:
    -> Add more standard lexicons like Liu, AFINN, GI, VADER as these are pretty standard.
    - Economic lexicon: Currently: LoughranMcDonald + NTUSDFin + henry2008(rather small) best on Roberta-large and base
    -> ADD Senticonelex: uses MWE but high quality SotA -> could improve performance further
    
-> clause-based polarity clf: split all data in CLAUSES using splitter script (pos, neg, neu, none) -> more realistic engineering task
-> Model selection:
FIRST FOCUS on Roberta-base for faster iteration, also more stable.
  - Other modeling avenues: 
    - Other transformer architectures (disadvantage: need to code specialty classes for each one, know how to)
    - Other models (?) tfidf-ngram svm + feats or other clfs (?) not really interesting
    -> Issue: some overfitting on dev data in hyperoptim
  solution: take 3-fold? -> 3x longer model selection not really feasible
  
  
## MODEL FINAL on simpletransformers + lexicon feats
- Roberta-base hyperoptim: first 60 runs: batch_size-16|32, lr-1e-5->4e-5, pol-all good, posnegpol-seqsumnorm
- Roberta-base hyperoptim: ECON-LEXICON features: best std ratio and median performance is Econ-lexicon all features and econ-lexicon-polarity comes close
- OPTIONS DO HYPEROPTIM for model comparison: Roberta-base-ECONLEXICON vs Roberta-base-ALLLEXICON vs. Roberta-base-GENERALLEXICON


- BEST: roberta_large + polarity_all 8 epoch
```
Dev set: {'mcc': 0.5777876063605787, 'f1_micro': 0.7748478701825557, 'f1_macro': 0.6338066786602078, 'eval_loss': 1.0496873797908906}
Test set: {'mcc': 0.5466407892681818, 'f1_micro': 0.754442649434572, 'f1_macro': 0.6295099084274117, 'eval_loss': 1.2079995087622546}
```
=> NOTE LOSS DOESNT DECREASE THAT MUCH -> increase lr? -> increase epochs?
  
- roberta_large + polarity_all 16 epoch
```
Dev set: {'mcc': 0.38394962956033046, 'f1_micro': 0.6977687626774848, 'f1_macro': 0.4610058775912435, 'eval_loss': 0.7569411243161848}
Test set: {'mcc': 0.30442553304937564, 'f1_micro': 0.6558966074313409, 'f1_macro': 0.4283435862383231, 'eval_loss': 0.8420870479864951}
```
idem but with model_args.learning_rate = 1e-4

### Model prototype based on simpletransformers fine-tuning
- src/model_proto.py
```
Basis Bert
Dev set: {'mcc': 0.5591721896263163, 'f1_micro': 0.7667342799188641, 'f1_macro': 0.6237267073416611, 'eval_loss': 1.3180186481974576}
Test set: {'mcc': 0.47618812704105107, 'f1_micro': 0.7189014539579967, 'f1_macro': 0.5786110731288018, 'eval_loss': 1.5914541616705395}

+ SentiFM (Marjan haar dataset) toevoegen als traindata helpt niet veel op het eerste zicht, scores dalen gemiddeld met een procentje.
(+SentiFM en de train-dev-test-split behouden is het slechtste).
- target tokens wegdoen, i.e., enkel de polar expression zonder de target tokens en wat er tussen zit, is ook iets slechter.

- RoBERTa ipv BERT (best):
Dev set: {'mcc': 0.5689670341477926, 'f1_micro': 0.7697768762677485, 'f1_macro': 0.639110290724504, 'eval_loss': 1.0877807861637143}
Test set: {'mcc': 0.5253007222388643, 'f1_micro': 0.7439418416801292, 'f1_macro': 0.6191855814719985, 'eval_loss': 1.2177074456226922}
```

### BERTICON

Exp 1: config-sentivent-roberta
```
24-Mar 00:12:56 - [INFO]: Classification report:
              precision    recall  f1-score   support
    positive       0.81      0.77      0.79       727
    negative       0.67      0.72      0.69       363
     neutral       0.33      0.33      0.33       148
    accuracy                           0.70      1238
   macro avg       0.60      0.61      0.60      1238
weighted avg       0.71      0.70      0.71      1238
24-Mar 00:12:56 - [INFO]: !! New best loss (CrossEntropy 0.9125): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
24-Mar 00:12:56 - [INFO]: !! New highest metric (f1 0.6037): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
24-Mar 00:12:57 - [INFO]: Done processing all 1 parameters combinations.
24-Mar 00:12:57 - [INFO]: Model with smallest loss (CrossEntropy 0.9125): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
24-Mar 00:12:57 - [INFO]: Model with highest metric (f1 0.6037): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
```
Exp 2: with lexicons filter feature name: (positive|negative(combo)?|market_sentiment|polarity(combo)?)-(seqsum|seqnorm)(mean)?
- sum and token length normal. positive, negative, and combined polarity for each lexicon individ. and combined + ntusd market_sentiment
```
27-Mar 01:13:08 - [INFO]: Classification report:
              precision    recall  f1-score   support

    positive       0.81      0.73      0.77       727
    negative       0.62      0.82      0.71       363
     neutral       0.32      0.24      0.28       148

    accuracy                           0.70      1238
   macro avg       0.59      0.60      0.58      1238
weighted avg       0.70      0.70      0.69      1238
```

Exp3: all 42 economic sentiment features including match normal. and subwordlist categories
```
07-Apr 12:55:22 - [INFO]: Classification report:
              precision    recall  f1-score   support

    positive       0.81      0.83      0.82       727
    negative       0.69      0.79      0.73       363
     neutral       0.32      0.15      0.20       148

    accuracy                           0.74      1238
   macro avg       0.60      0.59      0.59      1238
weighted avg       0.71      0.74      0.72      1238

07-Apr 12:55:22 - [INFO]: !! New best loss (CrossEntropy 1.2788): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
07-Apr 12:55:22 - [INFO]: !! New highest metric (f1 0.5859): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
07-Apr 12:55:22 - [INFO]: Done processing all 1 parameters combinations.

07-Apr 12:55:22 - [INFO]: Model with smallest loss (CrossEntropy 1.2788): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
07-Apr 12:55:22 - [INFO]: Model with highest metric (f1 0.5859): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
```
Exp4: added general domain sen lex. LIWC all features
```
              precision    recall  f1-score   support

    positive       0.68      0.87      0.76       727
    negative       0.61      0.51      0.55       363
     neutral       0.00      0.00      0.00       148

    accuracy                           0.66      1238
   macro avg       0.43      0.46      0.44      1238
weighted avg       0.58      0.66      0.61      1238
```

Exp5: BEST only polarity "./lexfeatdict_pol-all.pkl"
```
08-Apr 18:37:34 - [INFO]: Classification report:
              precision    recall  f1-score   support

    positive       0.82      0.77      0.80       727
    negative       0.64      0.79      0.71       363
     neutral       0.40      0.28      0.33       148

    accuracy                           0.72      1238
   macro avg       0.62      0.62      0.61      1238
weighted avg       0.72      0.72      0.71      1238

08-Apr 18:37:34 - [INFO]: !! New best loss (CrossEntropy 1.5161): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
08-Apr 18:37:34 - [INFO]: !! New highest metric (f1 0.6113): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
08-Apr 18:37:34 - [INFO]: Done processing all 1 parameters combinations.

08-Apr 18:37:34 - [INFO]: Model with smallest loss (CrossEntropy 1.5161): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
08-Apr 18:37:34 - [INFO]: Model with highest metric (f1 0.6113): /home/gilles/repos/sentivent-implicit-economic-sentiment/src/BERTICON/output/sentivent-implicit/model.pth
```
Exp6: pol-all-seqsumseqnorm
```
09-Apr 00:49:15 - [INFO]: Classification report:
              precision    recall  f1-score   support

    positive       0.82      0.75      0.78       727
    negative       0.61      0.79      0.69       363
     neutral       0.35      0.23      0.28       148

    accuracy                           0.70      1238
   macro avg       0.59      0.59      0.58      1238
weighted avg       0.70      0.70      0.70      1238
```