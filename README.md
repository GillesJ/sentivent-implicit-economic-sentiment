# 

## Exploratory Experiment notes

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
Exp 2: with lexicons
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