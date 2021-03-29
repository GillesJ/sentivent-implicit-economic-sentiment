from scipy.stats import pearsonr
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import numpy as np

def load(file):
	f = open(file, 'rt', encoding='utf-8')
	lines = f.read().split('\n')
	lines = lines[1:-1]
	return lines

def cat_metrics(f_true, f_pred):
	true = load(f_true)
	pred = load(f_pred)
	true_labels = [instance.split('\t')[2] for instance in true]
	#true_labels = ['3' if instance == '4' else instance for instance in true_labels]
	pred_labels = [instance.split(',')[1] for instance in pred]
	#pred_labels = ['3' if instance == '4' else instance for instance in pred_labels]
	acc = np.mean(np.asarray(true_labels) == np.asarray(pred_labels))
	micro_f1 = f1_score(true_labels, pred_labels, average='micro')
	macro_f1 = f1_score(true_labels, pred_labels, average='macro')
	return acc, micro_f1, macro_f1

def vad_metrics(f_true, f_pred):
	true = load(f_true)
	pred = load(f_pred)
	true_labels_v = [float(instance.split('\t')[2].split(',')[0]) for instance in true]
	pred_labels_v = [float(instance.split(',')[1].strip('"')) for instance in pred]
	true_labels_a = [float(instance.split('\t')[2].split(',')[1]) for instance in true]
	pred_labels_a = [float(instance.split(',')[2].strip('"')) for instance in pred]
	true_labels_d = [float(instance.split('\t')[2].split(',')[2]) for instance in true]
	pred_labels_d = [float(instance.split(',')[3].strip('"')) for instance in pred]
	r_v = pearsonr(true_labels_v, pred_labels_v)[0]
	r_a = pearsonr(true_labels_a, pred_labels_a)[0]
	r_d = pearsonr(true_labels_d, pred_labels_d)[0]
	r = (r_v + r_a + r_d)/3
	return r_v, r_a, r_d, r

def cat_confusion_matrix(f_true, f_pred):
	true = load(f_true)
	pred = load(f_pred)
	true_labels = [instance.split('\t')[2] for instance in true]
	pred_labels = [instance.split(',')[1] for instance in pred]
	matrix = confusion_matrix(true_labels, pred_labels)
	return matrix

def vad_to_cat(labels):
	new_labels = []
	for instance in labels:
		if instance <= 0.2:
			new_labels.append('very_low')
		elif 0.2 < instance <= 0.4:
			new_labels.append('low')
		elif 0.4 < instance <= 0.6:
			new_labels.append('medium')
		elif 0.6 < instance <= 0.8:
			new_labels.append('high')
		elif 0.8 < instance:
			new_labels.append('very_high')
	return new_labels

def vad_to_cat2(labels):
	new_labels = []
	for instance in labels:
		if instance <= 0.4:
			new_labels.append('low')
		elif 0.4 < instance <= 0.6:
			new_labels.append('medium')
		elif 0.6 < instance:
			new_labels.append('high')
	return new_labels

def vad_confusion_matrix(f_true, f_pred):
	true = load(f_true)
	pred = load(f_pred)
	true_labels_v = vad_to_cat([float(instance.split('\t')[2].split(',')[0]) for instance in true])
	pred_labels_v = vad_to_cat([float(instance.split(',')[1].strip('"')) for instance in pred])
	true_labels_a = vad_to_cat([float(instance.split('\t')[2].split(',')[1]) for instance in true])
	pred_labels_a = vad_to_cat([float(instance.split(',')[2].strip('"')) for instance in pred])
	true_labels_d = vad_to_cat([float(instance.split('\t')[2].split(',')[2]) for instance in true])
	pred_labels_d = vad_to_cat([float(instance.split(',')[3].strip('"')) for instance in pred])
	matrix_v = confusion_matrix(true_labels_v, pred_labels_v)
	matrix_a = confusion_matrix(true_labels_a, pred_labels_a)
	matrix_d = confusion_matrix(true_labels_d, pred_labels_d)
	return matrix_v, matrix_a, matrix_d

def vad_to_cat_metrics(f_true, f_pred):
	true = load(f_true)
	pred = load(f_pred)[800:]
	true_labels = [instance.split('\t')[2] for instance in true]
	true_labels = ['3' if instance == '4' else instance for instance in true_labels]
	pred_labels_v = vad_to_cat2([float(instance.split(',')[1].strip('"')) for instance in pred])
	pred_labels_a = vad_to_cat2([float(instance.split(',')[2].strip('"')) for instance in pred])
	pred_labels_d = vad_to_cat2([float(instance.split(',')[3].strip('"')) for instance in pred])
	pred_labels = list(zip(pred_labels_v, pred_labels_a, pred_labels_d))
	cat_pred_labels = []
	for label in pred_labels:
		if label == ('low', 'high', 'high'):
			new_label = '1' # anger
		elif label == ('low', 'high', 'low'):
			new_label = '2' #fear
		elif label[0] == 'high':
			new_label = '3' #joy
		elif label == ('low', 'low', 'low'):
			new_label = '5' #sadness
		else:
			new_label = '0' #neutral
		cat_pred_labels.append(new_label)
	acc = np.mean(np.asarray(true_labels) == np.asarray(cat_pred_labels))
	micro_f1 = f1_score(true_labels, cat_pred_labels, average='micro')
	macro_f1 = f1_score(true_labels, cat_pred_labels, average='macro')
	return acc, micro_f1, macro_f1

def cost_corr_acc(f_true, f_pred):
	conf_m = np.array(cat_confusion_matrix(f_true, f_pred))
	cost_m = np.array([[0, 2/3, 2/3, 2/3, 2/3, 2/3], [2/3, 0, 1/3, 1, 1, 1/3], [2/3, 1/3, 0, 1, 1, 1/3], [2/3, 1, 1, 0, 1/3, 1], [2/3, 1, 1, 1/3, 0, 1], [2/3, 1/3, 1/3, 1, 1, 0]])
	cost = np.sum(np.multiply(conf_m, cost_m))/np.sum(conf_m)
	ccacc = 1 - cost
	return ccacc


dataset = 'captions_vad'


if dataset == 'tweets_cat':
	test_labels = 'Data/tweets_cat_test_bertje.txt'
	predictions = 'predictions/predictions_tweets_cat/robbert_withlex_max_test.txt'

elif dataset == 'captions_cat':
	test_labels = 'Data/subtitles_cat_test_bertje.txt'
	predictions = 'predictions/predictions_subtitles_cat/robbert_withlex_max_test.txt'

elif dataset == 'tweets_vad':
	test_labels = 'Data/tweets_vad_test_bertje.txt'
	#test_labels2 = '../Data/Tweets-cat/tweets_cat_test_bertje.txt'
	predictions = 'predictions/predictions_tweets_vad/robbert_withlex_max_test.txt'

elif dataset == 'captions_vad':
	test_labels = 'Data/subtitles_vad_test_bertje.txt'
	#test_labels2 = '../Data/Subtitles-cat/subtitles_cat_test_bertje.txt'
	predictions = 'predictions/predictions_subtitles_vad/robbert_withlex_max_test.txt'



print(dataset)


if dataset in ['tweets_cat', 'captions_cat']:
	print(cat_metrics(test_labels, predictions))
	print(cat_confusion_matrix(test_labels, predictions))
	print(cost_corr_acc(test_labels, predictions))

elif dataset in ['tweets_vad', 'captions_vad']:
	print(vad_metrics(test_labels, predictions))
	print(vad_confusion_matrix(test_labels, predictions)[0])
	print(vad_confusion_matrix(test_labels, predictions)[1])
	print(vad_confusion_matrix(test_labels, predictions)[2])


