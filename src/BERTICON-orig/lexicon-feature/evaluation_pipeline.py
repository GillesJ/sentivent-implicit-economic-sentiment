import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams, skipgrams, FreqDist, pos_tag

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report
from sklearn.feature_extraction import DictVectorizer

from scipy.stats import pearsonr

import spacy
from spacy.lang.nl import Dutch
from spacy.tokenizer import Tokenizer

nlp = spacy.load('nl_core_news_sm')

from lex_parser import (
  parse_pattern,
  parse_duoman,
  parse_liwc,
  parse_nrcemotion,
  parse_nrcvad,
  parse_moors,
  parse_memolon_vad,
  parse_memolon_cat,
  parse_senticnet,
  fpaths,
)


def cross_val_files(fpath):
	f = open(fpath, encoding='utf-8')
	lines = f.read().split('\n')
	lines = lines[1:-1]
	f.close()

	#print(lines[0])
	#print(lines[-1])

	folds = []
	fold1 = lines[:100]
	fold2 = lines[100:200]
	fold3 = lines[200:300]
	fold4 = lines[300:400]
	fold5 = lines[400:500]
	fold6 = lines[500:600]
	fold7 = lines[600:700]
	fold8 = lines[700:800]
	fold9 = lines[800:900]
	fold10 = lines[900:1000]
	folds.extend([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10])
	return folds


def read_lexica():
	sent_to_dict = lambda x: x.set_index("word")["scores"].to_dict()
	lexica = {
		'pattern': sent_to_dict(parse_pattern(fpaths['pattern'])),
		'duoman': sent_to_dict(parse_duoman(fpaths['duoman'])),
		'liwc': sent_to_dict(parse_liwc(fpaths['liwc'])),
		'nrcemotion': sent_to_dict(parse_nrcemotion(fpaths['nrcemotion'])),
		'nrcvad': sent_to_dict(parse_nrcvad(fpaths['nrcvad'])),
		'moors': sent_to_dict(parse_moors(fpaths['moors'])),
		'memolon_vad': sent_to_dict(parse_memolon_vad(fpaths['memolon'])),
		'memolon_cat': sent_to_dict(parse_memolon_cat(fpaths['memolon'])),
		'senticnet': sent_to_dict(parse_senticnet())
	}

	return lexica


def lexicon_score(sentence, lexicon, lexica):
	"""
	Gives a binary score for the presence or absence of word n-grams (sequences of words) in the tweet
	Args:
		data: nested list with for every instance the ID, tweet (raw string) and the label for each emotion category
	Preprocessing of tweet in function:
		tokenization
	Returns:
		feature_matrix: a list of feature vectors. The features vectors are dictionaries with a binary value for n-grams: 1 if the n-gram is present in the tweet, 0 if it isn't.
	"""
	#sent_to_dict = lambda x: x.set_index("word")["scores"].to_dict()
	feature_matrix = []
	tokens = []
	lemmas = []
	tokens_with_sent = 0
	lexicon_vector = np.zeros(len(list(lexica[lexicon].values())[0]))
	
	doc = nlp(sentence)
	for tok in doc:
		tokens.append(tok.lower_)
		lemmas.append(tok.lemma_)

	if lexicon == 'senticnet':
		i = -1 		# start at -1 so the loop can start with index 0
		for token in tokens:
			#i = tokens.index(token)
			if 'multi_token7' in locals():
				del multi_token7
			if 'multi_token6' in locals():
				del multi_token6
			if 'multi_token5' in locals():
				del multi_token5
			if 'multi_token4' in locals():
				del multi_token4
			if 'multi_token3' in locals():
				del multi_token3
			if 'multi_token2' in locals():
				del multi_token2


			i += 1
	
			if len(tokens) - i > 6:
				multi_token7 = '_'.join(tokens[i:i+7])
				try:
					lex_score = np.array(lexica[lexicon][multi_token7])
					lexicon_vector += lex_score
					tokens_with_sent += 1
					i += 6
				except KeyError:
					pass
	
			if ('multi_token7' in locals() and multi_token7 not in lexica[lexicon].keys()) or 'multi_token7' not in locals():
				if len(tokens) - i > 5:
					multi_token6 = '_'.join(tokens[i:i+6])
					try:
						lex_score = np.array(lexica[lexicon][multi_token6])
						lexicon_vector += lex_score
						tokens_with_sent += 1
						i += 5
					except KeyError:
						pass
	
				if ('multi_token6' in locals() and multi_token6 not in lexica[lexicon].keys()) or 'multi_token6' not in locals():
					if len(tokens) - i > 4:
						multi_token5 = '_'.join(tokens[i:i+5])
						try:
							lex_score = np.array(lexica[lexicon][multi_token5])
							lexicon_vector += lex_score
							tokens_with_sent += 1
							i += 4
						except KeyError:
							pass
						
	
					if ('multi_token5' in locals() and multi_token5 not in lexica[lexicon].keys()) or 'multi_token5' not in locals():
						if len(tokens) - i > 3:
							multi_token4 = '_'.join(tokens[i:i+4])
							try:
								lex_score = np.array(lexica[lexicon][multi_token4])
								lexicon_vector += lex_score
								tokens_with_sent += 1
								i += 3
							except KeyError:
								pass
							
	
						if ('multi_token4' in locals() and multi_token4 not in lexica[lexicon].keys()) or 'multi_token4' not in locals():
							if len(tokens) - i > 2:
								multi_token3 = '_'.join(tokens[i:i+3])
								try:
									lex_score = np.array(lexica[lexicon][multi_token3])
									lexicon_vector += lex_score
									tokens_with_sent += 1
									i += 2
								except KeyError:
									pass
	
	
							if ('multi_token3' in locals() and multi_token3 not in lexica[lexicon].keys()) or 'multi_token3' not in locals():
								if len(tokens) - i > 1:
									if len(tokens) - i == 2:
										multi_token2 = '_'.join(tokens[i:])
									else:
										multi_token2 = '_'.join(tokens[i:i+2])
									try:
										lex_score = np.array(lexica[lexicon][multi_token2])
										lexicon_vector += lex_score
										tokens_with_sent += 1
										i += 1
									except KeyError:
										pass
									
	
								if ('multi_token2' in locals() and multi_token2 not in lexica[lexicon].keys()) or 'multi_token2' not in locals():		
									if i < len(lemmas):
										try:
											lex_score = np.array(lexica[lexicon][lemmas[i]])
											lexicon_vector += lex_score
											tokens_with_sent += 1
										except KeyError:
											pass
	else:
		for lemma in lemmas:
			try:
				lex_score = np.array(lexica[lexicon][lemma])
				lexicon_vector += lex_score
				tokens_with_sent += 1
			except KeyError:
				pass

	"""
	if tokens_with_sent > 0:
		#lexicon_vector = lexicon_vector / tokens_with_sent
		lexicon_vector = np.concatenate((lexicon_vector, lexicon_vector / tokens_with_sent))
	else:
		lexicon_vector = np.concatenate((lexicon_vector, lexicon_vector))
	"""
	if len(lemmas) > 0:
		lexicon_vector = lexicon_vector / len(lemmas)
	
	return lexicon_vector

#def score_sentences(fpath, lexica):


def gen_tweets_cat_data(data, lexicon, lexica):
	#f = open(fpath, 'rt', encoding = 'utf-8')
	#data = f.read().split('\n')
	#data = data[1:-1]
	#data = [instance.split('\t') for instance in data] # ['ID', 'Text', 'emotion']
	#f.close()
	data = [instance.split('\t') for instance in data] # ['ID', 'Text', 'emotion']

	n = len(data)
	i = 0

	y = np.zeros((n, 1))
	x = np.zeros((n, len(lexicon_score('goed', lexicon, lexica))))
	
	for instance in data:
		lexicon_vector = lexicon_score(instance[1], lexicon, lexica) # instance[1] = text
		x[i] = lexicon_vector
		if instance[2] == '1': # instance[2] = label (anger)
			y[i] = 1
		elif instance[2] == '2': # instance[2] = label (fear)
			y[i] = 2
		elif instance[2] == '3': # instance[2] = label (joy)
			y[i] = 3
		elif instance[2] == '4': # instance[2] = label (love)
			y[i] = 4
		elif instance[2] == '5': # instance[2] = label (sadness)
			y[i] = 5
		elif instance[2] == '0': # instance[2] = label (neutral)
			y[i] = 0
		i += 1
		#print('Sentence ' + str(i) + 'is scored.')
	
	return x, y

def gen_tweets_cat_data_all_lexica(data, lexica):
	#f = open(fpath, 'rt', encoding = 'utf-8')
	#data = f.read().split('\n')
	#data = data[1:-1]
	#data = [instance.split('\t') for instance in data] # ['ID', 'Text', 'emotion']
	#f.close()
	data = [instance.split('\t') for instance in data] # ['ID', 'Text', 'emotion']

	n = len(data)
	i = 0

	data_dict = {}

	y = np.zeros((n, 1))
	for instance in data:
		if instance[2] == '1': # instance[2] = label (anger)
			y[i] = 1
		elif instance[2] == '2': # instance[2] = label (fear)
			y[i] = 2
		elif instance[2] == '3': # instance[2] = label (joy)
			y[i] = 3
		elif instance[2] == '4': # instance[2] = label (love)
			y[i] = 4
		elif instance[2] == '5': # instance[2] = label (sadness)
			y[i] = 5
		elif instance[2] == '0': # instance[2] = label (neutral)
			y[i] = 0
		i += 1
	data_dict['y'] = y
	
	for lexicon in lexica.keys():
		i = 0
		x = np.zeros((n, len(lexicon_score('goed', lexicon, lexica))))
		for instance in data:
			lexicon_vector = lexicon_score(instance[1], lexicon, lexica) # instance[1] = text
			x[i] = lexicon_vector
			i += 1
		data_dict[lexicon] = x

	return data_dict


def eval(train, test, lexicon, lexica):

	train_data = gen_tweets_cat_data_all_lexica(train, lexica)
	test_data = gen_tweets_cat_data_all_lexica(test, lexica)
	print('Data is loaded.')
	eval_name = 'tweets_cat'

	dims = {'tweets_cat': 6, 'tweets_vad': 3, 'subtitles_cat': 6, 'subtitles_vad': 3}

	print(f'{eval_name} accuracy')

	if lexicon == 'all':
		i = 0
		for lex in lexica:
			if i == 0:
				x_train = train_data[lex]
				x_dev = test_data[lex]
			else:
				x_train = np.concatenate((x_train, train_data[lex]), axis=1)
				x_dev = np.concatenate((x_dev, test_data[lex]), axis=1)
			i += 1
	else:
		x_train = train_data[lexicon]
		x_dev = test_data[lexicon]
	y_train = train_data['y']
	y_dev = test_data['y']

	accuracies = []
	final_label = []
	if eval_name in ['tweets_vad', 'subtitles_vad']:
		dimensions = dims[eval_name]
		
		y_train_dict = {}
		for i in range(dimensions):
			labels = []
			for label in y_train:
				labels.append(label[i])
			y_train_dict['y_train_' + str(i)] = labels

		y_dev_dict = {}
		for i in range(dimensions):
			labels = []
			for label in y_dev:
				labels.append(label[i])
			y_dev_dict['y_dev_' + str(i)] = labels

		for i in range(dimensions):
			#classifier = LinearRegression()
			classifier = SVR()
			classifier.fit(x_train, y_train_dict['y_train_' + str(i)])
			pred = classifier.predict(x_dev)
			metric = pearsonr(y_dev_dict['y_dev_' + str(i)], pred)[0]
			#metric = mean_absolute_percentage_error(y_dev_dict['y_dev_' + str(i)], pred)
			#metric = np.mean(pred == y_dev_dict['y_dev_' + str(i)])
			accuracies.append(metric)
			final_label.append(pred)

			print(f'{lexicon:15}{metric:0.3f}')

		final_label = np.asarray(final_label).T
		#print(final_label)
		#print(accuracies)
		avg_accuracy = np.mean(accuracies)
		print('Avg accuracy is ' + str(avg_accuracy))

	if eval_name in ['tweets_cat', 'subtitles_cat']:
		#classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
		classifier = LinearSVC(random_state = 7)
		classifier.fit(np.asarray(x_train), np.asarray(y_train))
		pred = classifier.predict(np.asarray(x_dev))
		metric = np.mean(pred == y_dev)
		print(f'{lexicon:15}{metric:0.3f}')
		#print(lexicon)
		micro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='micro')
		macro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='macro')
		print('Micro-F1 is ' + str(micro_f1))
		print('Macro-F1 is ' + str(macro_f1))
		print(classification_report(y_dev, pred))
		return y_dev, pred
	print('\n')

def eval_all(train, test, lexica):

	train_data = gen_tweets_cat_data_all_lexica(train, lexica)
	test_data = gen_tweets_cat_data_all_lexica(test, lexica)
	print('Data is loaded.')
	eval_name = 'tweets_cat'

	dims = {'tweets_cat': 6, 'tweets_vad': 3, 'subtitles_cat': 6, 'subtitles_vad': 3}

	print(f'{eval_name} accuracy')

	
	i = 0
	for lex in list(lexica.keys()):
		if i == 0:
			train_lex = train_data[lex]
			dev_lex = test_data[lex]
		else:
			train_lex = np.concatenate((train_lex, train_data[lex]), axis=1)
			dev_lex = np.concatenate((dev_lex, test_data[lex]), axis=1)
		i += 1
	train_data['all'] = train_lex
	test_data['all'] = dev_lex
	
	y_train = train_data['y']
	y_dev = test_data['y']

	true_dict = {}
	pred_dict = {}
	for lexicon in (list(lexica.keys()) + ['all']):
		print(lexicon)
		x_train = train_data[lexicon]
		x_dev = test_data[lexicon]
	

		accuracies = []
		final_label = []
		if eval_name in ['tweets_vad', 'subtitles_vad']:
			dimensions = dims[eval_name]
			
			y_train_dict = {}
			for i in range(dimensions):
				labels = []
				for label in y_train:
					labels.append(label[i])
				y_train_dict['y_train_' + str(i)] = labels
	
			y_dev_dict = {}
			for i in range(dimensions):
				labels = []
				for label in y_dev:
					labels.append(label[i])
				y_dev_dict['y_dev_' + str(i)] = labels
	
			for i in range(dimensions):
				#classifier = LinearRegression()
				classifier = SVR()
				classifier.fit(x_train, y_train_dict['y_train_' + str(i)])
				pred = classifier.predict(x_dev)
				metric = pearsonr(y_dev_dict['y_dev_' + str(i)], pred)[0]
				#metric = mean_absolute_percentage_error(y_dev_dict['y_dev_' + str(i)], pred)
				#metric = np.mean(pred == y_dev_dict['y_dev_' + str(i)])
				accuracies.append(metric)
				final_label.append(pred)
	
				print(f'{lexicon:15}{metric:0.3f}')
	
			final_label = np.asarray(final_label).T
			#print(final_label)
			#print(accuracies)
			avg_accuracy = np.mean(accuracies)
			print('Avg accuracy is ' + str(avg_accuracy))
	
		if eval_name in ['tweets_cat', 'subtitles_cat']:
			#classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
			classifier = LinearSVC(random_state = 7)
			classifier.fit(np.asarray(x_train), np.asarray(y_train))
			pred = classifier.predict(np.asarray(x_dev))
			print(lexicon)
			micro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='micro')
			macro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='macro')
			print('Micro-F1 is ' + str(micro_f1))
			print('Macro-F1 is ' + str(macro_f1))
			print(classification_report(y_dev, pred))
			f.write(str(micro_f1) + '\t' + str(macro_f1) + '\t')
			pred_dict[lexicon] = pred.tolist()
			true_dict[lexicon] = y_dev.tolist()
		print('\n')
	return true_dict, pred_dict

def cat_metrics(true_labels, pred):
	acc = np.mean(np.asarray(true_labels) == np.asarray(pred_labels))
	micro_f1 = f1_score(true_labels, pred_labels, average='micro')
	macro_f1 = f1_score(true_labels, pred_labels, average='macro')
	return acc, micro_f1, macro_f1

if __name__ == '__main__':
	#sentence = 'ik.'
	lexica = read_lexica()
	print('Ready to evaluate')
	fpath = '../Data/Tweets-cat/tweets_cat_all_bertje.txt'
	f.open('results_tweets_cat_meantokens.txt', 'w')
	true_all = []
	pred_all = {}
	for i in range(10):
		testfold = [line for line in cross_val_files(fpath)[i]]
		if i == 0:
			trainfold = [line for fold in (cross_val_files(fpath)[i+1:]) for line in fold]
		elif i == 9:
			trainfold = [line for fold in (cross_val_files(fpath)[:i]) for line in fold]
		else:
			trainfold = [line for fold in (cross_val_files(fpath)[:i] + cross_val_files(fpath)[i+1:]) for line in fold]
		print('On testfold ' + str(i+1))
	#print(lexicon_score(sentence, 'senticnet', lexica))
	#print(gen_tweets_cat_data_all_lexica(tweets_cat_test, lexica))
		true_dict, pred_dict = eval_all(trainfold, testfold, lexica)
		f.write('\n')
		for lexicon, pred in pred_dict.items():
			if lexicon not in pred_all.keys():
				pred_all[lexicon] = pred
			else:
				#pred_all[lexicon] = np.concatenate((pred_all[lexicon], pred), axis=1)
				pred_all[lexicon].extend(pred)
		for element in true_dict['all']:
			true_all.extend(element)
	for lexicon, pred_labels in pred_all.items():
		print('Total Micro-F1 ' + lexicon + ':')
		acc, micro_f1, macro_f1 = cat_metrics(true_all, pred_labels)
		print(cat_metrics(true_all, pred_labels))
		f.write(str(micro_f1) + '\t' + str(macro_f1) + '\t')
	f.close()
