import numpy as np
import pickle

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams, skipgrams, FreqDist, pos_tag
from sklearn.feature_extraction import DictVectorizer

import spacy
from spacy.lang.nl import Dutch

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
  fpaths,
)

datafile = '../subtitles_cat_all_bertje.txt'



def word_ngrams(data):
	"""
	Gives a binary score for the presence or absence of word n-grams (sequences of words) in the tweet
	Args:
		data: nested list with for every instance the ID, tweet (raw string) and the label for each emotion category
	Preprocessing of tweet in function:
		tokenization
	Returns:
		feature_matrix: a list of feature vectors. The features vectors are dictionaries with a binary value for n-grams: 1 if the n-gram is present in the tweet, 0 if it isn't.
	"""
	feature_matrix = []
	for instance in data:
		text = instance[1]
		tokens = word_tokenize(text)
		vector = {}
		for n in [1,2,3]:
			grams = ngrams(tokens, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
			for gram,count in fdist.items():
				vector['w'+str(n)+'+'+' '.join(gram)] = 1
		feature_matrix.append(vector)
	return feature_matrix

def char_ngrams(data):
	"""
	Gives a binary score for the presence or absence of character n-grams (sequences of characters) in the tweet
	Args:
		data: nested list with for every instance the ID, tweet (raw string) and the label for each emotion category
	Preprocessing of tweet in function:
		none
	Returns:
		feature_matrix: a dictionary with ID's as keys and feature vectors as values. The features vectors are dictionaries n-grams: 1 if the n-gram is present in the tweet, 0 if it isn't.
	"""
	feature_matrix = []
	for instance in data:
		text = instance[1]
		vector = {}
		for n in [3,4,5]:
			grams = ngrams(text, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
			for gram,count in fdist.items():
				vector['c'+str(n)+'+'+''.join(gram)] = 1
		feature_matrix.append(vector)
	return feature_matrix



def gen_data(fpath, sent_data, score_fn):
	f = open(fpath, 'rt', encoding = 'utf-8')
	data = f.read().split('\n')
	data = data[1:-1]
	data = [instance.split('\t') for instance in data] # ['ID', 'Text', 'emotion']
	f.close()

	sentences = [instance[1].rstrip() for instance in data]
	n = len(data)
	i = 0

	y = np.zeros((n, 1))
	x = np.zeros((n, len(score_fn('good bad', sent_data))))
	bow = word_ngrams(data)
	cow = char_ngrams(data)
	
	for instance in data:
		score = score_fn(instance[1], sent_data) # instance[1] = text
		x[i] = score
		i += 1
	return sentences, x, bow, cow

def score_sent(text, sent_data, normalize=False):
	"""
	Evaluate the data
	"""
	test_sent = next(iter(sent_data.values()))
	sents = np.zeros_like(test_sent).astype(np.float).reshape(-1)

	tokens = []
	doc = nlp(text)
	for tok in doc:
		tokens.append(tok.lemma_)

	for token in tokens:
		try:
			sent = np.array(sent_data[token])
		except KeyError:
			continue
			
		if normalize:
			sent = sent / sent.sum()

		sents += sent

	# if statement zelf toegevoegd
	if len(tokens) > 0:
		score = sents / len(tokens)
	else:
		score = 0
	return score


def read_lexica():
	sent_to_dict = lambda x: x.set_index("word")["scores"].to_dict()

	sentiments = {
		'pattern': sent_to_dict(parse_pattern(fpaths['pattern'])),
		'duoman': sent_to_dict(parse_duoman(fpaths['duoman'])),
		'liwc': sent_to_dict(parse_liwc(fpaths['liwc'])),
		'nrcemotion': sent_to_dict(parse_nrcemotion(fpaths['nrcemotion'])),
		'nrcvad': sent_to_dict(parse_nrcvad(fpaths['nrcvad'])),
		'moors': sent_to_dict(parse_moors(fpaths['moors'])),
		'memolon_vad': sent_to_dict(parse_memolon_vad(fpaths['memolon'])),
		'memolon_cat': sent_to_dict(parse_memolon_cat(fpaths['memolon']))
	}

	return sentiments


def score_sentences(sentiments):
	dv = DictVectorizer(sparse=False)
	
	data_dict = {}

	for lexicon in sentiments:
		scorer = lambda text, sent_data, : score_sent(text, sent_data, normalize=False)
		sentences, x_lex, x_bow, x_cow = gen_data(datafile, sentiments[lexicon], scorer)
		data_dict['sentences'] = sentences
		data_dict[lexicon] = x_lex
		data_dict['bow'] = dv.fit_transform(x_bow)
		data_dict['cow'] = dv.fit_transform(x_cow)

	return data_dict


def make_combined_score(sentiments, data_dict):
	data_dict['combined'] = np.hstack([data_dict[lexicon] for lexicon in sentiments])
	sentiments['combined'] = None  # dummy such that it's included in iterations

	return sentiments, data_dict

	
def get_feat_vec(sentiments, data_dict):
	lexicon = 'combined'
	x_feat = data_dict[lexicon]
	x_sentences = data_dict['sentences']
	#x_bow = data_dict['bow']
	#x_cow = data_dict['cow']
	#x = np.concatenate((x, x_bow, x_cow), axis=1)
	#print(x.shape)
	#print(x)
	x = {}
	for i in range(len(x_feat)):
		x[x_sentences[i]] = x_feat[i]
	return x

def save_object(obj, fpath):
	"""
	Pickle an object and save it to file
	"""
	with open(fpath, 'wb') as o:
		pickle.dump(obj, o)

def load_object(fpath):
	"""
	Load a pickled object from file
	"""
	with open(fpath, 'rb') as i:
		return pickle.load(i)


if __name__ == '__main__':
	
	sentiments = read_lexica()
	data_dict = score_sentences(sentiments)
	sentiments, data_dict = make_combined_score(sentiments, data_dict)
	#feat_vec = get_feat_vec(sentiments, data_dict)
	feat_dict = get_feat_vec(sentiments, data_dict)
	print(feat_dict)
	#save_object(feat_dict, 'feat_dict_subtitles.pkl')
	"""


	feat_dict = load_object('feat_dict_subtitles.pkl')
	feat_vec = feat_dict['Ik heb euh ooit es… bja ik swing op elk feestje. Me iedereen da wilt swingen. Ik doe da supergraag.']
	print(feat_vec)
	#print(feat_dict.keys())
	"""
