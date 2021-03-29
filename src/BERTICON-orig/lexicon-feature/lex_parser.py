import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET

from sklearn import preprocessing

import spacy
from spacy.lang.nl import Dutch

nlp = spacy.load('nl_core_news_sm')

def load_data(file):
	f = open(file, 'rt', encoding='utf-8')
	lines = f.read().split('\n')
	f.close()
	return lines

def dict_to_pd(lexicon):
	for key, value in lexicon.items():
		scores = tuple([item[1] for item in value])
		lexicon[key] = scores
	lex = pd.DataFrame(list(lexicon.items()), columns=['word','scores'])
	return lex


fpaths = {
    # lexica
    'pattern': '../Lexica/pattern_new.xml',
    'duoman': '../Lexica//duoman.txt',
    'liwc': '../Lexica/liwc_emo.txt',
    'nrcemotion': '../Lexica/NRC.csv',
    'nrcvad': '../Lexica/nrcvad.txt',
    'moors': '../Lexica/moors.txt',
    'memolon': '../Lexica/memolon_nl.tsv',
    'senticnet': '../Lexica/senticnet_nl.py'
}
"""

fpaths = {
    # lexica
    'pattern': 'Lexica/pattern_new.xml',
    'duoman': 'Lexica//duoman.txt',
    'liwc': 'Lexica/liwc_emo.txt',
    'nrcemotion': 'Lexica/NRC.csv',
    'nrcvad': 'Lexica/nrcvad.txt',
    'moors': 'Lexica/moors.txt',
    'memolon': 'Lexica/memolon_nl.tsv'
}
"""




"""
keys = []
for key in senticnet.keys():
	keys.append(key.split('_'))





tokens = ['dit', 'is', 'een', 'test']
for token in tokens:
	i = tokens.index(token)
	if len(tokens) - i > 2:
		multi_token3 = '_'.join(tokens[i:i+3])
		
"""


# Lexicon1: Pattern.nl

def parse_pattern(fpath): # meestal lemma, geen MW, soms capitals
	"""
	Output: {word: [('polarity', score)]}
	"""
	pattern_lexicon = {}
	root = ET.parse('../Lexica/pattern_new.xml').getroot()
	for word in root.findall('word'):
		form = word.get('form')
		polarity = word.get('polarity')
		pattern_lexicon[form.lower()] = [('polarity', float(polarity))]
	pattern_lexicon = dict_to_pd(pattern_lexicon)
	return pattern_lexicon


# Lexicon2: DuOMAn

def parse_duoman(fpath): # lemma, een paar MW, soms capitals
	"""
	Output: {word: [('polarity', score)]}
	"""
	duoman = load_data(fpath)
	duoman_lexicon = {}
	for line in duoman:
		line = line.split('\t')
		if line[1] != '?':
			if line[1] == '--':
				duoman_lexicon[line[0][:-2].lower()] = [('sentiment', float(-2))]
			elif line[1] == '-':
				duoman_lexicon[line[0][:-2].lower()] = [('sentiment', float(-1))]
			elif line[1] == '+':
				duoman_lexicon[line[0][:-2].lower()] = [('sentiment', float(1))]
			elif line[1] == '++':
				duoman_lexicon[line[0][:-2].lower()] = [('sentiment', float(2))]
	duoman_lexicon = dict_to_pd(duoman_lexicon)
	return duoman_lexicon


# Lexicon3: LIWC

def parse_liwc(fpath): # zowel vervoegingen/-buigingen als lemma's, geen MW, geen capitals
	"""
	Output: {word: [('posemo', value), ('posfeel', value), ('optim', value), ('negemo', value), ('anger', value), ('anx', value), ('sad', value)]}
	"""
	liwc = load_data(fpath)
	liwc_lexicon = {}
	for line in liwc:
		fields = line.split('\t')
		#liwc_lexicon[fields[0]] = [('posemo', 0), ('posfeel', 0), ('optim', 0), ('negemo', 0), ('anger', 0), ('anx', 0), ('sad', 0)]
		liwc_lexicon[fields[0]] = {'posemo': 0.0, 'posfeel': 0.0, 'optim': 0.0, 'negemo': 0.0, 'anger': 0.0, 'anx': 0.0, 'sad': 0.0}
		if 'posemo' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['posemo'] += 1.0
		if 'posfeel' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['posfeel'] += 1.0
		if 'optim' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['optim'] += 1.0
		if 'negemo' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['negemo'] += 1.0
		if 'anger' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['anger'] += 1.0
		if 'anx' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['anx'] += 1.0
		if 'sad' in fields:
			previous = liwc_lexicon.get(fields[0])
			previous['sad'] += 1.0
	for key, value in liwc_lexicon.items():
		l = value.items()
		liwc_lexicon[key] =  list(l)
	liwc_lexicon = dict_to_pd(liwc_lexicon)
	return liwc_lexicon


# Lexicon4: NRC Emotion

def parse_nrcemotion(fpath): # soms verbuigingen, een paar MW, een paar capitals
	"""
	Output: {word: [('Positive', value), ('Negative', value), ('Anger', value), ('Anticipation', value), ('Disgust', value), ('Fear', value), ('Joy', value), ('Sadness', value), ('Surprise', value), ('Trust', value)]}
	"""
	nrcemotion = load_data(fpath)
	nrcemotion_lexicon = {}
	for line in nrcemotion[1:]:
		fields = line.split('\t')
		if fields[1] != 'NO TRANSLATION':
			nrcemotion_lexicon[fields[1].lower()] = [('positive', float(fields[3])), ('negative', float(fields[4])), ('anger', float(fields[5])), ('anticipation', float(fields[6])), ('disgust', float(fields[7])), ('fear', float(fields[8])), ('joy', float(fields[9])), ('sadness', float(fields[10])), ('surprise', float(fields[11])), ('trust', float(fields[12]))]
	nrcemotion_lexicon = dict_to_pd(nrcemotion_lexicon)
	return nrcemotion_lexicon


# Lexicon5: NRC VAD

def parse_nrcvad(fpath): # soms verbuigingen, een paar MW, een paar capitals
	"""
	Output: {word: [('valence', score), ('arousal', score), ('dominance', score)]}
	"""
	nrcvad = load_data(fpath)
	nrcvad_lexicon = {}
	for line in nrcvad[1:-1]:
		fields = line.split('\t')
		if fields[23] != 'NO TRANSLATION':
			nrcvad_lexicon[fields[23].lower()] = [('valence', float(fields[1])), ('arousal', float(fields[2])), ('dominance', float(fields[3]))]
	nrcvad_lexicon = dict_to_pd(nrcvad_lexicon)
	return nrcvad_lexicon

#print(parse_nrcvad(fpaths['nrcvad']))

# Lexicon6: Moors

def parse_moors(fpath): # geen verbuigingen, geen MW, geen capitals
	"""
	Output: {word: [('valence', score), ('arousal', score), ('dominance', score)]}
	"""
	moors = load_data(fpath)
	moors_lexicon = {}
	for line in moors[1:-1]:
		fields = line.split('\t')
		moors_lexicon[fields[0]] = [('valence', float(fields[1])), ('arousal', float(fields[2])), ('dominance', float(fields[3]))]
	moors_lexicon = dict_to_pd(moors_lexicon)
	return moors_lexicon

#print(parse_moors(fpaths['moors']))



# Lexicon7: MEmoLon_VAD

def parse_memolon_vad(fpath): # soms verbuigingen, een paar MW, een paar capitals
	"""
	Output: {word: [('valence', score), ('arousal', score), ('dominance', score), ('joy', score), ('anger', score), ('sadness', score), ('fear', score), ('disgust', score)]}
	"""
	memolon = load_data(fpath)
	memolon_lexicon = {}
	for line in memolon[1:]:
		fields = line.split('\t')
		memolon_lexicon[fields[0].lower()] = [('valence', float(fields[1])), ('arousal', float(fields[2])), ('dominance', float(fields[3]))]
	memolon_lexicon = dict_to_pd(memolon_lexicon)
	return memolon_lexicon


# Lexicon8: MEmoLon_cat

def parse_memolon_cat(fpath): # soms verbuigingen, een paar MW, een paar capitals
	"""
	Output: {word: [('valence', score), ('arousal', score), ('dominance', score), ('joy', score), ('anger', score), ('sadness', score), ('fear', score), ('disgust', score)]}
	"""
	memolon = load_data(fpath)
	memolon_lexicon = {}
	for line in memolon[1:]:
		fields = line.split('\t')
		memolon_lexicon[fields[0].lower()] = [('joy', float(fields[4])), ('anger', float(fields[5])), ('sadness', float(fields[6])), ('fear', float(fields[7])), ('disgust', float(fields[8]))]
	memolon_lexicon = dict_to_pd(memolon_lexicon)
	return memolon_lexicon

# Lexicon9: Senticnet

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../Lexica')

from senticnet_nl import senticnet

def parse_senticnet(): # geen verbuigingen, vrij veel MW, geen capitals
	sentic_lexicon = {}
	for key,value in senticnet.items():
			emotions = [0, 0, 0, 0, 0, 0, 0, 0] # anger, anticipation, disgust, fear, joy, sadness, surprise, trust = boosheid, interesseren, walg, schrik, vreugde, droefheid, verrassing, bewondering
			if value[4] == '#boosheid' or value[5] == '#boosheid':
				emotions[0] = 1
			if value[4] == '#interesseren' or value[5] == '#interesseren':
				emotions[1] = 1
			if value[4] == '#walg' or value[5] == '#walg':
				emotions[2] = 1
			if value[4] == '#schrik' or value[5] == '#schrik':
				emotions[3] = 1
			if value[4] == '#vreugde' or value[5] == '#vreugde':
				emotions[4] = 1
			if value[4] == '#droefheid' or value[5] == '#droefheid':
				emotions[5] = 1
			if value[4] == '#verrassing' or value[5] == '#verrassing':
				emotions[6] = 1
			if value[4] == '#bewondering' or value[5] == '#bewondering':
				emotions[7] = 1
			emotions.extend(value[:4])
			emotions.extend([value[6]])
			emotion_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'pleasentness', 'attention', 'sensitivity', 'aptitude', 'polarity']
			sentic_lexicon[key] = list(zip(emotion_names, emotions))

	sentic_lexicon = dict_to_pd(sentic_lexicon)
	return sentic_lexicon


def read_all_lexica(fpaths):
    all_lexica = pd.concat([
    	parse_pattern(fpaths['pattern']).assign(source='pattern'),
		parse_duoman(fpaths['duoman']).assign(source='duoman'),
		parse_liwc(fpaths['liwc']).assign(source='liwc'),
		parse_nrcemotion(fpaths['nrcemotion']).assign(source='nrcemotion'),
		parse_nrcvad(fpaths['nrcvad']).assign(source='nrcvad'),
		parse_moors(fpaths['moors']).assign(source='moors'),
		parse_memolon_vad(fpaths['memolon']).assign(source='memolon_vad'),
		parse_memolon_cat(fpaths['memolon']).assign(source='memolon_cat'),
		parse_senticnet().assign(source='senticnet')
	], axis=0, ignore_index=True)
    return all_lexica


lex1 = parse_pattern(fpaths['pattern'])
lex2 = parse_duoman(fpaths['duoman'])
lex3 = parse_liwc(fpaths['liwc'])
lex4 = parse_nrcemotion(fpaths['nrcemotion'])
lex5 = parse_nrcvad(fpaths['nrcvad'])
lex6 = parse_moors(fpaths['moors'])
lex7 = parse_memolon_vad(fpaths['memolon'])
lex8 = parse_memolon_cat(fpaths['memolon'])
lex9 = parse_senticnet()

#print(lex1)

"""
scores = []

for scoresset in lex7.scores:
	for score in scoresset[3:]:
		scores.append(score)
print(min(scores))
print(max(scores))
"""


"""
w1 = [w for w in lex1.word]
w2 = [w for w in lex2.word]
w3 = [w for w in lex3.word]
w4 = [w for w in lex4.word]
w5 = [w for w in lex5.word]
w6 = [w for w in lex6.word]
w7 = [w for w in lex7.word]

mutual = []
for w in w1:
	if w in w2:
		if w in w3:
			if w in w4:
				if w in w5:
					if w in w6:
						if w in w7:
							mutual.append(w)



for term in mutual:
	print(term + '\t' + str(lex1[lex1.word==term].scores.item()) + '\t' + str(lex2[lex2.word==term].scores.item()) + '\t' + str(lex3[lex3.word==term].scores.item()) + '\t' + str(lex4[lex4.word==term].scores.item()) + '\t' + str(lex5[lex5.word==term].scores.item()) + '\t' + str(lex6[lex6.word==term].scores.item()) + str(lex7[lex7.word==term].scores.item()))
"""



if __name__ == '__main__':
    pass

"""
duoman = load_data('Lexica/duoman.txt')
duoman_lexicon = []
for line in duoman[1:]:
	fields = line.split('\t')
	duoman.append(fields[0])

duoman_annais = load_data('Lexica/Duoman.csv')
duoman_lexicon_annais = []
for line in duoman_annais[1:]:
	fields = line.split('\t')
	duoman_lexicon_annais.append(fields[0])

compare = list(zip(duoman_lexicon, duoman_lexicon_annais))
for pair in compare:
	if pair[0] != pair[1]:
		print(pair)
"""

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#print(lex)

#values = []
#for score in lex.scores:
#	if score[8] != 0:
#		values.append(score[8])
#print(values)
#print(max(values))
#sns.set(color_codes=True)

scores = []

for scoresset in lex7.scores:
	for score in scoresset[:3]:
		scores.append(score)

X = np.array(scores)
sns.distplot(X)

plt.show()

#for score in lex.scores:
#	print(len(score))
"""