import torch
from torch.utils.data import Dataset, DataLoader
import random
from progressbar import ProgressBar
import os
import sys
sys.setrecursionlimit(100000)
import pickle
from collections import defaultdict
import numpy as np
from glob import glob
import json
import pdb
import string

rem_punc = string.punctuation.replace('\'','').replace('_', '').replace('?', '').replace('@', '')
table = str.maketrans('', '', rem_punc)

def tokenize_text(raw_text):
	without_punc = raw_text.translate(table)
	words = without_punc.split()
	return words

proof_steps = glob(os.path.join('processed/proof_steps', 'train/*.pickle')) + \
                               glob(os.path.join('processed/proof_steps', 'valid/*.pickle'))

proofs = {}
vocab = {'<pad>': 0, '<start>': 1}

print(len(proof_steps))
print("Collecting and rewriting proofs from steps")
for idx in range(len(proof_steps)):
	f = open(proof_steps[idx], 'rb')
	proof_step = pickle.load(f)
	f.close()
	key = (proof_step['file'], proof_step['proof_name'])
	f_pickle_name = key[0][:-5]+'.p'
	gal_words = []
	if os.path.isfile(f_pickle_name):
		f_pickle = open(f_pickle_name , 'rb')
		gal_dict = pickle.load(f_pickle)
		f_pickle.close()
		command = proof_step['tactic']['text']
		if key[1] in gal_dict:
			if command in gal_dict[key[1]]:
				raw_gal = gal_dict[key[1]][command]
				gal_words = tokenize_text(raw_gal)
				for word in gal_words:
					if word not in vocab:
						vocab[word] = len(vocab)
	proof_step['gal_tokens'] = gal_words
	new_file_name = os.path.join('postprocessed', proof_steps[idx])
	new_f = open(new_file_name, 'wb')
	pickle.dump(proof_step, new_f)
	new_f.close()

print(len(vocab))
pickle.dump(vocab, open("gal_vocab.pickle", 'wb'))