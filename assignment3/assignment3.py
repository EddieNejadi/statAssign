from __future__ import division

'''
Created on Mars 18, 2014

Lab assignment 3

@author: Mahdi Abdinejadi
@version: 1.0
'''


'''
import libraries
'''
import math
import nltk
from math import log10


'''
Function definitions
'''

def run():
	all_docs = read_tagged_corpus(corpus_file)
	training, testing = split_data(all_docs)
	tagger = train_nltk_baseline(training)
	print 'Accuracy of train_nltk_baseline is: %4.2f%%' % (100.0 * tagger.evaluate(testing))
	seen_words = set([w for sent in training for w,_t in sent])
	tagger_data = hmm_train_tagger(training)
	w_tags = []
	errors = []
	testing_size = 0
	not_seen = 0
	not_seen_error = 0
	for sent in testing:
		w_tags = hmm_tag_sentence(tagger_data, sent)
		for i, (w,t) in enumerate(w_tags):
			testing_size += 1
			if w not in seen_words: not_seen += 1
			if t != sent[i][1]:
				errors.append(sent[i][0])
				if sent[i][0] not in seen_words : not_seen_error +=1
	accuracy = (testing_size - len(errors)) / testing_size
	print 'Accuracy of my tagger is: %4.2f%%' % (100.0 * accuracy)
	accuracy_not_seen = (not_seen - not_seen_error) / not_seen
	print 'Accuracy of my tagger for not seen words is: %4.2f%%' % (100.0 * accuracy_not_seen)
	print len(errors)


def split_data(all_docs):
	""" Split the data to taring part 80% and 
		testing part 20% 

		all_docs is a list
	"""
	split_point = int(0.8*len(all_docs))
	results = []
	train_docs = all_docs[:split_point]
	eval_docs = all_docs[split_point:]
	return (train_docs, eval_docs)


# functions to read the corpus

def read_tagged_sentence(f):
	line = f.readline()
	if not line:
		return None
	sentence = []
	while line and (line != "\n"):
		line = line.strip().decode("utf-8")
		word, tag = line.split("\t", 2)
		sentence.append( (word, tag) )
		line = f.readline()
	return sentence

def read_tagged_corpus(filename):
	sentences = []
	with open(filename) as f:
		sentence = read_tagged_sentence(f)
		while sentence:
			sentences.append(sentence)
			sentence = read_tagged_sentence(f)
	return sentences

# baseline using NLTK

def most_common_tag(tagged_sentences):
	tags = {}
	for sentence in tagged_sentences:
		for _, tag in sentence:
			tags[tag] = tags.get(tag, 0) + 1
	return max(tags, key=tags.get)

def train_nltk_baseline(tagged_sentences):
	backoff_tagger = nltk.DefaultTagger(most_common_tag(tagged_sentences))
	return nltk.UnigramTagger(tagged_sentences, backoff=backoff_tagger)


# skeleton for the bigram tagger code


def hmm_train_tagger(tagged_sentences):
	dic = {"wt": {}, "t": {}, "tt" :{}}
	fdist = nltk.FreqDist(word.lower() for sentence in tagged_sentences for word,_t in sentence)
	hapaxes = set(fdist.hapaxes())
	# print len(hapaxes)
	for sent in tagged_sentences:
		add_one(dic["t"], START)
		add_one(dic["wt"], (START, START))
		last_tag = START

		for w,t in sent:
			add_one(dic["wt"], (w,t))
			add_one(dic["t"], t)
			add_one(dic["tt"], (last_tag,t))
			if w in hapaxes: 
				add_one(dic["wt"], (u"<UNKNOWN>", t))
			last_tag = t
		
		add_one(dic["t"], END)
		add_one(dic["wt"], (END, END))
		add_one(dic["tt"], (last_tag, END))

	emission = {}
	transition = {}
	tmp_min = 0.0
	for w,t in dic["wt"]:
		# P(w|t)=count(w,t)/count(t)
		emission[(w,t)] = log10( dic["wt"][(w,t)] / dic["t"][t])
		if tmp_min > emission[(w,t)] : tmp_min = emission[(w,t)]
	emission[(u"<MIN>",u"<MIN>")] = tmp_min
	
	any_tag_sum = sum([dic["t"][t] for t in dic["t"]])
	tmp_min = 0.0
	for t1,t2 in dic["tt"]:
		lamda1 = dic["t"][t2] / any_tag_sum
		lamda2 = dic["tt"][(t1,t2)] / dic["t"][t1]
		transition[(t1,t2)] = log10(lamda2 + lamda1)
		if tmp_min > transition[(t1,t2)] : tmp_min = transition[(t1,t2)]
	transition[(u"<MIN>",u"<MIN>")] = tmp_min


	return (emission, transition, dic["t"].keys() )


def hmm_tag_sentence(tagger_data, sentence):
	lst = viterbi(tagger_data, sentence)
	sq = find_sequences(tagger_data,lst)
	best_seq = fbs(sq, tagger_data[0][(u"<MIN>",u"<MIN>")] + tagger_data[1][(u"<MIN>",u"<MIN>")])
	return zip([w for w,_t in sentence], [t for t,_l in best_seq])

def viterbi(tagger_data, sentence):
	# make a dummy item with a START tag, no predecessor, and log probability 0
	current_list = [[(START, 0.0)]]

	emission, _transition, all_tags = tagger_data

	for w,_t in sentence:
		tmp = []	
		w_tags = [t for t in all_tags if (w,t) in emission]
		# For empty list, considering <UNKNOWN> word 
		if not w_tags:
			for tag in all_tags:
				if (u"<UNKNOWN>", tag) in emission:
					tmp.append((tag, emission[(u"<UNKNOWN>",tag)]))
		else:
			for w_tag in w_tags:
				tmp.append((w_tag, emission[(w,w_tag)]))
		current_list.append(tmp)
	current_list.append([(END, 0.0)])

	return current_list

def find_sequences(tagger_data, words_tags):
	emission, transition, all_tags = tagger_data
	seqs = []
	for i, word_tags in enumerate(words_tags):
		# back tracking parent nods, this list stores a  word before tags with transition probability   
		bt = []
		for t, ep in word_tags:
		# check if it is the last tag jump
			if t != START:
				last_w_tags = words_tags[i-1]
				for last_w_tag,_ep in last_w_tags:
					if (last_w_tag, t) in transition:
						bt.append((last_w_tag, transition[(last_w_tag, t)]))
					else:
						tmp_max = transition[(u"<MIN>",u"<MIN>")]
						for tag in all_tags:
							if (u"<UNKNOWN>" ,tag) in emission and (last_w_tag, tag) in transition:
								if tmp_max < transition[(last_w_tag, tag)]:
									tmp_max = transition[(last_w_tag, tag)]
						bt.append((last_w_tag, tmp_max))
		seqs.append((t, ep, bt))
	return seqs[::-1]

def fbs(seqs, min_etp, recursive = False):
	if recursive : 
		return fbsr(seqs, min_etp, [])
	best_tags = []
	for t, ep, bt in seqs:
		max_tmp = (u"<UNKNOWN>", min_etp)
		for last_tag, tp in bt:
			if max_tmp[1] < ( ep + tp ):
				max_tmp = t, ( ep + tp )
		best_tags.append(max_tmp)
	return best_tags[::-1][1:-1]

# Recursive implementation
def fbsr(seqs, min_etp, acc):
	# Base case 
	if not seqs:
		return acc[::-1][1:-1]

	t, ep, bt = seqs[0]
	max_tmp = (u"<UNKNOWN>", min_etp)
	for last_tag, tp in bt:
		if max_tmp[1] < ( ep + tp ):
			max_tmp = t, ( ep + tp )
	acc.append(max_tmp)
	return fbsr(seqs[1:], min_etp, acc)


def add_one(dic, key):
	""" Helper function to add one to counter in dictionary
		It would initialize key with number one if the key does not exist
	"""
	if key in dic:
		counter_tmp = dic[key]
		dic[key] = counter_tmp + 1
	else: 
		dic[key] = 1

'''
Global
'''
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
corpus_file = "english.tagged"
# corpus_file = "persian.tagged"

if __name__ == "__main__":
	run()