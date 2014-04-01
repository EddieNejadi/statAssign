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
import copy


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
	print "Error list length is:" + str(len(errors))

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
	best_tags = viterbi(tagger_data, sentence)
	return zip([w for w,_t in sentence], best_tags)

def viterbi(tagger_data, sentence):
	""" calculates the most probable tag sequence for the sentence
	"""
	# Contains a list of list for each word; inner list contains (tag, emission probability) for each tag
	# make a dummy item with a START tag, no predecessor, and log probability 0
	words_tags_list = [[(START, 0.0)]]

	emission, transition, all_tags = tagger_data

	# For each word in sentence, generating a list of (tag, emission probability) based on training data
	for w,_t in sentence:
		tmp = []	
		w_tags = [t for t in all_tags if (w,t) in emission]
		# For empty list, considering <UNKNOWN> word 
		if not w_tags:
			# Considering all tags only if they are tagged for hapaxes 
			for tag in all_tags:
				if (u"<UNKNOWN>", tag) in emission:
					tmp.append((tag, emission[(u"<UNKNOWN>",tag)]))
		else:
			for w_tag in w_tags:
				tmp.append((w_tag, emission[(w,w_tag)]))
		words_tags_list.append(tmp)
	words_tags_list.append([(END, 0.0)])

	# New changes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# passes contains all passes as (total probability, list of tags)
	# it initialized with start point 
	passes = [(0.0,[START])]

	# Ignoring start point because it is add to pass before
	for i,w_tags in enumerate(words_tags_list[1:]):
		# For each word passes get updated
		new_passes = []
		# Only considering the max probable tag sequence for the words before
		for total_prb, tag_pass in [max(passes)]:
			# Iterating in all tags of the word
			for t, ep in w_tags:
				# Check if last tag and new tag have transition probability 
				if (tag_pass[-1], t) in transition:
					p = total_prb + ep + transition[(tag_pass[-1], t)] 
				# if not considering minimum value 
				else:
					p = total_prb + ep + transition[(u"<MIN>",u"<MIN>")] 
				# Update new pass for each tag
				new_passes.append((p,tag_pass + [t]))
		# Update passes with new passes 
		passes = copy.copy(new_passes)
		# print len(passes)
	# Returning most probable tag sequence without STAR and END dummy tags
	return max(passes)[1][1:-1]
	# END New changes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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