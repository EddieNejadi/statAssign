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
	print sorted(errors)



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

	for sent in tagged_sentences:
		add_one(dic["t"], START)
		add_one(dic["wt"], (START, START))
		last_tag = START

		for w,t in sent:
			add_one(dic["wt"], (w,t))
			add_one(dic["t"], t)
			add_one(dic["tt"], (last_tag,t))
			last_tag = t
		
		add_one(dic["t"], END)
		add_one(dic["wt"], (END, END))
		add_one(dic["tt"], (last_tag, END))

	emission = {}
	transition = {}
	tmp_min = 0.0
	for w,t in dic["wt"]:
		emission[(w,t)] = log10( dic["wt"][(w,t)] / sum([ dic["wt"][wx,tx] for wx,tx in dic["wt"] if wx == w]))
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
	best_seq = find_best_sequence(tagger_data, lst)
	return zip([w for w,_t in sentence], [t for t,_l in best_seq])

def viterbi(tagger_data, sentence):
	# make a dummy item with a START tag, no predecessor, and log probability 0
	current_list = [[(START, 0.0)]]

	emission, _transition, all_tags = tagger_data

	for w,_t in sentence:
		w_tags = [t for t in all_tags if (w,t) in emission]
		# For empty list, considering <UNKNOWN> word as minimum value in table
		if not w_tags:
			current_list.append([(w, emission[(u"<MIN>",u"<MIN>")])])
		else:
			tmp = []	
			for w_tag in w_tags:
				tmp.append((w_tag, emission[(w,w_tag)]))
			current_list.append(tmp)
	current_list.append([(END, 0.0)])

	return current_list

# sequence 
def find_best_sequence(tagger_data, seqs):
	""" tagger_data is tables of possibilities
		seqs is all possible tags with emission probabilities for each words of sentence

		return a list of best (tag, possible log) for each words of sentence
	"""
	_emission, transition, all_tags = tagger_data
	best_seq = [seqs[0][0]]
	for i,seq in enumerate(seqs[1:]):
		tmp = []
		for t,l in seq:
			if (best_seq[-1][0], t) in transition:
				tmp.append((t, l + transition[(best_seq[-1][0], t)]))
			else:
				tmp.append((t, l + transition[(u"<MIN>",u"<MIN>")]))

		best_seq.append(max(tmp, key = lambda t: t[1]))
	return best_seq[1:-1]
	
def find_best_item(word, tag, possible_predecessors):    
	# determine the emission probability: 
	#  the probability that this tag will emit this word
	
	# find the predecessor that gives the highest total log probability,
	#  where the total log probability is the sum of
	#    1) the log probability of the emission,
	#    2) the log probability of the transition from the tag of the 
	#       predecessor to the current tag,
	#    3) the total log probability of the predecessor
	
	# return a new item (tag, best predecessor, best total log probability)
	pass

def retrace(end_item, sentence_length):
	# tags = []
	# item = predecessor of end_item
	# while the tag of the item isn't START:
	#     add the tag of item to tags
	#     item = predecessor of item
	# reverse the list of tags and return it
	pass



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

if __name__ == "__main__":
	run()