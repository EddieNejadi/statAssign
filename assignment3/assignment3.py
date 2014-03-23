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
	print len(all_docs)
	training, test = split_data(all_docs)
	print len(training)
	print len(test)
	print len(test) + len(training)
	tagger = train_nltk_baseline(training)
	print tagger
	print 'Accuracy: %4.2f%%' % (100.0 * tagger.evaluate(test))
	# print training[:1]
	emission, transition = hmm_train_tagger(training)
	tags = []
	for sent in training:
		tags.append(hmm_tag_sentence((emission, transition), sent))


def split_data(all_docs):
	""" Split the data to traing part 80% and 
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
	# estimate the emission and transition probabilities
	# return the probability tables
	# emission = {"wt": {}, "t": {}}
	# transition = {}
	dic = {"wt": {}, "t": {}, "tt" :{}}
	# for sent in tagged_sentences[:2]:
	# initial START and END tag value in emission as default 
	# emission["t"][START] = 0
	# emission["wt"][(START, START)] = 0
	# emission["t"][END] = 0
	# emission["wt"][(END, END)] = 0

	for sent in tagged_sentences:
		add_one(dic["t"], START)
		add_one(dic["wt"], (START, START))
		# counter_tmp = emission["t"][START] 
		# emission["t"][START] = counter_tmp + 1
		
		last_tag = START

		for w,t in sent:
			add_one(dic["wt"], (w,t))
			add_one(dic["t"], t)

			# if (w,t) in emission["wt"]:
			# 	# print (w,t)
			# 	counter_tmp = emission["wt"][(w,t)]
			# 	emission["wt"][(w,t)] = counter_tmp + 1
			# else:
			# 	emission["wt"][(w,t)] = 1
			# if t in emission["t"]:
			# 	counter_tmp = emission["t"][t]
			# 	emission["t"][t] = counter_tmp + 1
			# else:
			# 	emission["t"][t] = 1

			# if (last_tag, t) in transition:
			# 	counter_tmp = transition[(last_tag,t)]
			# 	transition[(last_tag,t)] = counter_tmp + 1
			# else:
			# 	transition[(last_tag,t)] = 1

			add_one(dic["tt"], (last_tag,t))
			last_tag = t

		
		add_one(dic["t"], END)
		add_one(dic["wt"], (END, END))
		add_one(dic["tt"], (last_tag, END))
		# print "========================"
	# print emission["wt"][(u"the",u"DT")]
	emission = {}
	transition = {}
	# print w
	# print dic["wt"][("see","VB")]
	# print sum([ dic["wt"][wx,tx] for wx,tx in dic["wt"] if wx == "see"])
	for w,t in dic["wt"]:
		emission[(w,t)] = log10( dic["wt"][(w,t)] / sum([ dic["wt"][wx,tx] for wx,tx in dic["wt"] if wx == w]))
	
	any_tag_sum = sum([dic["t"][t] for t in dic["t"]])
	for t1,t2 in dic["tt"]:
		lamda1 = dic["t"][t2] / any_tag_sum
		lamda2 = dic["tt"][(t1,t2)] / dic["t"][t1]
		transition[(t1,t2)] = log10(lamda2 + lamda1)
		# transition[(t1,t2)] = 10
	return (emission, transition)


def hmm_tag_sentence(tagger_data, sentence):
	# apply the Viterbi algorithm
	lst = viterbi(tagger_data, sentence)
	# then retrace your steps
	return retrace(END, lst)
	# finally return the list of tagged words
	pass



def viterbi(tagger_data, sentence):
	# make a dummy item with a START tag, no predecessor, and log probability 0
	# current list = [ the dummy item ]
	
	# for each word in the sentence:
	#    previous list = current list
	#    current list = []        
	#    determine the possible tags for this word
	#  
	#    for each tag of the possible tags:
	#         add the highest-scoring item with this tag to the current list

	# end the sequence with a dummy: the highest-scoring item with the tag END
	pass
	
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


# all_sentences = read_tagged_corpus(YOUR_CORPUS)

# divide the sentences into a training and a test part

# train the bigram tagger
# train the baseline tagger

# evaluate the bigram tagger and the baseline


def add_one(dic, key):
	""" Helper function to add one to counter in dictionary
		It will add number one if key does not exist
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