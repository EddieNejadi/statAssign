from __future__ import division

'''
Created on Feb 18, 2014

Lab assignment 1

@author: Mahdi Abdinejadi
@version: 1.0
'''


'''
import libraries
'''
from collections import Counter
from math import log10


'''
Function definitions
'''
def main():
	all_docs = read_corpus("../assignment1/all_sentiment_shuffled.txt")
	# all_docs = read_corpus("toy_dataset.txt")
	all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
	split_point = int(0.8*len(all_docs))
	train_docs = all_docs[:split_point]
	eval_docs = all_docs[split_point:]
	trained_data = train_nb(train_docs)
	print "accuracy is: " + str( int (evaluate_nb(trained_data,eval_docs) * 10000) / 100 ) + "%"




def read_corpus(corpus_file):
    out = []
    with open(corpus_file) as f:
        for line in f:
            tokens = line.strip().split()
            out.append( (tokens[0], tokens[1], tokens[3:]) )
    return out

	
# Write a Python function that uses a training set of documents to estimate the probabilities in the Naive Bayes model.
# Return some data structure containing the probabilities.
# The input parameter of this function should be a list of documents with sentiment labels, i.e. a list of pairs like train_docs above
def train_nb(training_docs):

	voc = [v for (_sentiment,lst) in training_docs for v in lst]
	pos_lst = [v for (sentiment,lst) in training_docs if sentiment == "pos"  for v in lst]
	neg_lst = [v for (sentiment,lst) in training_docs if sentiment == "neg"  for v in lst]
	pos = {}
	neg = {}
	
	pos_lst_len = len(pos_lst)
	neg_lst_len = len(neg_lst)
	voc_len = len(voc)
	
	# print(voc_len, pos_lst_len, neg_lst_len)
	
	pos_fd = Counter(pos_lst)
	neg_fd = Counter(neg_lst)
	
	# implementation of Laplace smoothing
	for (v,fn) in pos_fd.iteritems():
		pos[v] = log10((fn + 1)) - log10((pos_lst_len + voc_len))

	for (v,fn) in neg_fd.iteritems():
		neg[v] = log10((fn + 1)) - log10((neg_lst_len + voc_len))

	# print 'pos and neg dictionaries are done'

	return {"pos": pos, "neg": neg, "pos_lst_len": pos_lst_len, "neg_lst_len": neg_lst_len, "voc_len":voc_len}

# Then write a Python function that classifies a new document.
# The inputs are 1) the probabilities returned by the first function; 2) the document to classify, which is a list of tokens.
def classify_nb(classifier_data, document):
    prob_pos = prob_neg = 0.0


    for w in document:
    	if w in classifier_data.get("pos") :
    		prob_pos += classifier_data.get("pos").get(w)
    	else:
    		prob_pos += -1.0 * log10(classifier_data.get("voc_len") + classifier_data.get("pos_lst_len"))
    	
    	if w in classifier_data.get("neg") :
    		prob_neg += classifier_data.get("neg").get(w)
    	else:
    		prob_neg += -1.0 * log10(classifier_data.get("voc_len") + classifier_data.get("neg_lst_len"))

    voc_len = len(classifier_data.get("pos")) + len(classifier_data.get("neg"))
    if prob_pos > prob_neg:
    	return "pos"
    else:
    	return "neg"


# we just compute the accuracy, i.e. the number of correctly classified documents divided by the total number of documents.
# Write a function that classifies each document in the test set, compares each label to the gold-standard label, and returns the accuracy.
def evaluate_nb(classifier_data, evaluation_documents):
    correct_nr = 0
    for (sentiment, doc) in evaluation_documents:
    	if sentiment == classify_nb(classifier_data, doc): correct_nr += 1
    	# else: print "The sentiment is wrong: " + sentiment

    return correct_nr / len(evaluation_documents)


'''
Global
'''
if __name__ == '__main__': main()