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
	all_docs = read_corpus("all_sentiment_shuffled.txt")
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
    	# else: print "The sentiment is wrong: " + sentiment + " Doc is:" + str(doc)

    return correct_nr / len(evaluation_documents)


'''
Global
'''
if __name__ == '__main__': main()


'''
	Error analysis:
	Item 1:
		DOCUMENT:
			'health neg 366.txt 
			i have found another product that i want to recommend . 
			it 's called alert and is sold by consultants like avon , but you can also go the starlight
			website and order it . it 's a nutritional supplement and for me does provide energy without 
			the jitters , and as a bonus i do n't feel hungry when i 'm using it . they also have a product 
			specifically for weight loss . i have tried that as well . i did feel a bit jittery at first.. . 
			but it settled after a week or so of taking it . 
			i would recommend it also . i used it a few years ago and lost 35 lbs . i did consistent diet and
			exercise which obviously is what did the weight loss , but i credit the supplement with giving 
			me the energy to get moving and helping me curb cravings . i 'm now looking for energy and help 
			maintaining that weightless so for me the alert does both of what i need ... 
			energy and hunger control and it 's less expensive than the weight loss supplement so i now use it . 
			i 've used this for years and have not had any problems .'
		REASON:
			It is hard to judge on this comment only by wight of the words since this comment is very long.
	Item 2:
		DOCUMENT:
			'health pos 715.txt 
			you wo n't have to use much lotion on your body after using this product . i love it... .
		REASON:
			It is impossible to make the machine understand of consequence of the first sentence 
			with the implemented algorithm in this assignment.
	Item 3:
		DOCUMENT:
			'dvd neg 885.txt well , at least the movie is faithful to its source in its spirit . based on a book 
			written on 1902 , it 's nostalgia of the british empire belongs to that era . the plot deals with a 
			soldier who is expelled from his regiment for refusing to go to fight a colonial war in sudan because
			of his impending marriage . not only his colleagues regard him now as a coward , but also his future wife . 
			he had no choice then but to go to war to prove he is a real man and not a coward . the filmmakers try 
			nothing in terms to bring the material up to date , to our more contemporary ( and one hopes , 
			more enlightened ) attitudes . there 's no post colonial guilt here whatsoever . in some ways , this 
			speaks well of the filmmakers in terms of not trying to tamper with the original material ; in another way , 
			it is a bit shocking seeing such jingoism in a contemporary movie . all of this would n't matter much if 
			the movie was entertaining ; unfortunately , it is only intermittently so'
		REASON:
			This comment have many positive wighted words which are mostly irrelevant.
	Item 4:
		DOCUMENT:
		'music pos 165.txt 
		this was not a waste of brad 's time or mine . good job !'
		REASON:
		This comment contains more negative words compare to positive ones. if our implementation 
		consider negate sentences, this comment would be classified correctly.
'''
