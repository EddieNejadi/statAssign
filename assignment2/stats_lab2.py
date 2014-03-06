from __future__ import division
'''
Created on Feb 20, 2014

Lab assignment 2

@author: Mahdi Abdinejadi
@version: 1.0
'''


'''
import libraries
'''
import math
import scipy.stats
import random
from matplotlib import pyplot

'''
import external modules
'''
import assignment1
import external_classifiers as ec


'''
Function definitions
'''


# Learning curve
def lern_cur():
    """Shows a learning curve of assignment1 classifier
    """
    sizes = range(1,11)
    accuracies = est_acc(len(sizes))
    # print accuracies
    pyplot.plot(sizes, accuracies[:len(sizes)], 'ro')
    pyplot.show()
    
# Estimating the accuracy
def est_acc(size = 1):
    """Returns a list of booleans that shows if classifier guess is correct or not
        
        size is number that document would divide to; it can be used for cross validation 
    """
    all_docs = assignment1.read_corpus("all_sentiment_shuffled.txt")
    all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
    split_point = int(0.8*len(all_docs))
    results = []
    train_docs = all_docs[:split_point]
    eval_docs = all_docs[split_point:]

    trained_data_pices = [train_docs[i:i+size] for i in range(0, len(train_docs), size)]

    for n in range(0,size):
        trained_data = assignment1.train_nb(train_docs[n:int((n + 1)*(len(train_docs)/size))])
        results.append(assignment1.evaluate_nb(trained_data,eval_docs))
        
    return results


def eval_by_labale(lable, docs, trained_data):
    """Calculate precision and recall and return it as dictionary structure
    """
    corr_lab = [(l, doc) for (l,doc) in docs if l == lable and lable == assignment1.classify_nb(trained_data, doc)] # number of labeled docs which is correct
    gss_lab = [(l, doc) for (l,doc) in docs if lable == assignment1.classify_nb(trained_data, doc)] # number of labeled docs which is correct
    all_lab = [(l, doc) for (l,doc) in docs if l == lable] # number of labeled docs which is correct
    return {"precision": len(corr_lab)/len(gss_lab), "recall": len(corr_lab)/len(all_lab)}



def classify(classifier):
    """Returns a list of booleans that shows if classifier guess is correct or not

        classifier is either assignment1 or scikit classifier 
    """

    all_docs = assignment1.read_corpus("all_sentiment_shuffled.txt")
    all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
    split_point = int(0.8*len(all_docs))
    results = []
    train_docs = all_docs[:split_point]
    eval_docs = all_docs[split_point:]
    if classifier == "assignment1":
        trained_data = assignment1.train_nb(train_docs)
        for (s,d) in eval_docs:
            results.append( s == assignment1.classify_nb(trained_data,d))    
    elif classifier == "scikit":
        trained_data = ec.train_sk(train_docs)
        for (s,d) in eval_docs:
            results.append(s == ec.classify_sk(d, trained_data))
    else :
        print "Please set classifier as assignment1 or scikit"
    return results

# Computing a confidence interval for the accuracy
def cal_conf_intr(classifier = "assignment1"):
    return acc_ci(classify(classifier), 0.95)


# Implement the cross-validation method. Then estimate the accuracy and compute a new confidence interval.
def cross_val(N = 5):
    """Returns Returns a list of booleans that shows if classifier guess is correct or not 
        for whole test iterations
        And it prints confidence interval of whole test iterations

        N is number for iteration in document to divided to training and test parts 
    """
    all_docs = assignment1.read_corpus("all_sentiment_shuffled.txt")
    all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
    results = []
    for fold_nbr in range(N):
        split_point_1 = int(float(fold_nbr)/N*len(all_docs))
        split_point_2 = int(float(fold_nbr+1)/N*len(all_docs))
        train_docs = all_docs[:split_point_1] + all_docs[split_point_2:]
        eval_docs = all_docs[split_point_1:split_point_2]
        trained_data = assignment1.train_nb(train_docs)
        for (s,d) in eval_docs:
            results.append( s == assignment1.classify_nb(trained_data,d))
    print acc_ci(results, 0.95)
    return results

# Comparing two classifiers
def comp_classifiers():
    """Compare assignment1 and external_classifiers 'scikit' by mcnemar_difftest

        Returns boolean as significantly difference
    """
    result1 = classify("assignment1")
    result2 = classify("scikit")
    return mcnemar_difftest(result1, result2, 0.95)

# Implement the bootstrapping algorithm
def bootstrap_resampling():
    """
    """
    num_test_sets = 1000
    results = cross_val()
    accuracies = []
    rts_results = []
    for _n in range(num_test_sets):
        rad_ts = random_testset(results)
        accuracies.append(len([i for i in rad_ts if i]) / len(results))
        rts_results += rad_ts
    pyplot.hist(accuracies, bins=50)
    pyplot.show()
    sorted(accuracies)
    lower = accuracies[int(num_test_sets * 0.025)]
    upper = accuracies[int(num_test_sets * 0.975)]
    print "Cross-validation confidence interval: " + str(acc_ci(results, 0.95))
    print "Bootstrap_resampling confidence interval: " + str((lower,upper))

    # return (lower, upper)


def acc_ci(evals, significance):
    """Returns the lower and upper bounds of a confidence interval for the '
       accuracy.

       evals is a list of booleans, representing successful and 
       unsuccessful tests.
       
       significance is e.g. 0.95.
       """

    # number of tests in the test set
    ntests = len(evals)

    # number of successful tests
    nsuccesses = evals.count(True)

    # MLE of the accuracy
    acc_mle = float(nsuccesses) / ntests
    
    # standard deviation of the estimation
    sd_est = math.sqrt(acc_mle*(1.0 - acc_mle)/ntests)

    # quantile of the normal distribution
    z = scipy.stats.norm.ppf(1.0 - (1.0 - significance)/2)

    # bounds of the confidence interval
    upper = acc_mle + sd_est*z
    lower = acc_mle - sd_est*z

    return (lower, upper)

def mcnemar_difftest(evals1, evals2, significance):
    """Returns True if there is a significant difference.
    
       evals1 and evals2 are lists of booleans, representing successful and 
       unsuccessful tests.
       
       significance is e.g. 0.95
    """

    # McNemar uses the following two values
    count_corr1_fail2 = 0
    count_fail1_corr2 = 0    
    for (e1, e2) in zip(evals1, evals2):
        if not e1 and e2:
            count_fail1_corr2 = count_fail1_corr2 + 1
        elif e1 and not e2:
            count_corr1_fail2 = count_corr1_fail2 + 1

    # the more different the two classifiers are, the greater
    # will be the value of diff12
    diff12 = count_corr1_fail2 - count_fail1_corr2
    sum12 = count_corr1_fail2 + count_fail1_corr2

    if sum12 == 0 or diff12 == 0:
        return False

    # compute the McNemar test quantity
    testquantity = float(diff12*diff12) / sum12

    # compare the test quantity to a quantile of the chi-squared distribution
    threshold = scipy.stats.chi2.ppf(significance, 1)

    print "McNemar test: {0:.3f} compared to {1:.3f}".format(testquantity, threshold)
    print "Difference confidence value: {0:.3f}".format(scipy.stats.chi2.cdf(testquantity, 1))
    
    return testquantity > threshold

def random_testset(ts):
    """Returns a random set sampled from the original test set ts."""
    # the size should be the same as the size of the original test set
    size = len(ts)

    # first draw the positions randomly...
    positions = (random.randint(0, size-1) for _ in xrange(size))

    # then return a list of values taken from the test set
    return [ ts[pos] for pos in positions ]



'''
Global
'''
if __name__ == '__main__':
    """Simply runs all required function on the assignment 2
    """
    lern_cur()
    print cal_conf_intr()
    cross_val()
    print comp_classifiers()
    bootstrap_resampling()
    

