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
    sizes = range(1,11)
    accuracies = est_acc(len(sizes))
    # print sizes
    # print accuracies
    pyplot.plot(sizes, accuracies[:len(sizes)], 'ro')
    pyplot.show()
    
# Estimating the accuracy
def est_acc(size = 1):
    all_docs = assignment1.read_corpus("../assignment1/all_sentiment_shuffled.txt")
    all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
    split_point = int(0.8*len(all_docs))
    results = []
    train_docs = all_docs[:split_point]
    eval_docs = all_docs[split_point:]

    # print "len train_docs:" + str(len(train_docs)) + "  size:" + str(size)
    trained_data_pices = [train_docs[i:i+size] for i in range(0, len(train_docs), size)]

    for n in range(0,size):
        trained_data = assignment1.train_nb(train_docs[n:int((n + 1)*(len(train_docs)/size))])
        results.append(assignment1.evaluate_nb(trained_data,eval_docs))
        
    # print "accuracy is: " + str( int (assignment1.evaluate_nb(trained_data,eval_docs) * 10000) / 100 ) + "%"
    # print "precision for positive class is: " + str(eval_by_labale("pos",eval_docs, trained_data).get("precision"))
    # print "recall is for positive class: " + str(eval_by_labale("pos",eval_docs, trained_data).get("recall"))
    return results


def eval_by_labale(lable, docs, trained_data):
    
    corr_lab = [(l, doc) for (l,doc) in docs if l == lable and lable == assignment1.classify_nb(trained_data, doc)] # number of labled docs which is correct
    gss_lab = [(l, doc) for (l,doc) in docs if lable == assignment1.classify_nb(trained_data, doc)] # number of labled docs which is correct
    all_lab = [(l, doc) for (l,doc) in docs if l == lable] # number of labled docs which is correct
    return {"precision": len(corr_lab)/len(gss_lab), "recall": len(corr_lab)/len(all_lab)}


# Computing a confidence interval for the accuracy
def cal_conf_intr(classifier = "assignment1"):
    all_docs = assignment1.read_corpus("../assignment1/all_sentiment_shuffled.txt")
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
    # for (s,d) in eval_docs:
    #     if classifier == "assignment1": results.append( s == assignment1.classify_nb(trained_data,d))
    #     else : results.append(s == ec.classify_sk(d, trained_data))
    return results

# Implement the cross-validation method. Then estimate the accuracy and compute a new confidence interval.
def cross_val(N = 4):
    all_docs = assignment1.read_corpus("../assignment1/all_sentiment_shuffled.txt")
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
    return results

# Comparing two classifiers
def comp_classifiers():
    # 
    result1 = cal_conf_intr("assignment1")
    result2 = cal_conf_intr("scikit")
    return mcnemar_difftest(result1, result2, 0.95)




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
    # lern_cur()
    # l,u = acc_ci(cal_conf_intr(), 0.95)
    # print (l, u, u-l)
    # l,u = acc_ci(cross_val(), 0.95)
    # print (l, u, u-l)
    # all_docs = assignment1.read_corpus("../assignment1/all_sentiment_shuffled.txt")
    # all_docs = [(sentiment, doc) for (_, sentiment, doc) in all_docs]
    # split_point = int(0.8*len(all_docs))
    # results = []
    # train_docs = all_docs[:split_point]
    # eval_docs = all_docs[split_point:]
    # trained_data = ec.train_sk(train_docs)
    # for (s,d) in eval_docs:
    #     results.append(s == ec.classify_sk(d, trained_data))
    # l,u = acc_ci(cal_conf_intr(), 0.95)
    # print len(results)
    # print len([r for r in results if r])
    # print (l, u, u-l)
    print comp_classifiers()

