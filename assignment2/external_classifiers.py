
# scikit-learn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def bow_to_avmap(doc):
    return dict((w, True) for w in doc)

def train_sk(ldocs, clname='perc'):
    docs = [bow_to_avmap(doc) for _, doc in ldocs]
    lbls = [l for l, _ in ldocs]
    vec = DictVectorizer()
    encoded_docs = vec.fit_transform(docs)

    if clname == 'perc':
        classifier = Perceptron(n_iter=20)
    elif clname == 'sgd':
        classifier = SGDClassifier(penalty='elasticnet', alpha=0.0001, l1_ratio=0.85, n_iter=1000, n_jobs=-1)
    elif clname == 'nb':
        classifier = MultinomialNB()
    elif clname == 'svm':
        classifier = LinearSVC()

    classifier.fit(encoded_docs, lbls)
    return vec, classifier

def classify_sk(doc, v_c):
    vec, classifier = v_c
    encoded_doc = vec.transform(bow_to_avmap(doc))
    return classifier.predict(encoded_doc)
