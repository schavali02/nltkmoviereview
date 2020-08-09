#File: textanalysismodule.py


import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        
short_pos = open("/Users/shashankchavali/Documents/positive.txt","r", encoding='ISO-8859-1').read()
short_neg = open("/Users/shashankchavali/Documents/negative.txt","r", encoding='ISO-8859-1').read()

documents = []
all_words = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


document_pickle = open("documents.pickle", "rb")
documents = pickle.load(document_pickle)
document_pickle.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


wf_pickle = open("wf.pickle", "rb")
word_features = pickle.load(wf_pickle)
wf_pickle.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set =  featuresets[10000:]




classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier1 = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier1)
classifier1.close()



MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
classifier2 = open("MNB.pickle", "rb")
MNB_classifier = pickle.load(classifier2)
classifier2.close()


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
classifier3 = open("Bernoulli.pickle", "rb")
BernoulliNB_classifier = pickle.load(classifier3)
classifier3.close()


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
classifier4 = open("LogReg.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier4)
classifier4.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
classifier6 = open("LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier6)
classifier6.close()


voted_classifier = VoteClassifier(
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier(feats),voted_classifier.confidence(feats)