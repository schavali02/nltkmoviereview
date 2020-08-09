'''
Created on Aug 1, 2020

@author: shashankchavali
'''
from nltk.corpus import twitter_samples
import nltk
from nltk.tokenize import word_tokenize
import random
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
import pickle

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
        

pos_tweets = twitter_samples.strings("positive_tweets.json")
neg_tweets = twitter_samples.strings("negative_tweets.json")

    
pos_data_words = twitter_samples.tokenized("positive_tweets.json")
neg_data_words = twitter_samples.tokenized("negative_tweets.json")

documents = []

for w in pos_tweets:
    documents.append((w, "pos"))
for w in neg_tweets:
    documents.append((w, "neg"))
    

all_words = []

for list in pos_data_words:
    for w in list:
        all_words.append(w.lower())

for list in neg_data_words:
    for w in list:
        all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

top_words = all_words.most_common(4800)
word_features = []
for pair in top_words:
    for item in pair:
        if isinstance(item, str) is True:
            word_features.append(item)


def find_features(document):
    words = set(document) 
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
      

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets) 

training_set = featuresets[:10000]
testing_set = featuresets[10000:]  


classifier = nltk.NaiveBayesClassifier.train(training_set)
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
print("Original Naive Bayes Algorithm Accuracy: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
save_classifier2 = open("MNB.pickle","wb")
pickle.dump(MNB_classifier, save_classifier2)
save_classifier2.close()
print("Multinomial Naive Bayes Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
save_classifier3 = open("BernoulliNB.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier3)
save_classifier3.close()
print("Bernoulli Naive Bayes Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
save_classifier4 = open("LogisticRegression.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier4)
save_classifier4.close()
print("Logistic Regression Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
save_classifier5 = open("SGDClassifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier5)
save_classifier5.close()
print("SGDClassifier Naive Bayes Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
save_classifier6 = open("SVC.pickle","wb")
pickle.dump(SVC_classifier, save_classifier6)
save_classifier6.close()
print("SVC Naive Bayes Accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
save_classifier7 = open("LinearSVC.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier7)
save_classifier7.close()
print("Linear SVC Naive Bayes Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
save_classifier8 = open("NuSVC.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier8)
save_classifier8.close()
print("NuSVC Naive Bayes Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  SVC_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
print("Voted Classifier Accuracy Percent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)



    