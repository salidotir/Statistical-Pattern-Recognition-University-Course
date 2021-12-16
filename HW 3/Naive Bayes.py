#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classification

# ## Read data
# 
# * Read data
# * Remove stop words & commonly used words -> using nltk library
# * Remove punctuations * some extra charachters e.g. ?><!^&......
# * Do stemming -> means that converting each word to its root so that forexmple: **program, programming, programmer -> program**
# 
# ##  Naive Bayes classifier
# * Create a dictionary:
#     - likelihood: {**word0**: (P(word0|y=0), P(word0|y=1)), **word1**: (P(word1|y=0), P(word1|y=1)), ...}
# * Prior: probability of each class p(y)
# * To find the final class of a new-x, we should find probability of belonging to each class and then do argmax.
#     - To do so, we should compute the logarithm of multiplying probabilities **p(xi|y)** and **prior p(y)**
#     - We use logarithm to avoid the prob becomes zero since we multiply some very little amounts of probabilities. By using logarithm, it changes to sum of probabilities instead of their multiplification.
# * **Laplace Smoothing**:
#     - Laplace smoothing is also used to avoid the final probability becomes zero.
#     - It is useful in the cases that the word which we are going to find its probability given y(each class) either is the first time showing up in the dataset and have not been seen before in training phase or it has not been used in one of the classes at all. So, in each of the cases mentioned, the probability of the said word will be 0 which leads to probability equal to zero for class-y.
#     - It has a parameter $\alpha$ which can be changed.

# In[34]:


# imports

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import collections
import numpy as np
import math
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# In[14]:


class Dataset():
    def __init__(self, path, test_size=0.2):
        self.test_size = test_size
        with open(path,"r") as text_file:
            lines = text_file.read().split('\n')
        lines = [line.split("\t") for line in lines if line!=""]

        # do processing on a sentence
        self.sentences = [self.process_sentence(line[0]) for line in lines]
        self.labels = np.array([int(line[1]) for line in lines])
        
        # split words & labels
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.sentences,self.labels, test_size=self.test_size)
        
    def process_sentence(self, sentence):
        """
        Do all the necessary things here:
        - remove punctuation
        - remove stop-words
        - stem words of a sentence
        """
        new_sentence = self.stem_words(self.remove_stop_words(self.remove_puncuation(sentence)))
        return new_sentence
        
    def remove_puncuation(self, sentence):
        """
        Remove some charachters like: ? ! . , " ' ^ * ( )
        input sentence is a string
        output is a string
        """
        regex = r'[?|!|,|.|\|/|\'|\"|#|*|^|(|)]'
        new_sentence = re.sub(regex, r'', sentence)
        return new_sentence

    def remove_stop_words(self, sentence):
        """
        Remove stop words with help of nltk stopwords list:
        e.g. 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves','you', ...
        This function does not remove any punctuation.
        input sentence is an string.
        output is an array of words
        """
        stop_words = stopwords.words('english')
        new_sentence = []
        for word in sentence.split():
            # skip words that contain numbers or *|()&^%$#@!~`/\|><,.;'"
            word = word.lower()
            if word not in stop_words and len(word)>2:       # checking len > 2 to avoid words like 'go' or 'us' in the words-list
#             if word not in stop_words:
                new_sentence.append(word)
        return new_sentence

    def stem_words(self, sentence):
        """
        Find root of each word and change the word to the same format.
        e.g. program, programs, programmer, programming -> all have the root 'program'
        input sentence is an array of words
        output is an array of words
        """
        snow_stemmer = SnowballStemmer(language='english')
        new_sentence = []
        for word in sentence:
            x = snow_stemmer.stem(word)
            new_sentence.append(x)
        return new_sentence


# In[138]:


class NaiveBayesClassifier():
    def __init__(self, dataset):
        self.dataset = dataset
        self.x_train = dataset.x_train
        self.y_train = dataset.y_train
        
        self.total_dictionary, self.class_dictionary, self.total_num_words_class_dictionary = self.create_dictionary(self.x_train, self.y_train)
        self.prior = self.prior()
    
    # train-phase
    # find frequency of each word in whole dataset
    # find frequency of each word in each class 0 | 1
    def create_dictionary(self, sentences, labels):
        total_dictionary = {}
        class_dictionary = {0: dict(), 1: dict()}
        for index in range(len(sentences)):
            words_cnt = collections.Counter(sentences[index])
            for word in words_cnt:
                cnt = words_cnt[word]
                # check if word has already added to total_dictionary
                if word in total_dictionary.keys():
                    total_dictionary[word] += cnt
                else:
                    total_dictionary[word] = cnt
                
                # check if word has already added to class_dictionary
                if word in class_dictionary[labels[index]].keys():
                    class_dictionary[labels[index]][word] += cnt
                else:
                    class_dictionary[labels[index]][word] = cnt
        
        # total number of words ever used in each class
        total_num_words_class_dictionary = {}
        for i in range(len(class_dictionary)):
            values = class_dictionary[i].values()
            total_num_words_class_dictionary[i] = sum(values)
        
        return total_dictionary, class_dictionary, total_num_words_class_dictionary
    
    # probability of each class
    def prior(self):
        prior = {0:0, 1:0}
        prior[0] = len(self.y_train[self.y_train==0]) / len(self.y_train)
        prior[1] = len(self.y_train[self.y_train==1]) / len(self.y_train)

        return prior
        
    # use laplace smoothing with alpha-parameter
    def likelihood(self, sentence, alpha=1):
        likelihoods = {}
        # calculate p(x|y) for words of sentence
        for c in range(len(self.class_dictionary)):
            for word in sentence:
                if word in self.class_dictionary[c]:
                    cnt = self.class_dictionary[c][word]
                else:
                    cnt = 0
                
                if word in likelihoods:
                    likelihoods[word][c] = cnt
                else:
                    likelihoods[word] = dict()
                    likelihoods[word][c] = cnt
        
        # now that we have the counts for each word of sentence for each class, lets compute the probabilities
        # to do so, we use laplace smoothing method, means that add alpha to all counts in order to avoid prob=0
        # count + alpha / current total number of each class + k*alpha
        # where k is the number of features, meaning the number of words in each class
        class_0_features = len(self.class_dictionary[0])
        class_1_features = len(self.class_dictionary[1])
        
        for word in likelihoods:
            likelihoods[word][0] = (likelihoods[word][0] + alpha) / (self.total_num_words_class_dictionary[0] + class_0_features*alpha)
            likelihoods[word][1] = (likelihoods[word][1] + alpha) / (self.total_num_words_class_dictionary[1] + class_1_features*alpha)
        
        return likelihoods
    
    def predict_one_sentence(self, new_sentence):
        likelihoods = self.likelihood(new_sentence, alpha=0.5)
        probabilities = []
        for c in range(len(self.class_dictionary)):
            p = 0
            for word in new_sentence:
                p += math.log(likelihoods[word][c])
            p += math.log(self.prior[c])
            probabilities.append(p)
        
        # print(probabilities)
        return np.argmax(probabilities, axis=0)
    
    def predict_all(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.predict_one_sentence(x[i]))
        return np.array(res)
    
    def accuracy_with_x(self, x, y):
        res = 0
        for i in range(len(x)):
            if self.predict_one_sentence(x[i]) == y[i]:
                res += 1
        return res*100/len(x)
    
    def accuracy(self, y_predicted, y):
        res = 0
        for i in range(len(y)):
            if y[i] == y_predicted[i]:
                res += 1
        return res * 100 / len(y)
            
    # report accuracies
    def report(self, x, y):
        print("total accuracy: ")
        print(naive_bayes_classifier.accuracy_with_x(x, y))
        
        # pre class accuracies
        predicted_y = naive_bayes_classifier.predict_all(x)
        # split predicted_y for each class
        predicted_y_0 = predicted_y[y==0]
        predicted_y_1 = predicted_y[y==1]
        # split y for each class
        y_0 = y[y==0]
        y_1 = y[y==1]


        print("accuracy on class-0: ")
        print(naive_bayes_classifier.accuracy(y_0, predicted_y_0))
        print("accuracy on class-1: ")
        print(naive_bayes_classifier.accuracy(y_1, predicted_y_1))

        # testing with accuracy score
#         print("Total : ", accuracy_score(y, predicted_y))
#         print("Class 0 : ", accuracy_score(y_0, predicted_y_0))
#         print("Class 1 : ", accuracy_score(y_1, predicted_y_1))


# In[139]:


imdb_path = "Sentiment Labelled Sentences/imdb_labelled.txt"

print("Dataset imdb")
dataset = Dataset(imdb_path)
naive_bayes_classifier = NaiveBayesClassifier(dataset)

print("\n<< data train >>")
naive_bayes_classifier.report(dataset.x_train, dataset.y_train)

print("\n<< data test >>")
naive_bayes_classifier.report(dataset.x_test, dataset.y_test)


# In[140]:


yelp_path = "Sentiment Labelled Sentences/yelp_labelled.txt"

print("Dataset yelp")
dataset = Dataset(yelp_path)
naive_bayes_classifier = NaiveBayesClassifier(dataset)

print("\n<< data train >>")
naive_bayes_classifier.report(dataset.x_train, dataset.y_train)

print("\n<< data test >>")
naive_bayes_classifier.report(dataset.x_test, dataset.y_test)


# In[141]:


amazon_path = "Sentiment Labelled Sentences/amazon_cells_labelled.txt"

print("Dataset amazon")
dataset = Dataset(amazon_path)
naive_bayes_classifier = NaiveBayesClassifier(dataset)

print("\n<< data train >>")
naive_bayes_classifier.report(dataset.x_train, dataset.y_train)

print("\n<< data test >>")
naive_bayes_classifier.report(dataset.x_test, dataset.y_test)

