import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import treebank
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.book import *
import datetime
import math
import operator
import importlib
import sys


class PreprocessData:
    """
    1. Tokenize text
    2. Remove infrequent words
    3. Prepend spacial start and end tokens
    """
    def __init__(self, data, vocabulary_size=8000):
        self.text_data = data
        self.UNKNOWN_WORD = 'UNKNOWN_WORD'
        self.SENTENCE_START = 'SENTENCE_START'
        self.SENTENCE_END = 'SENTENCE_END'
        self.vocabulary_size = vocabulary_size
        unified_text = self.text_data.iloc[:]
        unified_text = unified_text.str.cat(sep=' ')
        tokenized_text = word_tokenize(unified_text)
        self.freq_dist = FreqDist(tokenized_text)
        freq_dist_comm = self.freq_dist.most_common(vocabulary_size-1)
        self.freq_dist_dict = dict(freq_dist_comm)
        self.freq_dist_dict['SENTENCE_START'] = 0
        self.freq_dist_dict['SENTENCE_END'] = 1
        print('Using vocabulary size: {:1d}'.format(vocabulary_size))
        print('The least frequent word is "{:s}" and appeard {:1d} time'.format(freq_dist_comm[-1][0], freq_dist_comm[-1][1]))
        self.build_useful_vectors()
    def lowercase_data(self):
        pass
    def remove_symbols(self):
        pass
    def remove_nan(self):
        text_data = self.text_data.dropna().reset_index()
        text_data = text_data.drop(columns=['index'], axis=0)
        self.text_data = text_data.body
        return self.text_data
    def tokenize_sentence(self):
        print('Starting tokenization of data by sentence..')
        start = datetime.datetime.utcnow()
        self.text_data = self.text_data.apply(lambda x: sent_tokenize(x))
        end = datetime.datetime.utcnow()
        print('Sentence tokenization finished in {:1d} minutes, {:1d} seconds'.format((end.second//60)%60, end.second))
        return self.text_data
    def tokenize_word(self):
        def go_over_rows(row):
            tokenized_sent = []
            words = []
            for sentence in row:
                words_temp = ['SENTENCE_START'] + word_tokenize(sentence) + ['SENTENCE_END']
                tokenized_words_serie = pd.Series(words_temp)
                tokenized_words_serie = tokenized_words_serie.apply(lambda x: x if (x in self.freq_dist_dict) else self.UNKNOWN_WORD)
                words.append(tokenized_words_serie[:])
                #tokenized_sent.append(word_tokenize(sentence))
            tokenized_sent.append(words)
            return words
        text_data_serie = pd.Series(self.text_data)
        text_data_serie = text_data_serie.apply(go_over_rows)
        return text_data_serie
        # self.text_data = text_data_serie
        #return text_data_serie
    def get_vobucabulary(self):
        return self.freq_dist_dict
    def build_useful_vectors(self):
        index_to_word = [x[0] for x in self.freq_dist.most_common(self.vocabulary_size-1)]
        index_to_word = [self.SENTENCE_START] + [self.SENTENCE_END]  + index_to_word
        #index_to_word.append(self.SENTENCE_START)
        #index_to_word.append(self.SENTENCE_END)
        index_to_word.append(self.UNKNOWN_WORD)
        self.index_to_word = index_to_word
        self.word_to_index = dict([w,i] for i,w in enumerate(index_to_word))
        return self.word_to_index

    def print_data(self, num_rows):
        print(self.text_data[0:num_rows])

class TextDataPipeline:
    def __init__(self, preprocessedData, word_to_index):
        self.preprocessedData = preprocessedData
        self.word_to_index = word_to_index
        self.create_training_data()
        self.create_evaluation_data()


    def create_training_data(self, train_percentage=60):
        X_train = []
        y_train = []
        #consider using list comprehension to simply this task
        total_data = self.preprocessedData.shape[0]
        to_train = math.ceil((total_data*train_percentage)/100)
        for sentences in self.preprocessedData[:to_train-1]:
            X_train.append([[self.word_to_index[w] for w in sentence[:-1]] for sentence in sentences])
            y_train.append([[self.word_to_index[w] for w in sentence[1:]] for sentence in sentences])
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.X_train = X_train
        self.y_train = y_train
        return self.X_train, y_train

    def create_evaluation_data(self):
        pass

#usage
    #1st:Read .csv file
data = pd.read_csv("reddit_comments.csv", sep=',', nrows=50000)
data = data.dropna(subset=['body']).reset_index()
data = data.drop(columns=['index'], axis=0)
    #2nd: Use PreprocessData class
np.random.seed(10)
pdObj = PreprocessData(data.body, vocabulary_size=8000)
txt_data = pdObj.remove_nan()
txt_data = pdObj.tokenize_sentence()
txt_data = pdObj.tokenize_word()
