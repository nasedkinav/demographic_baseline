#!/usr/bin/python
# -*- coding: utf-8

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords


class SimpleClassifier:
    def __init__(self):
        # regexp initialization
        self.urls = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.address = re.compile('^\w+\s*,\s*')
        self.html = re.compile(r"</?\w+[^>]*>")
        self.vk_tags = re.compile(r'(\s|^)#\w+')

        self.stop_words = list(set(stopwords.words('russian')).union(set(stopwords.words('english'))))

        # model estimators
        self.vectorizer = CountVectorizer(max_df=0.75, ngram_range=(1, 3), stop_words=self.stop_words)
        self.classifier = MultinomialNB()

    def fit(self, samples, labels):
        texts = [self.clean_text(i) for i in samples]
        n_gram_features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(n_gram_features, labels)

    def predict(self, samples):
        texts = [self.clean_text(i) for i in samples]
        n_gram_features = self.vectorizer.transform(texts)
        return self.classifier.predict(n_gram_features)

    def clean_text(self, text):
        """
        Delete any redundant information from comment, including URLs, HTML tags, mentioning, three dots

        :param text:    comment message
        :return:        cleaned text
        """
        text = self.urls.sub('', text)
        text = self.address.sub('', text)
        text = self.html.sub('', text)
        text = text.replace('_', ' ')
        text = text.replace('...', '.')

        text = self.vk_tags.sub('', text)
        return text

    def __str__(self):
        return str(self.__dict__)
