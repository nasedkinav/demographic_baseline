#!/usr/bin/python
# -*- coding: utf-8

import numpy as np
import json
import time
import codecs

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from simple_clf import SimpleClassifier

# your data filed, which stores user messages
COMMENTS_STORAGE_FIELD = 'user_comments'

# your data field, which stands for classes
LABEL_FIELD = 'bdate'

DATA_FILENAME = 'test_data.json'


def manual_cv(classifier, texts, labels, n_folds):
    folds = cross_validation.KFold(len(texts), n_folds=n_folds, shuffle=True)
    labels, texts = np.array(labels), np.array(texts)
    scores, times = [], []
    count = 0

    for train_indices, test_indices in folds:
        count += 1
        print('k_fold #{0}'.format(count))

        train_texts, train_labels = texts[train_indices], labels[train_indices]
        test_texts, test_labels = texts[test_indices], labels[test_indices]

        cv_time = time.time()

        print('training...')
        classifier.fit(train_texts, train_labels)

        print('classifying...')
        test_result = classifier.predict(test_texts)
        score = accuracy_score(test_labels, test_result)
        scores.append(score)
        print classification_report(test_labels, test_result)

        cur_time = time.time() - cv_time
        times.append(cur_time)
        print("testing elapsed time of {0} fold: {1:.3f} sec\n".format(count, cur_time))

    mean_score = sum(scores) / len(scores)
    print('mean score = {0}'.format(mean_score))
    mean_time = sum(times) / len(times)
    print('mean time = {0}'.format(mean_time))


if __name__ == "__main__":
    print('start')
    # assume UTF-8 without BOM encoding
    with codecs.open(DATA_FILENAME, mode='r', encoding='utf-8') as data_file:
        data = json.load(data_file)

    # gather comments with corresponding labels
    X = []
    y = []
    for person in data:
        for comment in person[COMMENTS_STORAGE_FIELD]:
            X.append(comment)
            y.append(person[LABEL_FIELD])

    manual_cv(classifier=SimpleClassifier(), texts=X, labels=y, n_folds=5)
    print('finish')
