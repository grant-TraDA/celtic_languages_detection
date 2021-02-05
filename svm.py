from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline, FeatureUnion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

NGRAM_MIN = 2
NGRAM_MAX = 3
analyzer = 'char_wb'

consonants = list("bcdfghjklmnpqrstvwxzŵ")


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def average_word_length(self, sent):
        return np.mean([len(word) for word in sent.split()])

    def transform(self, df, y=None):
        return pd.DataFrame(df.apply(self.average_word_length))

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CountConsonantsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def consonants_count(self, sent):
        return sum(word.count(c) for c in consonants for word in sent.split()) / len(sent.split())

    def transform(self, df, y=None):
        return pd.DataFrame(df.apply(self.consonants_count))

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CountLetterExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, letter):
        self.letter = letter

    def letter_count(self, sent):
        return sent.count(self.letter) / len(sent.split())

    def transform(self, df, y=None):
        return pd.DataFrame(df.apply(self.letter_count))

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


def get_data():
    file = "./data/1preproc.tsv"
    data_all = pd.read_csv(file, sep='\t', header=None, quoting=3, error_bad_lines=False)
    data_all[0] = data_all[0].astype(str)

    data = data_all.iloc[:, 0]
    target = data_all.iloc[:, 1]

    target = target.astype("category").cat.codes

    split = 0.8
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]
    test_data = data[n:]
    test_target = target[n:]

    # uncomment for reduced dataset
    #split = 0.3
    #n = int(len(data) * split)
    #train_data = train_data[:n]
    #train_target = train_target[:n]

    print("classes: 0-cy, 1-en, 2-ga, 3-gd")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(train_data[train_data.str.split().str.len().lt(3)])

    return train_data, train_target, test_data, test_target


def print_cm(test_target, y_pred):
    ax = plt.subplot()
    sns.heatmap(confusion_matrix(test_target, y_pred), annot=True, cmap='Spectral', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    ax.yaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    plt.show()


def SVM_classification(train_data, train_target, test_data, test_target, features, letter='w'):
    if features == 'ngram':
        text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer)),
                                ('clf-svm', SVC(kernel='linear', C=1))])
    elif features == 'ave':
        text_clf_svm = Pipeline([('feats', FeatureUnion([
                                ('ave', AverageWordLengthExtractor()),
                                ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
                                ])),
                             ('clf-svm', SVC(kernel='linear', C=1))])
    elif features == 'cons':
        text_clf_svm = Pipeline([('feats', FeatureUnion([
                                ('cons', CountConsonantsExtractor()),
                                 ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
                                ])),
                                 ('clf-svm', SVC(kernel='linear', C=1))])
    elif features == 'ave-cons':
        text_clf_svm = Pipeline([('feats', FeatureUnion([
                                ('ave', AverageWordLengthExtractor()),
                                ('cons', CountConsonantsExtractor()),
                                 ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
                                ])),
                                 ('clf-svm', SVC(kernel='linear', C=1))])
    elif features == 'letter':
        text_clf_svm = Pipeline([('feats', FeatureUnion([
            ('ave', AverageWordLengthExtractor()),
            ('cons', CountConsonantsExtractor()),
            # ('w-letter', CountLetterExtractor('w')),
            # ('h-letter', CountLetterExtractor('h'))
            ('letter', CountLetterExtractor(letter)),
            ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
        ])),
                                 ('clf-svm', SVC(kernel='linear', C=1))])
    else:
        return

    text_clf_svm = text_clf_svm.fit(train_data, train_target)

    y_pred = text_clf_svm.predict(test_data)
    SVM_acc = np.mean(y_pred == test_target)
    print("SVM accuracy ", SVM_acc)

    print('test set')
    print(classification_report(test_target, y_pred))
    print('MCC: ', matthews_corrcoef(test_target, y_pred))

    print_cm(test_target, y_pred)


if __name__ == "__main__":
    train_data, train_target, test_data, test_target = get_data()

    print("*****(only ngrams)*****")
    SVM_classification(train_data, train_target, test_data, test_target, 'ngram')
    print("--------------------------------------------------")

    print("*****(ngrams + average word length)*****")
    SVM_classification(train_data, train_target, test_data, test_target, 'ave')
    print("--------------------------------------------------")

    print("*****(ngrams + average number of consonants per word)*****")
    SVM_classification(train_data, train_target, test_data, test_target, 'cons')
    print("--------------------------------------------------")

    print("*****(ngrams + average word length + average number of consonants per word)*****")
    SVM_classification(train_data, train_target, test_data, test_target, 'ave-cons')
    print("--------------------------------------------------")

    # print("*****(ngrams + average word length + average number of consonants per word + letter count)*****")
    # SVM_classification(train_data, train_target, test_data, test_target, 'letter', 'n')
    # print("--------------------------------------------------")
    # # for l in ('i', 'a', 'c', 'n'):
    # #     print('----------')
    # #     print('letter: ' + l)
