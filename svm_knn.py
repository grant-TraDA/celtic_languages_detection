import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift

from sklearn.metrics import classification_report


def get_data(prop):
    file = "./data/final.tsv"
    data_all = pd.read_csv(file, sep='\t', header=None, quoting=3, error_bad_lines=False)
    data_all[0] = data_all[0].astype(str)
    data_all = shuffle(data_all)

    # splt = int(len(data_all) * prop)
    # data_all = data_all[:splt]

    data = data_all.iloc[:, 0]
    target = data_all.iloc[:, 1]

    data = data.apply(lambda row: row.translate(str.maketrans('', '', string.punctuation)).
                      translate(str.maketrans('', '', string.digits)).lower())

    target = target.astype("category").cat.codes

    split = 0.8
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]
    test_data = data[n:]
    test_target = target[n:]

    print("classes: 0-cy, 1-en, 2-ga, 3-gd")

    return train_data, train_target, test_data, test_target


def SVM_classification(train_data, train_target, test_data, test_target):
    text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), analyzer='char')),
                             # ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                             ('clf-svm', SVC(kernel='linear', C=1))])

    text_clf_svm = text_clf_svm.fit(train_data, train_target)

    predicted_svm = text_clf_svm.predict(test_data)
    SVM_acc = np.mean(predicted_svm == test_target)
    print("SVM accuracy ", SVM_acc)

    # predicted_svm_tr = text_clf_svm.predict(train_data)
    # SVM_tr = np.mean(predicted_svm_tr == train_target)
    # print("SVM accuracy on train ", SVM_tr)

    # print('train set')
    # y_pred = text_clf_svm.predict(train_data)
    # print(classification_report(train_target, y_pred))

    print('test set')
    y_pred = text_clf_svm.predict(test_data)
    print(classification_report(test_target, y_pred))


def kNN_classification(train_data, train_target, test_data, test_target):
    text_clf_knn = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), analyzer='char')),
                             # ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                             ('clf-knn', KNeighborsClassifier(n_neighbors=4))])

    text_clf_knn = text_clf_knn.fit(train_data, train_target)

    predicted_knn = text_clf_knn.predict(test_data)
    knn_acc = np.mean(predicted_knn == test_target)
    print("kNN accuracy ", knn_acc)

    # predicted_knn_tr = text_clf_knn.predict(train_data)
    # knn_tr = np.mean(predicted_knn_tr == train_target)
    # print("kNN accuracy on train ", knn_tr)

    # print('train set')
    # y_pred = text_clf_knn.predict(train_data)
    # print(classification_report(train_target, y_pred))

    print('test set')
    y_pred = text_clf_knn.predict(test_data)
    print(classification_report(test_target, y_pred))


# def meanShift(train_data, train_target, test_data, test_target):
#     text_clf_ms = Pipeline([('vect', CountVectorizer(ngram_range=(3, 3), analyzer='char')),
#                              # ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
#                              ('clf-ms', MeanShift())])
#     vect = CountVectorizer(ngram_range=(2, 2), analyzer='char')
#     X = vect.fit_transform(train_data)
#     text_clf_ms = MeanShift(bandwidth=3)
#
#     text_clf_ms = text_clf_ms.fit(X.toarray())
#
#     predicted_ms = text_clf_ms.predict(train_data)
#     clusters0 = set()
#     clusters1 = set()
#     clusters2 = set()
#     clusters3 = set()
#     for (cl, lab) in zip(predicted_ms, train_target):
#         if lab == 0:
#             clusters0.add(cl)
#         if lab == 1:
#             clusters1.add(cl)
#         if lab == 2:
#             clusters2.add(cl)
#         if lab == 3:
#             clusters3.add(cl)
#
#     print("clusters 0 {}".format(clusters0))
#     print("clusters 1 {}".format(clusters1))
#     print("clusters 2 {}".format(clusters2))
#     print("clusters 3 {}".format(clusters3))
#
#     predicted_ms_test = text_clf_ms.predict(test_data)
#     predicted_labels = list()
#     for (pred, lab) in zip(predicted_ms_test, test_target):
#         if lab == 0:
#             if pred in clusters0:
#                 predicted_labels.append(0)
#             else:
#                 predicted_labels.append(-1)
#         if lab == 1:
#             if pred in clusters1:
#                 predicted_labels.append(0)
#             else:
#                 predicted_labels.append(-1)
#         if lab == 2:
#             if pred in clusters2:
#                 predicted_labels.append(0)
#             else:
#                 predicted_labels.append(-1)
#         if lab == 3:
#             if pred in clusters3:
#                 predicted_labels.append(0)
#             else:
#                 predicted_labels.append(-1)
#
#     ms_acc = np.mean(predicted_labels == test_target)
#     print("Mean Shift accuracy ", ms_acc)
#
#     # predicted_knn_tr = text_clf_knn.predict(train_data)
#     # knn_tr = np.mean(predicted_knn_tr == train_target)
#     # print("kNN accuracy on train ", knn_tr)
#     #
#     # # report
#     # from sklearn.metrics import classification_report
#     #
#     # print('train set')
#     # y_pred = text_clf_knn.predict(train_data)
#     # print(classification_report(train_target, y_pred))
#     #
#     print('test set')
#     # y_pred = text_clf_knn.predict(test_data)
#     print(classification_report(test_target, predicted_labels))


if __name__ == "__main__":
    train_data, train_target, test_data, test_target = get_data(0)
    SVM_classification(train_data, train_target, test_data, test_target)
    print("--------------------------------------------------")
    kNN_classification(train_data, train_target, test_data, test_target)
