import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from keras.layers import LSTM, Dense, Input, Dropout
from keras import Model
from keras.layers.core import Reshape
from keras.utils import to_categorical
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer


def get_data(prop):
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
    # split = 0.3
    # n = int(len(data) * split)
    # train_data = train_data[:n]
    # train_target = train_target[:n]

    return train_data, train_target, test_data, test_target, data


def cm_print(test_target, y_pred):
    ax = plt.subplot()
    sns.heatmap(confusion_matrix(test_target, y_pred), annot=True, cmap='Spectral', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    ax.yaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    plt.show()


train_data, train_target, test_data, test_target0, texts = get_data(0)

train_target = to_categorical(train_target)
test_target = to_categorical(test_target0)

NGRAM_MIN = 2
NGRAM_MAX = 3
analyzer = 'char_wb'
fu = FeatureUnion([
        # ('ave', AverageWordLengthExtractor()),
        # ('cons', CountConsonantsExtractor()),
        ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
    ])
fu.fit(texts)
train_data = fu.transform(train_data).toarray()
test_data = fu.transform(test_data).toarray()

# ff model
inputs = Input(shape=(train_data.shape[1],))
x = Dense(100, activation="relu")(inputs)
x = Dense(40, activation="relu")(x)
x = Dense(10, activation="relu")(x)
x = Dense(units=4, activation="softmax")(x)

model = Model(inputs, x)
print(model.summary())

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
model.fit(train_data, train_target, batch_size=128, epochs=8, verbose=1, validation_split=0.2)

print('FF')
scores = model.evaluate(test_data, test_target, verbose=2)
print(scores)

y_pred = model.predict(test_data)
y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
print(classification_report(test_target, y_norm))
print('MCC: ', matthews_corrcoef(test_target0, np.argmax(y_norm, axis=1)))
print("classes: 0-cy, 1-en, 2-ga, 3-gd")

cm_print(test_target0, np.argmax(y_norm, axis=1))
