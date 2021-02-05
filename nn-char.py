import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.layers import Embedding, Flatten, Convolution1D, GlobalMaxPooling1D, Concatenate, AlphaDropout
from keras import regularizers

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

    return train_data, train_target, test_data, test_target, data


def print_cm(test_target, y_pred):
    ax = plt.subplot()
    sns.heatmap(confusion_matrix(test_target, y_pred), annot=True, cmap='Spectral', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    ax.yaxis.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    plt.show()


train_data, train_target0, test_data, test_target0, texts = get_data(0)

train_target = to_categorical(train_target0)
test_target = to_categorical(test_target0)

unique_symbols = Counter()
for _, message in texts.iteritems():
    unique_symbols.update(message)
print("Unique symbols:", len(unique_symbols))

tk = Tokenizer(num_words=len(unique_symbols), char_level=True, oov_token='UNK')
tk.fit_on_texts(texts)

char_dict = {}
i = 0
for char in unique_symbols:
    char_dict[char] = i + 1
    i += 1

tk.word_index = char_dict
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

vocab_size = len(tk.word_index)

embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

train_sequences = tk.texts_to_sequences(train_data)
test_texts = tk.texts_to_sequences(test_data)

for i, s in enumerate(train_sequences):
    print(len(s))
    if i > 4:
        break

train_data = pad_sequences(train_sequences, maxlen=500, padding='post')
test_data = pad_sequences(test_texts, maxlen=500, padding='post')

train_data = np.array(train_data)
test_data = np.array(test_data)

input_size = 500
embedding_size = vocab_size

for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)
embedding_layer = Embedding(vocab_size+1,
                            embedding_size,
                            weights=[embedding_weights],
                            input_length=input_size)

embedding_weights1 = []
embedding_weights1.append(np.zeros(vocab_size))
for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights1.append(onehot)
embedding_weights1 = np.array(embedding_weights1)
embedding_layer1 = Embedding(vocab_size+1,
                            embedding_size,
                            weights=[embedding_weights1],
                            input_length=input_size)

embedding_weights2 = []
embedding_weights2.append(np.zeros(vocab_size))
for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights2.append(onehot)
embedding_weights2 = np.array(embedding_weights2)
embedding_layer2 = Embedding(vocab_size+1,
                            embedding_size,
                            weights=[embedding_weights2],
                            input_length=input_size)

# simple model with embeddings
inputs = Input(shape=(train_data.shape[1],))
x = embedding_layer(inputs)
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
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

print_cm(test_target0, np.argmax(y_norm, axis=1))

# cnn model
inputs = Input(shape=(train_data.shape[1],), name='sent_input', dtype='int64')
x = embedding_layer1(inputs)
convolution_output = []
conv_layers = [[3, 2], [3, 3], [3, 4]]
for num_filters, filter_width in conv_layers:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=filter_width,
                         activation='relu',
                         name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
    pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
    convolution_output.append(pool)
x = Concatenate()(convolution_output)
dropout = 0.3
fully_connected_layers = [9, 6]
for fl in fully_connected_layers:
    x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(dropout)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
print(model.summary())

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
model.fit(train_data, train_target, batch_size=128, epochs=15, verbose=1, validation_split=0.2)

print('CNN')
scores = model.evaluate(test_data, test_target, verbose=2)
print(scores)

y_pred = model.predict(test_data)
y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
print(classification_report(test_target, y_norm))
print('MCC: ', matthews_corrcoef(test_target0, np.argmax(y_norm, axis=1)))
print("classes: 0-cy, 1-en, 2-ga, 3-gd")

print_cm(test_target0, np.argmax(y_norm, axis=1))

# # lstm model
# inputs = Input(shape=(train_data.shape[1],))
# x = embedding_layer2(inputs)
# # x = Flatten()(x)
# # x = Reshape((train_data.shape[1], 1))(x)
# x = LSTM(100, activation="tanh")(x)
# x = Dropout(0.3)(x)
# x = Dense(20, activation="selu")(x)
# x = Dense(units=4, activation="softmax")(x)
#
# model = Model(inputs, x)
# print(model.summary())
#
# adam = Adam(lr=0.001)
# model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
# model.fit(train_data, train_target, batch_size=128, epochs=4, verbose=1, validation_split=0.2)
#
# print('LSTM')
# scores = model.evaluate(test_data, test_target, verbose=2)
# print(scores)
#
# y_pred = model.predict(test_data)
# y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
# print(classification_report(test_target, y_norm))
# print("classes: 0-cy, 1-en, 2-ga, 3-gd")
# print('MCC: ', matthews_corrcoef(test_target0, np.argmax(y_norm, axis=1)))
#
# print_cm(test_target0, np.argmax(y_norm, axis=1))

