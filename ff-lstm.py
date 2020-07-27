import string

import pandas as pd
from sklearn.utils import shuffle

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.layers import Embedding, Flatten
from keras import regularizers

from sklearn.metrics import classification_report
from keras.layers import LSTM, Dense, Input, Dropout
from keras import Model
from keras.layers.core import Reshape
from keras.utils import to_categorical
from keras.optimizers import Adam


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

    return train_data, train_target, test_data, test_target


train_data, train_target, test_data, test_target = get_data(0)

train_target = to_categorical(train_target)
test_target = to_categorical(test_target)

unique_symbols = Counter()
for _, message in train_data.iteritems():
    unique_symbols.update(message)
print("Unique symbols:", len(unique_symbols))

tk = Tokenizer(num_words=len(unique_symbols), char_level=True, oov_token='UNK')
tk.fit_on_texts(train_data)

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
model.fit(train_data, train_target, batch_size=512, epochs=25, verbose=1, validation_split=0.2)

scores = model.evaluate(test_data, test_target, verbose=2)
print(scores)

y_pred = model.predict(test_data)
y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
print(classification_report(test_target, y_norm))
print("classes: 0-cy, 1-en, 2-ga, 3-gd")


# lstm model
inputs = Input(shape=(train_data.shape[1],))
x = Reshape((train_data.shape[1], 1))(inputs)
x = LSTM(50, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.3)(x)
x = Dense(10, activation="relu")(x)
x = Dense(units=4, activation="softmax")(x)

model = Model(inputs, x)
print(model.summary())

adam = Adam(lr=0.01)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
model.fit(train_data, train_target, batch_size=512, epochs=4, verbose=1, validation_split=0.2)

scores = model.evaluate(test_data, test_target, verbose=2)
print(scores)

y_pred = model.predict(test_data)
y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
print(classification_report(test_target, y_norm))
print("classes: 0-cy, 1-en, 2-ga, 3-gd")
