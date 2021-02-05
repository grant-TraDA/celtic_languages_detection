from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from keras.layers import Lambda, Input, Dense
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
# from tensorflow.python.keras.layers import Conv1DTranspose

from svm import print_cm, AverageWordLengthExtractor, CountConsonantsExtractor
import numpy as np


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size,
                 model_name="vae"):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae-features.png")
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap=plt.cm.get_cmap('Spectral', 4))
    cbar = plt.colorbar()
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(["Welsh", "English", "Irish", "Scottish"])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()


def get_data():
    file = "./data/1preproc.tsv"
    data_all = pd.read_csv(file, sep='\t', header=None, quoting=3, error_bad_lines=False)
    data_all[0] = data_all[0].astype(str)

    data = data_all.iloc[:, 0]
    target = data_all.iloc[:, 1]

    data = data.apply(lambda row: row.replace(u'\xa0', u' '))
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


x_train, y_train, x_test, y_test, texts = get_data()

unique_symbols = Counter()
for _, message in texts.iteritems():
    unique_symbols.update(message)

print("Unique symbols:", len(unique_symbols))
print(unique_symbols)

uncommon_symbols = list()

for symbol, count in unique_symbols.items():
    if count < 5:
        uncommon_symbols.append(symbol)

print("Uncommon symbols:", len(uncommon_symbols))
print(uncommon_symbols)

num_unique_symbols = len(unique_symbols)

tokenizer = Tokenizer(
    char_level=True,
    filters=None,
    lower=False,
    num_words=num_unique_symbols
    # oov_token='UNK'
)

tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(x_train)

train_data = pad_sequences(sequences)
train_target = to_categorical(y_train)

test_data = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=train_data.shape[1])
test_target = to_categorical(y_test)

print(train_data.shape)


input_shape = (train_data.shape[1], )
intermediate_dim = 16
batch_size = 128
latent_dim = 2
epochs = 20
filter_sizes = [3]
num_filters = 50
filter_width = 3
l2_reg = 0.001
sequence_length = train_data.shape[1]

# encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(100, activation='relu')(inputs)
x = Dense(60, activation='relu')(x)
x = Dense(40, activation='relu')(x)
x = Dense(intermediate_dim, activation="relu")(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# decoder model
latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(40, activation='relu')(x)
x = Dense(60, activation='relu')(x)
x = Dense(100, activation='relu')(x)
outputs = Dense(sequence_length, activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)

    reconstruction_loss = train_data.shape[1] * mse(inputs, outputs)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    adam = Adam(lr=0.01)
    vae.compile(optimizer=adam)

    vae.fit(train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_data, None))

    # vae latent representation
    plot_results(models,
                 (test_data, y_test),
                 batch_size=batch_size,
                 model_name="vae_mlp")

    # vae.save_weights('vae_mlp.h5')

    # ----------------------------------------------------------------------------
    # get latent layer output for classification features
    z_mean, _, _ = encoder.predict(train_data, batch_size=batch_size)
    z_mean_test, _, _ = encoder.predict(test_data, batch_size=batch_size)
    z_mean = pd.DataFrame(z_mean)
    z_mean_test = pd.DataFrame(z_mean_test)

    # statistical features
    NGRAM_MIN = 2
    NGRAM_MAX = 3
    analyzer = 'char_wb'
    fu = FeatureUnion([
        # ('ave', AverageWordLengthExtractor()),
        # ('cons', CountConsonantsExtractor()),
        ('vect', CountVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX), analyzer=analyzer))
    ])
    fu.fit(texts)
    train_data = pd.concat([z_mean, pd.DataFrame(fu.transform(x_train).toarray())], axis=1)
    test_data = pd.concat([z_mean_test, pd.DataFrame(fu.transform(x_test).toarray())], axis=1)

    # ff model
    inputs = Input(shape=(train_data.shape[1],))
    x = Dense(100, activation="relu")(inputs)
    x = Dense(40, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(units=4, activation="softmax")(x)

    model = Model(inputs, x)
    print(model.summary())

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
    model.fit(train_data, train_target, batch_size=128, epochs=15, verbose=1, validation_split=0.2)

    print('FF')
    scores = model.evaluate(test_data, test_target, verbose=2)
    print(scores)

    y_pred = model.predict(test_data)
    y_norm = np.stack([to_categorical(np.asarray(x.argmax()), num_classes=4) for x in y_pred])
    print(classification_report(test_target, y_norm))
    print('MCC: ', matthews_corrcoef(y_test, np.argmax(y_norm, axis=1)))
    print("classes: 0-cy, 1-en, 2-ga, 3-gd")

    print_cm(y_test, np.argmax(y_norm, axis=1))

    # SVM
    print('SVM')
    svm = SVC(kernel='linear', C=1)
    text_clf_svm = svm.fit(train_data, y_train)

    y_pred = text_clf_svm.predict(test_data)
    SVM_acc = np.mean(y_pred == y_test)
    print("SVM accuracy ", SVM_acc)

    print('test set')
    print(classification_report(y_test, y_pred))
    print('MCC: ', matthews_corrcoef(y_test, y_pred))

    print_cm(y_test, y_pred)

