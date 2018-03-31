import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.classification import accuracy_score
import copy
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_lfw_pairs
import scipy.stats as st
import time
import keras.backend as K
from keras.layers import Input, Dense, LeakyReLU, Concatenate, Dropout, Conv2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
K.set_image_data_format('channels_last')
from layer_operations import *

import sys
sys.path.insert(0, '../utils')

import custom_functions as func
from VIPNet import VIPNetwork

def frozen_conv2D(model, trainable=False, layers=None):
    #Frozen the convolutional layers
    idx = 0
    try:
        model.get_layer(index=2).trainable = trainable
        for layer in model.get_layer(index=2).layers:
            if layer is None:
                layer.trainable = trainable
            if layer is not None and idx in layers:
                layer.trainable = trainable

            idx = idx + 1
    except:
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
            idx = idx + 1

    return model

def insert_fully(cnn_model, inp_shape):
    lrelu = 0.01
    dropout_rate = 0.3

    inp_a = Input(inp_shape)
    inp_b = Input(inp_shape)

    cnn_model = cnn_model
    cnn_model = Model(cnn_model.input, cnn_model.get_layer(index=18).output)

    stack_a = cnn_model(inp_a)
    stack_b = cnn_model(inp_b)

    stack_a = Flatten()(stack_a)
    stack_b = Flatten()(stack_b)
    diff = Diff()([stack_a, stack_b])

    H = Abs(name='abs_diff')(diff)

    H = Dense(1024)(H)
    H = LeakyReLU(lrelu)(H)
    H = Dropout(dropout_rate)(H)
    out = Dense(1, activation='sigmoid')(H)

    model = Model([inp_a, inp_b], out)
    return model

def face_verification(X_train, y_train, X_test, y_test):
    input_shape = X_train[:, 0, :][0].shape
    cnn_model = VGGFace(include_top=False, input_shape=input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-20, verbose=2)
    early = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
    callbacks = [reduce_lr, early]

    max_iterations = 5
    for i in range(0, max_iterations):
        vip_net = VIPNetwork(n_comp=2, model=cnn_model, layers=[], global_pooling='max', face_verif=True)
        # vip_net = VIPNetwork(n_comp=10, model=cnn_model, layers=[],face_verif=True
        #                      global_pooling=keras.layers.MaxPooling2D(pool_size=(2, 2)))

        vip_net.fit(X_train, y_train)
        cnn_model = vip_net.rebuild_net(percentage_discard=0.1)
        cnn_model = frozen_conv2D(cnn_model, False)

        model = insert_fully(cnn_model, input_shape)
        model.compile(Adam(0.005), binary_crossentropy, metrics=[binary_accuracy])
        model.fit([X_train[:, 0, :], X_train[:, 1, :]], y_train, batch_size=32, epochs=250,
               verbose=0, shuffle=True, callbacks=callbacks, validation_split=0.1)

        print('FC Training done')

        model = frozen_conv2D(model, True, layers=[15, 16, 17])
        model.compile(Adam(0.05), binary_crossentropy, metrics=[binary_accuracy])
        model.fit([X_train[:, 0, :], X_train[:, 1, :]], y_train, batch_size=32, epochs=250,
                  verbose=2, shuffle=True, callbacks=callbacks, validation_split=0.1)

        pred = model.predict([X_test[:, 0, :],X_test[:, 1, :]])

        auc = roc_auc_score(y_test, pred, average='macro')

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        acc = accuracy_score(y_test, pred)

        print('Number of parameters [{}] at Iteration [{}]. Accuracy of [{:.4f}]  AUC of [{:.4f}]'.format(
            model.count_params(), i, acc, auc))

        # Get the Convolutional (index 1)
        cnn_model = model.get_layer(index=2)
        cnn_model = vip_net.generate_conv_model(cnn_model)

    return acc

if __name__ == '__main__':
    np.random.seed(12227)
    debug = True
    lfw = fetch_lfw_pairs('10_folds', color=True, slice_=(slice(68, 196, None), slice(77, 173, None)), resize=1.0)
    X = lfw.pairs
    X /= 255

    y = lfw.target
    kfold = KFold(n_splits=10)
    fold_it = 0
    avg_acc = []
    for train_idx, test_idx in kfold.split(X, y):

        print('fold {} done \n{}\n'.format(fold_it, '_' * 100))
        fold_it += 1

        if debug is True:
            train_idx = np.array([600,  601,  602, 5697, 5698, 5699,  900,  901,  902, 5997, 5998, 5999])
            test_idx = np.array([0,   1,   2,   3,   4,   5, 303, 304, 305, 306, 307, 308, 309, 310, 311])

        acc = face_verification(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        avg_acc.append(acc)

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    print('Mean accuracy {} Interval confidence [{} {}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
