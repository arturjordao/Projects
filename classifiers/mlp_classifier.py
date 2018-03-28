import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, LeakyReLU, Dropout
from keras.models import Model

class FullyConnected(BaseEstimator, ClassifierMixin):
    __name__ = 'FC classifier'

    def __init__(self, n_nodes=1024, inp=None, lrelu = 0.01, dropout_rate = 0.4):
        self.n_nodes = n_nodes
        self.lrelu = lrelu
        self.dropout_rate = dropout_rate

        self.model = None

    def fit(self, X, y):
        inp = Input((X.shape[1],))
        H = Dense(self.n_nodes)(inp)
        H = LeakyReLU(self.lrelu)(H)
        H = Dropout(self.dropout_rate)(H)
        H = Dense(1, activation='sigmoid')(H)
        self.model = Model([inp], H)
        self.model.compile(Adam(0.005), binary_crossentropy, metrics=[binary_accuracy])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-20, verbose=0)
        early = EarlyStopping(monitor='val_loss', patience=8, verbose=2)

        self.model.fit([X], y, batch_size=32, epochs=250, verbose=0, shuffle=True,
                  callbacks=[reduce_lr, early], validation_split=0.1)

        return self

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def decision_function(self, X):
        pred = self.model.predict([X], batch_size=32)
        pred = pred[:, 0]
        return pred

