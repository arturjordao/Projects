import numpy as np
import copy
import time
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization
from keras.layers import Input
from keras.models import Model
import os.path
import sys

class VIPNetwork(BaseEstimator, ClassifierMixin):
    __name__ = 'VIP Network'

    def __init__(self, n_comp=2, model=None, layers=None, global_pooling='max', face_verif=False):
        if len(layers) == 0:
            self.layers = list(range(1, len(model.layers)))#Starts by one since the 0 index is the Input
        else:
            self.layers = layers

        if global_pooling == 'max':
            self.pool = GlobalMaxPooling2D()
        elif global_pooling == 'avg':
            self.pool = GlobalAveragePooling2D()
        else:
            self.pool = global_pooling

        self.n_comp = n_comp
        self.scores = None
        self.score_layer = None
        self.idx_score_layer = []
        self.template_model = model
        self.conv_net = self.custom_model(model=model, layers=self.layers)
        self.face_verif = face_verif

    def custom_model(self, model, layers):
        input_shape = model.input_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3])
        inp = Input(input_shape)

        feature_maps = [Model(model.input, self.pool(model.get_layer(index=i).output))(inp) for i in layers if isinstance(model.get_layer(index=i), Conv2D)]

        self.layers = list(range(0, len(feature_maps)))
        model = Model(inp, feature_maps)
        return model

    def flatten(self, features):
        n_samples = features[0].shape[0]
        X = None
        for layer_idx in range(0, len(self.layers)):
            if X is None:
                X = features[layer_idx].reshape((n_samples,-1))
                self.idx_score_layer.append((0, X.shape[1]-1))
            else:
                X_tmp = features[layer_idx].reshape((n_samples,-1))
                self.idx_score_layer.append((X.shape[1], X.shape[1]+X_tmp.shape[1] - 1))
                X = np.column_stack((X, X_tmp))

        X = np.array(X)
        return X

    def fit(self, X, y):
        if self.face_verif == True:
            faces1 = self.conv_net.predict(X[:, 0, :])
            faces2 = self.conv_net.predict(X[:, 1, :])
            faces1 = self.flatten(faces1)
            faces2 = self.flatten(faces2)
            X = np.abs(faces1 - faces2)#Make lambda function
        else:
            X = self.conv_net.predict(X)
            X = self.flatten(X)

        pls_model = PLSRegression(n_components=self.n_comp, scale=True)
        pls_model.fit(X, y)
        self.scores = self.vip(X, y, pls_model)
        self.score_by_filter()

        return self

    def vip(self, x, y, model):
        # Adapted from https://github.com/scikit-learn/scikit-learn/issues/7050
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_

        m, p = x.shape
        _, h = t.shape

        vips = np.zeros((p,))

        # s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        s = np.diag(np.dot(np.dot(np.dot(t.T, t), q.T), q)).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            #vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
            vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)

        return vips

    def th_to_discard(self, percentage=0.1):
        scores = self.score_layer[0]
        for i in range(1, len(self.score_layer)):
            scores = np.concatenate((scores, self.score_layer[i]))

        total = scores.shape[0]
        closest = np.zeros(total)
        for i in range(0, total):
            th = scores[i]
            idxs = np.where(scores <= th)[0]
            discarded = len(idxs) / total
            closest[i] = abs(percentage - discarded)

        th = scores[np.argmin(closest)]
        return th

    def score_by_filter(self):
        model = self.template_model
        self.score_layer = []
        idx_Conv2D = 0

        for layer_idx in range(1, len(model.layers)):

            layer = model.get_layer(index=layer_idx)

            if isinstance(layer, Conv2D):
                weights = layer.get_weights()

                n_filters = weights[0].shape[3]

                begin, end = self.idx_score_layer[idx_Conv2D]
                score_layer = self.scores[begin:end + 1]
                features_filter = int((len(self.scores[begin:end]) + 1) / n_filters)

                score_filters = np.zeros((n_filters))
                for filter_idx in range(0, n_filters):
                    score_filters[filter_idx] = np.mean(score_layer[filter_idx:filter_idx + features_filter])

                self.score_layer.append(score_filters)
                idx_Conv2D = idx_Conv2D + 1

        return self

    def rebuild_net(self, percentage_discard=0.1):
        idx_Conv2D = 0
        n_discarded_filters = 0
        total_filters = 0
        model = self.template_model
        inp = (model.inputs[0].shape.dims[1].value,
               model.inputs[0].shape.dims[2].value,
               model.inputs[0].shape.dims[3].value)

        th = self.th_to_discard(percentage=percentage_discard)

        H = Input(inp)
        inp = H
        idx_previous = []
        for layer_idx in range(1, len(model.layers)):

            layer = model.get_layer(index=layer_idx)
            config = layer.get_config()

            if isinstance(layer, MaxPooling2D):
                H = MaxPooling2D.from_config(config)(H)

            if isinstance(layer, Dropout):
                H = Dropout.from_config(config)(H)

            if isinstance(layer, Activation):
                H = Activation.from_config(config)(H)

            if isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                weights[0] = np.delete(weights[0], idx_previous)
                weights[1] = np.delete(weights[1], idx_previous)
                weights[2] = np.delete(weights[2], idx_previous)
                weights[3] = np.delete(weights[3], idx_previous)
                H = BatchNormalization(weights=weights)(H)

            elif isinstance(layer, Conv2D):
                weights = layer.get_weights()

                n_filters = weights[0].shape[3]
                total_filters = total_filters + n_filters

                score_filters = self.score_layer[idx_Conv2D]
                idx_Conv2D = idx_Conv2D+1

                idxs = np.where(score_filters <= th)[0]

                weights[0] = np.delete(weights[0], idxs, axis=3)
                weights[1] = np.delete(weights[1], idxs)
                n_discarded_filters += len(idxs)
                if len(idx_previous) != 0:
                    weights[0] = np.delete(weights[0], idx_previous, axis=2)

                config['filters'] = weights[1].shape[0]
                H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       # config=config['config'],
                       # scale=config['scale'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)

            idx_previous = idxs
        print('Percentage of discarded filters {}'.format(n_discarded_filters/float(total_filters)))
        return Model(inp, H)

    def generate_conv_model(self, model):
        inp = (model.inputs[0].shape.dims[1].value,
               model.inputs[0].shape.dims[2].value,
               model.inputs[0].shape.dims[3].value)

        H = Input(inp)
        inp = H

        for layer_idx in range(1, len(model.layers)):

            layer = model.get_layer(index=layer_idx)
            config = layer.get_config()

            if isinstance(layer, MaxPooling2D):
                H = MaxPooling2D.from_config(config)(H)

            if isinstance(layer, Dropout):
                H = Dropout.from_config(config)(H)

            if isinstance(layer, Activation):
                H = Activation.from_config(config)(H)

            if isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                H = BatchNormalization(weights=weights)(H)

            elif isinstance(layer, Conv2D):
                weights = layer.get_weights()

                config['filters'] = weights[1].shape[0]
                H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)
        return Model(inp, H)