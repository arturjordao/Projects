import numpy as np
import sklearn
import copy

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Model
from keras.layers import Input


class LatentHyperNet(BaseEstimator, ClassifierMixin):
    __name__ = 'Latent Hyper Net'

    def __init__(self, n_iter=1500, eps=1e-6, n_comp=2, mode='regression', dm_method=None, model=None, layers=None):
        self.n_iter = n_iter
        self.eps = eps
        self.n_comp = n_comp
        self.mode = mode
        self.dm_layer = []
        self.dm_method = dm_method
        self.model = self.custom_model(model=model, layers=layers)
        self.layers = layers

    def custom_model(self, model, layers):
        input_shape = model.input_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3])
        inp = Input(input_shape)
        feature_maps = [Model(model.input, model.get_layer(index=i).output)(inp) for i in layers]
        model = Model(inp, feature_maps)
        return model

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError()

        #self.classes_, target = np.unique(y, return_inverse=True)
        target = y
        target[target == 0] = -1
        if self.dm_method == 'lda':
            target = np.argmax(target, axis=1)

        X = self.extract_features(X)

        if self.dm_method == 'pls':
            dm = PLSRegression(n_components=self.n_comp, scale=True, max_iter=self.n_iter, tol=self.eps)
        elif self.dm_method == 'pca':
            dm = PCA(self.n_comp)
        elif self.dm_method == 'lda':
            dm = LinearDiscriminantAnalysis()

        for layer_idx in range(0, len(self.layers)):
            dm_ = copy.copy(dm)
            dm_.fit(X[layer_idx], target)
            self.dm_layer.append(dm_)
            del dm_

        return self

    def transform(self, x):
        import numpy as np
        proj_x = None

        x = self.extract_features(x)

        for layer_idx in range(0, len(self.layers)):
            if proj_x is None:
                proj_x = self.dm_layer[layer_idx].transform(x[layer_idx])
            else:
                proj_tmp = self.dm_layer[layer_idx].transform(x[layer_idx])
                proj_x = np.column_stack((proj_x, proj_tmp))

        return proj_x

    def extract_features(self, X, verbose=False):
        import time
        feat_layers = [[] for x in range(0, len(self.layers))]

        idx_sample = 0
        for sample in X:
            start = time.time()
            sample = np.expand_dims(sample, axis=0)
            feat = self.model.predict(sample)
            for layer in range(0, len(self.layers)):
                feat_layers[layer].append(np.reshape(feat[layer], -1))

            if verbose == True:
                print('Extracting features {}/{} Time[{}]'.format(idx_sample, len(X), time.time() - start))
            idx_sample = idx_sample + 1

        return feat_layers

    def get_features(self, X):
        features = self.extract_features(X)
        X = None
        for layer_idx in range(0, len(self.layers)):
            if X is None:
                X = features[layer_idx]
            else:
                X_tmp = features[layer_idx]
                X = np.column_stack((X, X_tmp))
        X = np.array(X)
        return X

    def save_dm_models(self, file_name):
        import pickle
        pickle.dump(self.dm_layer, open(file_name, "wb"))

    def load_dm_models(self, file_name):
        import pickle
        self.dm_layer = pickle.load(open(file_name, 'rb'))
        self.dm_method = 'dm is not none'

#if __name__ == '__main__':
    # from sklearn.model_selection import KFold
    # from sklearn.datasets import fetch_lfw_pairs
    # lfw = fetch_lfw_pairs('10_folds', color=True, slice_=(slice(68, 196, None), slice(77, 173, None)), resize=1.0)
    # X = lfw.pairs
    # X /= 255
    #
    # y = lfw.target
    # kfold = KFold(n_splits=2)
    # import sklearn
    # from keras_vggface.vggface import VGGFace
    #
    # model = VGGFace(include_top=False, input_shape=(128, 96, 3), pooling='avg')
    #
    # for train_idx, test_idx in kfold.split(X, y):
    #     train_idx = [3000, 3001, 3002, 5997, 5998, 5999]
    #     test_idx = [   0,    1,    2, 2997, 2998, 2999]
    #
    #     hn_pls = HyperNetPLS(n_comp=2, n_iter=500, model=model, layers=[14, 18])
    #     hn_pls.fit(X[train_idx], y[train_idx])
    #     hn_pls.project(X[test_idx])