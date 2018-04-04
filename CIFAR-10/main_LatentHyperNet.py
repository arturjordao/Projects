from sklearn.metrics.classification import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import numpy as np
import time
import keras

import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '../hyperNets')

import custom_functions as func
from latent_hyper_net import LatentHyperNet


if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    layers = [44, 51]
    n_comp = 19
    dm_method = 'pls'

    X_train, y_train, X_test, y_test = func.cifar_vgg_data(debug)


    cnn_model = func.load_model(architecture_file='../architectures/cifar10VGG',
                                weights_file='')

    print('Layers{}'.format(layers)) if dm_method=='' else print('Layers{} Number of Components[{}] Method[{}]'.format(layers, n_comp, dm_method))

    if dm_method != '':
        hyper_net = LatentHyperNet(n_comp=n_comp, model=cnn_model, layers=layers, dm_method=dm_method)
    else:
        hyper_net = LatentHyperNet(model=model, layers=layers)

    if hyper_net.dm_method is not None:
        hyper_net.fit(X_train, y_train)
        X_train = hyper_net.transform(X_train)
        X_test = hyper_net.transform(X_test)
    else:
        X_train = hyper_net.get_features(X_train)
        X_test = hyper_net.get_features(X_test)

    model = LinearSVC(random_state=0)
    model = OneVsRestClassifier(model).fit(X_train, y_train)
    model.fit(X_train, y_train)

    tmp = model.predict(X_test)
    tmp = np.argmax(tmp, axis=1)

    acc = accuracy_score(y_test, tmp)
    print('Accuracy of [{:.4f}]'.format(acc))