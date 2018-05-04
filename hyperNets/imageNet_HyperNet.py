from sklearn.metrics.classification import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import numpy as np
import time
import keras

import sys
sys.path.insert(0, '../utils')

import custom_functions as func
from latent_hyper_net import LatentHyperNet


if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    layers = [45, 52, 55]
    n_comp = 8
    dm_method = 'pls'

    cnn_model = func.load_model(architecture_file='../architectures/imageNetVGGType2',
                                weights_file='')

    X_train, y_train, X_test, y_test = func.image_net_data(subtract_pixel_mean=True, load_train=True, load_test=True,
                                                           path='../ImageNet/', train_size=0.1)

    y_test = np.argmax(y_test, axis=1)

    id_layer = ''
    for i in range(0, len(cnn_model.layers)):
        id_layer += '['+str(i)+' '+ str(type(cnn_model.get_layer(index=i))).split('.')[-1].replace('>', '') + '] '
    print(id_layer)
    print('Layers{}'.format(layers)) if dm_method=='' else print('Layers{} Number of Components[{}] Method[{}]'.format(layers, n_comp, dm_method))

    if dm_method != '':
        hyper_net = LatentHyperNet(n_comp=n_comp, model=cnn_model, layers=layers, dm_method=dm_method)
    else:
        hyper_net = LatentHyperNet(model=cnn_model, layers=layers)

    if hyper_net.dm_method is not None:
        hyper_net.fit(X_train, y_train)
        X_train = hyper_net.transform(X_train)
        X_test = hyper_net.transform(X_test)
    else:
        X_train = hyper_net.get_features(X_train)
        X_test = hyper_net.get_features(X_test)

    inp = keras.layers.Input((X_train.shape[1],))
    H = keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0005))(inp)
    H = keras.layers.Activation('relu')(H)
    H = keras.layers.BatchNormalization()(H)
    H = keras.layers.Dropout(0.5)(H)
    H = keras.layers.Dense(1000)(H)
    H = keras.layers.Activation('softmax')(H)
    model = keras.models.Model([inp], H)
    model = func.optimizer_compile(model)
    model.fit(X_train, y_train, verbose=2, epochs=400, batch_size=X_train.shape[0])

    tmp = model.predict(X_test)
    tmp = np.argmax(tmp, axis=1)

    acc = accuracy_score(y_test, tmp)
    print('Accuracy of [{:.4f}]'.format(acc))