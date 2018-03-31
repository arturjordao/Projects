import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_lfw_pairs
from sklearn import svm
import scipy.stats as st
import time
import copy
from VIPNet import VIPNetwork
import sys
sys.path.insert(0, '../hyperNets')
sys.path.insert(0, '../classifiers')

import latent_hyper_net
import oRF as oblique_rf
import mlp_classifier as fc

from keras_vggface.vggface import VGGFace
import keras
import keras.backend as K
K.set_image_data_format('channels_last')

def gini_index(y_predict, y_expected):
    gini = []
    c1 = np.where(y_expected != 1)#Negative samples
    c2 = np.where(y_expected == 1) #Positive samples
    n = y_expected.shape[0]
    thresholds = y_predict
    for th in thresholds:

        tmp = np.where( y_predict[c1] < th )[0] #Predict correctly the negative sample
        c1c1 = tmp.shape[0]
        n1 = c1c1

        tmp = np.where( y_predict[c1] >= th )[0] #Predict the negative sample as positive
        c1c2 = tmp.shape[0]
        n2 = c1c2


        tmp = np.where( y_predict[c2] >= th )[0] #Predict correctly the positive sample
        c2c2 = tmp.shape[0]
        n2 = n2 + c2c2
        tmp = np.where( y_predict[c2] < th )[0] #Predict the positive samples as negative
        c2c1 = tmp.shape[0]
        n1 = n1 + c2c1

        if n1 == 0 or n2 == 0:
            gini.append(9999)
            continue
        else:
            gini1 = (c1c1/n1)**2 - (c2c1/n1)**2
            gini1 = 1 - gini1
            gini1 = (n1/n) * gini1

            gini2 = (c2c2/n2)**2 - (c1c2/n2)**2
            gini2 = 1 - gini2
            gini2 = (n2/n) * gini2


            gini.append(gini1+gini2)
    if len(gini)>0:
        idx = gini.index(min(gini))
        best_th = thresholds[idx]
    else:
        print('Threshold not found')
        best_th = 0

    return best_th

def face_verification(model, X_tr, Y_tr, test, gt):

    model_ = copy.copy(model)
    cnn_model = VGGFace(include_top=False, input_shape=X_tr[:, 0, :][0].shape)
    vip_net = VIPNetwork(n_comp=1, model=cnn_model, layers=[], global_pooling='max', face_verif=True)
    #vip_net = VIPNetwork(n_comp=10, model=cnn_model, layers=[], global_pooling=keras.layers.MaxPooling2D(pool_size=(2, 2)), face_verif=True)
    vip_net.fit(X_tr, Y_tr)
    cnn_model = vip_net.rebuild_net(percentage_discard=0.1)

    hyper_net = latent_hyper_net.LatentHyperNet(model=cnn_model, layers=[18])
    faces1 = hyper_net.get_features(X_tr[:, 0, :])
    faces2 = hyper_net.get_features(X_tr[:, 1, :])
    X_tr = np.abs(faces1 - faces2)
    #X_tr = (faces1 - faces2)*(faces1 - faces2)
    #X_tr = ((faces1-faces2)*(faces1-faces2) )/(faces1+faces2)
    #X_tr = np.concatenate( (np.abs(faces1-faces1), (faces1*faces2)), axis=1 )

    start = time.time()
    faces1 = hyper_net.get_features(test[:, 0, :])
    faces2 = hyper_net.get_features(test[:, 1, :])
    test = np.abs(faces1 - faces2)
    #test = (faces1 - faces2)*(faces1 - faces2)
    #test = ((faces1-faces2)*(faces1-faces2) )/(faces1+faces2)
    #test = np.concatenate( (np.abs(faces1-faces1), (faces1*faces2)), axis=1 )
    end = time.time()

    model_.fit(X_tr, Y_tr)
    pred = model_.decision_function(test)
    aux = pred

    fpr, tpr, thr = roc_curve(gt, pred, pos_label=1)
    auc = roc_auc_score(gt, pred, average='macro')

    pred = model_.predict(test)
    acc = accuracy_score(gt, pred)

    th = gini_index(model_.decision_function(X_tr), Y_tr)
    print(th)
    aux[aux >= th] = 1
    aux[aux < th ] = 0
    gini_acc = accuracy_score(gt, np.asarray(aux, dtype=np.int))

    print('Acc:[{}] Acc Gini:[{}] Auc:[{}] Time[{:.4f}]'.format(str(acc), str(gini_acc), str(auc), end-start))
    return acc, auc, fpr, tpr, thr

if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    classifier = svm.LinearSVC(C=0.1, random_state=1227)
    # classifier = QuadraticDiscriminantAnalysis()
    #classifier = fc.FullyConnected()
    # classifier = oblique_rf.ObliqueRandomForest(n_trees=100, n_fs=250, node_classifier=pls.PLSClassifier(n_comp=10))
    # classifier = oblique_rf.ObliqueRandomForest(max_depth=2, n_fs=1500, n_trees=150, node_classifier=svm.LinearSVC(C=0.1))
    # classifier = oblique_rf.ObliqueRandomForest(n_trees=15, n_fs=550, node_classifier= pls.PLSClassifier(n_comp=10, estimator=QuadraticDiscriminantAnalysis()))
    # classifier = oblique_rf.ObliqueRandomForest(n_trees=15, n_fs=550, node_classifier= pls.PLSClassifier(n_comp=10, estimator=svm.LinearSVC(C=0.1, random_state=1227)))
    # print(classifier)

    lfw = fetch_lfw_pairs('10_folds', color=True, slice_=(slice(68, 196, None), slice(77, 173, None)), resize=1.0)
    X = lfw.pairs
    X /= 255
    y = lfw.target
    del lfw

    kfold = KFold(n_splits=10)
    fold_it = 0
    avg_acc = []
    for train_idx, test_idx in kfold.split(X, y):
        print('fold {} done \n{}\n'.format(fold_it, '_' * 100))
        fold_it += 1

        if debug == True:
            idx_pos, idx_neg = (np.where(y[train_idx] == 1)[0], np.where(y[train_idx] == 0)[0])
            train_idx = np.concatenate((np.random.choice(idx_pos, 5),np.random.choice(idx_neg, 5)))

            idx_pos, idx_neg = (np.where(y[test_idx] == 1)[0], np.where(y[test_idx] == 0)[0])
            test_idx = np.concatenate((np.random.choice(idx_pos, 5),np.random.choice(idx_neg, 5)))

            # X = np.concatenate((X[train_idx], X[test_idx]))
            # y = np.concatenate((y[train_idx], y[test_idx]))
            # train_idx = list(range(0, len(train_idx)))
            # test_idx = list(range(len(train_idx), X.shape[0]))

        acc, auc, fpr, tpr, thr = face_verification(classifier, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        avg_acc.append(acc)
    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    print('Mean accuracy {} Interval confidence [{} {}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))