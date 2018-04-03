import keras
import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score
import time

import sys
sys.path.insert(0, '../utils')

import custom_functions as func

if __name__ == '__main__':
    np.random.seed(12227)

    X_train, y_train, X_test, y_test = func.image_net_data(subtract_pixel_mean=True, load_train=False)
    model = func.load_model(architecture_file='../architectures/imageNetVGGType2',
                            weights_file='../weights/')

    y_pred = model.predict(X_test)
    top1_error = 1-accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    top5_error = 1-func.top_k_accuracy(y_test, y_pred, 5)
    top10_error = 1-func.top_k_accuracy(y_test, y_pred, 10)
    print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1_error, top5_error, top10_error))