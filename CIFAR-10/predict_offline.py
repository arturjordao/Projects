import keras
import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score
import time

import sys
sys.path.insert(0, '../utils')

import custom_functions as func

def baseline(architecture_file, weights_file, X_train, X_test, y_train, y_test):
    if architecture_file == '':
        return

    model = func.load_model(architecture_file, weights_file)

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    cat_acc = recall_score(y_test, y_pred, average='macro')

    n_filters = func.count_filters(model)

    print('Number of Parameters [{}] Number of Filters [{}] Accuracy [{:.4f}] Categorical Accuracy [{:.4f}] Time[{:.4f}]'.format(model.count_params(), n_filters, acc, cat_acc, end - start))

if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    X_train, y_train, X_test, y_test = func.cifar_vgg_data(debug)

    baseline('', '', X_train, X_test, y_train, y_test)

    max_iterations = 20
    initial_iteration = 0
    for i in range(initial_iteration, max_iterations):

        file_name='models/model_iteration{}'.format(i)
        model = func.load_model(file_name, file_name)

        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()

        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        cat_acc = recall_score(y_test, y_pred, average='macro')

        n_filters = func.count_filters(model.get_layer(index=1))

        print('Number of Parameters [{}] Number of Filters [{}] Accuracy [{:.4f}] Categorical Accuracy [{:.4f}] Time[{:.4f}]'.format(model.count_params(), n_filters, acc, cat_acc, end - start))