import keras
import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score
import time

import sys
sys.path.insert(0, '../utils')

import custom_functions as func

def baseline(architecture_file, weights_file, X_test, y_test):
    if architecture_file == '':
        return
    model =  func.load_model(architecture_file=architecture_file,
                            weights_file=weights_file)

    n_filters = func.count_filters(model)
    filter_layer = func.count_filters_layer(model)
    flops, _ = func.compute_flops(model)

    y_pred = model.predict(X_test)
    top1_error = 1-accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    top5_error = 1-func.top_k_accuracy(y_test, y_pred, 5)
    top10_error = 1-func.top_k_accuracy(y_test, y_pred, 10)

    print('Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.
          format(model.count_params(), n_filters, flops, top1_error, top5_error, top10_error))
    print('{}'.format(filter_layer))

if __name__ == '__main__':
    np.random.seed(12227)

    X_train, y_train, X_test, y_test = func.image_net_data(subtract_pixel_mean=True, load_train=False)

    baseline('', '', X_test, y_test)

    max_iterations = 14
    initial_iteration = 2
    for i in range(initial_iteration, max_iterations):
        file_name = '../VIPNet/models_imageNet_05/model_iteration{}'.format(i)
        model = func.load_model(file_name, file_name)

        n_filters = func.count_filters(model.get_layer(index=1))
        filter_layer = func.count_filters_layer(model.get_layer(index=1))
        flops, _ = func.compute_flops(model.get_layer(index=1))

        y_pred = model.predict(X_test)
        top1_error = 1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        top5_error = 1 - func.top_k_accuracy(y_test, y_pred, 5)
        top10_error = 1 - func.top_k_accuracy(y_test, y_pred, 10)

        print('Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.
              format(model.count_params(), n_filters, flops, top1_error, top5_error, top10_error))
        print('{}'.format(filter_layer))