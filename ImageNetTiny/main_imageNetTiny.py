import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
import keras
import time
import sys
sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks

def insert_fully(cnn_model, input_shape=(64, 64, 3)):
    inp = keras.Input(input_shape)
    H = keras.models.Model(cnn_model.input, cnn_model.layers[-1].output)
    H = H(inp)
    H = keras.layers.Flatten()(H)
    H = keras.layers.Dense(512,
                           kernel_regularizer=keras.layers.regularizers.l2(0.0005))(H)
    H = keras.layers.Activation('relu')(H)
    H = keras.layers.BatchNormalization()(H)
    H = keras.layers.Dense(200)(H)
    H = keras.layers.Activation('softmax')(H)
    model = keras.models.Model(inp, H)
    return model

if __name__ == '__main__':
    np.random.seed(12227)

    X_train, y_train, X_test, y_test = func.image_net_tiny_data(subtract_pixel_mean=True)

    model_type = 'VGG16'  # 'VGG16' or 'ResNetV1'
    if model_type == 'VGG16':
        lr = 0.01
        schedule = [(25, 1e-3), (50, 1e-4), (75, 1e-5), (100, 1e-6), (150, 1e-7)]
    if model_type == 'ResNetV1':
        lr = 1e-3
        schedule = [(80, 1e-4), (120, 1e-5), (160, 1e-6), (180, 5e-7)]

    model = func.load_model(architecture_file='../architectures/imageNetTinyVGG',
                            weights_file='')

    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    save_scheduler = custom_callbacks.SavelModelScheduler(file_name='', schedule=[1, 25, 50, 75, 100, 150])
    callbacks = [lr_scheduler, save_scheduler]

    max_epochs = 200
    model = func.optimizer_compile(model, model_type)

    model.fit(X_train, y_train, epochs=max_epochs, batch_size=64, callbacks=callbacks, verbose=2)

    y_pred = model.predict(X_test)
    top1_error, top5_error = (1 - func.top_k_accuracy(y_test, y_pred, 1), 1 - func.top_k_accuracy(y_test, y_pred, 5))
    print('Top1 Error of [{:.4f}] Top 5 Error of [{:.4f}]'.format(top1_error, top5_error))
    model.save_weights('imageNetTiny'+model_type)