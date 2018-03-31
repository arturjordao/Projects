import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from VIPNet import VIPNetwork
import time
import sys
sys.path.insert(0, '../utils')

import custom_functions as func

def frozen_conv2D(model, trainable=False):
    #Frozen the convolutional layers
    try:
        model.get_layer(index=1).trainable = trainable
        for layer in model.get_layer(index=1).layers:
            layer.trainable = trainable
    except:
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    return model

def insert_fully(cnn_model, input_shape=(32, 32, 3)):
    inp = keras.Input(input_shape)
    H = keras.models.Model(cnn_model.input, cnn_model.layers[-1].output)
    H = H(inp)
    H = keras.layers.Flatten()(H)
    H = keras.layers.Dense(512,
                           kernel_regularizer=keras.layers.regularizers.l2(0.0005))(H)
    H = keras.layers.Activation('relu')(H)
    H = keras.layers.BatchNormalization()(H)
    H = keras.layers.Dropout(0.5)(H)
    H = keras.layers.Dense(1000)(H)
    H = keras.layers.Activation('softmax')(H)
    model = keras.models.Model(inp, H)
    return model

if __name__ == '__main__':
    np.random.seed(12227)

    file_name = 'imagenet_models/'

    cnn_model = func.load_model(architecture_file='../CIFAR-10/imageNetVGG_design')
    X_train, y_train, X_test, y_test = func.image_net_data(subtract_pixel_mean=True)

    max_iterations = 20
    for i in range(0, max_iterations):

        vip_net = VIPNetwork(n_comp=2, model=cnn_model, layers=[], global_pooling='max')
        # vip_net = VIPNetwork(n_comp=10, model=cnn_model, layers=[],
        #                      global_pooling=keras.layers.MaxPooling2D(pool_size=(2, 2)))

        vip_net.fit(X_train, y_train)
        cnn_model = vip_net.rebuild_net(percentage_discard=0.1)
        cnn_model = frozen_conv2D(cnn_model, False)
        model = insert_fully(cnn_model)

        lr_decay = 1e-6
        lr = 0.01
        max_epochs = 200

        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=10, min_lr=1e-20, verbose=2)

        early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2)

        callbacks = [reduce_lr, early]

        model.fit(X_train, y_train, epochs=10 * (i + 1), batch_size=128,
                  callbacks=[reduce_lr], validation_split=0.1, verbose=0)

        print('FC Training done')

        model = frozen_conv2D(model, True)

        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=max_epochs, batch_size=128,
                  callbacks=callbacks, validation_split=0.1, verbose=2)


        func.save_model(file_name + 'model_iteration{}'.format(i), model)

        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        print('Number of parameters [{}] at Iteration [{}]. Top1 Error of [{:.4f}]'.format(model.count_params(), i, acc))

        # Get the Convolutional (index 1)
        cnn_model = model.get_layer(index=1)
        cnn_model = vip_net.generate_conv_model(cnn_model)