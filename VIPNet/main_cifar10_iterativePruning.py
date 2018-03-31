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
    H = keras.layers.Dense(10)(H)
    H = keras.layers.Activation('softmax')(H)
    model = keras.models.Model(inp, H)
    return model

def load_model(architecture_file='', weights_file=''):
    print(architecture_file)

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
    return model

def save_model(file_name=''):
    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    data_augmentation = True
    save = False
    file_name = 'models/' #Used only when save is True

    X_train, y_train, X_test, y_test = func.cifar_vgg_data(debug)

    if data_augmentation == True:
        datagen = func.generate_data_augmentation(X_train)

    cnn_model = func.load_model(architecture_file='../CIFAR-10/cifar10vgg_design', weights_file='C:/Users/ARTUR/Desktop/Residuals/cifar10vgg_weights')

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

        start = time.time()
        if debug is False:
            if data_augmentation is True:
                _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
                model.fit_generator(datagen.flow(X_train, y_train),
                                    steps_per_epoch=X_train.shape[0],
                                    epochs=10*(i+1),
                                    callbacks=[reduce_lr],
                                    validation_data=(X_val, y_val), verbose=2)
            else:
                model.fit(X_train, y_train, epochs=10*(i+1), batch_size=128,
                          callbacks=[reduce_lr], validation_split=0.1, verbose=0)

            print('FC Training done')

            model = frozen_conv2D(model, True)

            sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            if data_augmentation == True:
                model.fit_generator(datagen.flow(X_train, y_train),
                                        steps_per_epoch=X_train.shape[0],
                                        epochs=max_epochs, batch_size=128,
                                        callbacks=callbacks,
                                        validation_split=0.1, verbose=2)
            else:
                model.fit(X_train, y_train, epochs=max_epochs, batch_size=128,
                          callbacks=callbacks, validation_split=0.1, verbose=2)

        end = time.time()

        if save == True:
            func.save_model(file_name+'model_iteration{}'.format(i), model)


        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        print('Number of parameters [{}] at Iteration [{}]. Accuracy of [{:.4f}] Time [{:.4f}]'.format(model.count_params(), i, acc, end-start))

        # Get the Convolutional (index 1)
        cnn_model = model.get_layer(index=1)
        cnn_model = vip_net.generate_conv_model(cnn_model)


    tmp = model.predict(X_test)
    tmp = np.argmax(tmp, axis=1)

    acc = accuracy_score(y_test, tmp)
    print('Accuracy of [{:.4f}]'.format(acc))