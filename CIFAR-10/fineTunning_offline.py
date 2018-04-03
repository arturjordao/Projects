import keras
from keras import optimizers
import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../utils')
import custom_functions as func

def callbacks(file_name):
    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2)

    if file_name != '':
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_name + 'weights.{epoch:02d}.h5',
                                                     monitor='val_acc',
                                             verbose=1, save_best_only=True)
        return [early, checkpoint]
    else:
        return [early]

if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    save = False
    file_name = '~/models++/model_iteration'  # Used only when save is True
    input_model = ''
    checkpoint_path = 'checkpoint/'

    X_train, y_train, X_test, y_test = func.cifar_vgg_data(debug)

    max_iterations = 1
    initial_iteration = 0
    for i in range(initial_iteration, max_iterations):

        model = func.load_model(input_model+'model_iteration{}'.format(i),
                                input_model+'model_iteration{}'.format(i))

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        datagen = func.generate_data_augmentation(X_train)

        lr_decay = 1e-6
        maxepoches = 200
        lr = 0.01
        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        callbacks = callbacks(checkpoint_path)

        for epoch in range(1, maxepoches):

            if epoch % 25 == 0 and epoch > 0:
                lr /= 2
                sgd = optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            model.fit_generator(datagen.flow(X_train, y_train),
                                  steps_per_epoch=X_train.shape[0],
                                  epochs=epoch,
                                  initial_epoch=epoch - 1,
                                  callbacks=callbacks,
                                  validation_data=(X_val, y_val),
                                  verbose=2)

            print('Accuracy [{:.4f}]'.format(accuracy_score(y_test, np.argmax(model.predict(X_test), axis=1))))

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred)
        n_filters = func.count_filters(model)

        print('Number of Parameters [{}] Number of Filters [{}] Accuracy [{:.4f}]'.format(model.count_params(), n_filters, acc))

        if save is True:
            func.save_model(file_name+str(i), model)