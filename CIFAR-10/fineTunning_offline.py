import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from sklearn.metrics.classification import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import time

import sys
sys.path.insert(0, '../utils')
import custom_functions as func

if __name__ == '__main__':
    np.random.seed(12227)

    debug = True
    save = False
    file_name = '~/models++/model_iteration'  # Used only when save is True

    X_train, y_train, X_test, y_test = func.cifar_vgg_data(debug)

    max_iterations = 20
    initial_iteration = 0
    for i in range(initial_iteration, max_iterations):

        model = func.load_model('models/model_iteration{}'.format(i),
                                'models/model_iteration{}'.format(i))

        datagen = func.generate_data_augmentation(X_train)

        lr_decay = 1e-6
        maxepoches = 200
        lr = 0.01
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=10, min_lr=1e-20, verbose=2)
        early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5',
                                     monitor='val_acc',
                                     verbose=1, save_best_only=True)
        callbacks = [reduce_lr, early, checkpoint]
        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        _, X_val, _, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)

        start = time.time()
        model.fit_generator(datagen.flow(X_train, y_train),
                            steps_per_epoch=X_train.shape[0],
                            epochs=maxepoches,
                            callbacks=callbacks,
                            validation_data=(X_val, y_val), verbose=2)
        end = time.time()

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred)
        cat_acc = recall_score(y_test, y_pred, average='macro')
        n_filters = func.count_filters(model.get_layer(index=1))

        print('Number of Parameters [{}] Number of Filters [{}] Accuracy [{:.4f}] Categorical Accuracy [{:.4f}] Time[{:.4f}]'.format(model.count_params(), n_filters, acc, cat_acc, end - start))

        if save is True:
            func.save_model(file_name+str(i), model)