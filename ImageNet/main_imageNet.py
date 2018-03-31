import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
import keras
import time
import sys
sys.path.insert(0, '../utils')
import custom_functions as func

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

    X_train, y_train, X_test, y_test = func.image_net_data(subtract_pixel_mean=True)
    cnn_model = func.load_model(architecture_file='../CIFAR-10/cifar10vgg_design')

    for i in range(0, 7):
        cnn_model.pop()
    model = insert_fully(cnn_model, (32, 32, 3))

    lr_decay = 1e-6
    lr = 0.01
    max_epochs = 200
    for epoch in range(1, max_epochs):

        if epoch % 20 == 0 and epoch > 0:
            lr /= 2
            sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=2)
        top1_error = 1-accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
        print('Top1 Error of [{:.4f}] at Iteration [{}]'.format(top1_error, epoch))
        model.save_weights('imageNetVGG{}.h5'.format(epoch))