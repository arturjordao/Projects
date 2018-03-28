import numpy as np

def load_model(architecture_file='', weights_file=''):
    import keras
    print('Load model [{}]. Load weights [{}]'.format(architecture_file, weights_file))

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
    return model

def save_model(file_name='', model=None):
    import keras
    print('Salving model and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

def cifar_vgg_data(debug=True):
    import keras
    print('Debuging Mode') if debug is True else print('Real Mode')

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_test = np.argmax(y_test, axis=1)

    mean = 120.707
    std = 64.15
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, y_train, X_test, y_test

def cifar_resnet_data(debug=True):
    import keras
    print('Debuging Mode') if debug is True else print('Real Mode')

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_test = np.argmax(y_test, axis=1)

    return x_train, y_train, x_test, y_test

def generate_data_augmentation(X_train):
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)
    return datagen

def count_filters(model):
    import keras
    n_filters = 0
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) == True:
            config = layer.get_config()
            n_filters+=config['filters']

    return n_filters