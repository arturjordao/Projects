import numpy as np

def load_model(architecture_file='', weights_file=''):
    import keras

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load model [{}]. Load weights [{}]'.format(architecture_file, weights_file))
    else:
        print('Load model [{}]'.format(architecture_file))

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

def optimizer_compile(model, model_type='VGG16'):
    import keras
    if model_type == 'VGG16':
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if model_type == 'ResNetV1':
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    return model

def lr_schedule(epoch, init_lr=0.01, schedule=[(25, 0.001), (50, 0.0001)]):

    for i in range(0, len(schedule)-1):
        if epoch > schedule[i][0] and epoch < schedule[i+1][0]:
            print('Learning rate: ', schedule[i][0])
            return schedule[i][0]

    if epoch > schedule[-1][0]:
        print('Learning rate: ', schedule[-1][0])
        return schedule[-1][1]

    print('Learning rate: ', init_lr)
    return init_lr

def image_net_data(load_train=True, load_test=True, subtract_pixel_mean=False,
                   path='', train_size=1.0, test_size=1.0):
    import keras
    from sklearn.model_selection import train_test_split
    X_train, y_train, X_test, y_test = (None, None, None, None)
    if load_train is True:
        tmp = np.load(path+'imagenet_train.npz')
        X_train = tmp['X']
        y_train = tmp['y']

        if train_size != 1.0:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)

        X_train = X_train.astype('float32') / 255
        y_train = keras.utils.to_categorical(y_train, 1000)

    if load_test is True:
        tmp = np.load(path + 'imagenet_val.npz')
        X_test = tmp['X']
        y_test = tmp['y']

        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

        X_test = X_test.astype('float32') / 255
        y_test = keras.utils.to_categorical(y_test, 1000)

    if subtract_pixel_mean is True:
        X_train_mean = np.load(path + 'x_train_mean.npz')['X']

        if load_train is True:
            X_train -= X_train_mean#X_train_mean = np.mean(X_train, axis=0)
        if load_test is True:
            X_test -= X_train_mean
    print('#Training Samples [{}]'.format(X_train.shape[0])) if X_train is not None else print('#Training Samples [0]')
    print('#Testing Samples [{}]'.format(X_test.shape[0])) if X_test is not None else print('#Testing Samples [0]')
    return X_train, y_train, X_test, y_test

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
    #Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) == True:
            config = layer.get_config()
            n_filters+=config['filters']

    #Todo: Model contains Conv and Fully Connected layers
    # for layer_idx in range(1, len(model.get_layer(index=1))):
    #     layer = model.get_layer(index=1).get_layer(index=layer_idx)
    #     if isinstance(layer, keras.layers.Conv2D) == True:
    #         config = layer.get_config()
    #     n_filters += config['filters']
    return n_filters

def count_filters_layer(model):
    import keras
    n_filters = ''
    #Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) is True:
            config = layer.get_config()
            n_filters+=str(config['filters']) + ' '

    return n_filters

def compute_flops(model):
    import keras
    total_flops =0
    flops_per_layer = []

    for layer_idx in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            flops = output_map_H*output_map_W*previous_layer_depth*current_layer_depth*kernel_H*kernel_W
            total_flops += flops
            flops_per_layer.append(flops)


    # if total_flops / 1e9 > 1:  # for Giga Flops
    #     print(total_flops / 1e9, '{}'.format('GFlops'))
    # else:
    #     print(total_flops / 1e6, '{}'.format('MFlops'))
    return total_flops, flops_per_layer

def gini_index(y_predict, y_expected):
    gini = []
    c1 = np.where(y_expected != 1)#Negative samples
    c2 = np.where(y_expected == 1) #Positive samples
    n = y_expected.shape[0]
    thresholds = y_predict
    for th in thresholds:

        tmp = np.where( y_predict[c1] < th )[0] #Predict correctly the negative sample
        c1c1 = tmp.shape[0]
        n1 = c1c1

        tmp = np.where( y_predict[c1] >= th )[0] #Predict the negative sample as positive
        c1c2 = tmp.shape[0]
        n2 = c1c2

        tmp = np.where( y_predict[c2] >= th )[0] #Predict correctly the positive sample
        c2c2 = tmp.shape[0]
        n2 = n2 + c2c2
        tmp = np.where( y_predict[c2] < th )[0] #Predict the positive samples as negative
        c2c1 = tmp.shape[0]
        n1 = n1 + c2c1

        if n1 == 0 or n2 == 0:
            gini.append(9999)
            continue
        else:
            gini1 = (c1c1/n1)**2 - (c2c1/n1)**2
            gini1 = 1 - gini1
            gini1 = (n1/n) * gini1

            gini2 = (c2c2/n2)**2 - (c1c2/n2)**2
            gini2 = 1 - gini2
            gini2 = (n2/n) * gini2


            gini.append(gini1+gini2)
    if len(gini)>0:
        idx = gini.index(min(gini))
        best_th = thresholds[idx]
    else:
        print('Gini threshold not found')
        best_th = 0

    return best_th

def top_k_accuracy(y_true, y_pred, k):
    top_n = np.argsort(y_pred, axis=1)[:,-k:]
    idx_class = np.argmax(y_true, axis=1)
    hit = 0
    for i in range(idx_class.shape[0]):
      if idx_class[i] in top_n[i,:]:
        hit = hit + 1
    return float(hit)/idx_class.shape[0]