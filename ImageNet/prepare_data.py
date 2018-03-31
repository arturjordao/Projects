import numpy as np
import sys

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load(file_name='', img_size=32):
    import os
    import pickle

    d = unpickle(file_name)
    x = d['data']
    y = d['labels']
    #mean_image = d['mean']

    #x = x/np.float32(255)
    #mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    #x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)

    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    return X_train, Y_train

def train_data(path='E:/datasets/ImageNet', batch=10):
    X_train = None
    y_train=None
    for i in range(1, batch+1):
        file_name = path+'train_data_batch_'+str(i)
        if X_train is not None:
            X_batch, y_batch = load(file_name=file_name)
            X_train = np.concatenate((X_train, X_batch), axis=0)
            y_train = np.concatenate((y_train, y_batch), axis=0)
        else:
            X_train, y_train = load(file_name=file_name)

    print('Training data')
    print(X_train.shape)
    print(y_train.shape)
    np.savez_compressed('imagenet_train', X=X_train, y=y_train)

def validation_data(path='E:/datasets/ImageNet'):
    X_train, y_train = load(path+'val_data')
    print('Validation data')
    print(X_train.shape)
    print(y_train.shape)
    np.savez_compressed('imagenet_val', X=X_train, y=y_train)

if __name__ == '__main__':
    np.random.seed(12227)
    validation_data()
    train_data()