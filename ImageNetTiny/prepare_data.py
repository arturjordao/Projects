import numpy as np
import sys
import os
from skimage import io, transform

def label_generator(names=None):
    y = {}
    label_idx = -1
    for label in names:
        if label not in y.keys():
            label_idx = label_idx + 1
            y[label] = label_idx
    return y

def train_data(path='E:/datasets/tiny-imagenet-200/train'):
    X_train = []
    y_train= []
    dirnames = os.listdir(path)
    for dir in dirnames:
        images = os.listdir(path+'/'+dir+'/images')
        for image in images:
            img = io.imread(path+'/'+dir+'/images/'+image, grayscale=False)
            if img.shape != (64, 64, 3):
                img = np.stack((img,) * 3, -1)
            X_train.append(img)
            y_train.append(dir)


    X_train = np.array(X_train)
    y_label = label_generator(y_train)
    y_train = [y_label[x] for x in y_train]
    y_train = np.array(y_train)
    print('Training data')
    print(X_train.shape)
    print(y_train.shape)
    np.savez_compressed('imagenetTY_train', X=X_train, y=y_train, label=y_label)

def validation_data(path='E:/datasets/tiny-imagenet-200/val'):
    X_val = []
    y_val= []
    f = open(path + '/val_annotations.txt')
    images = f.readlines()
    for image in images:
        label = image.split('	')[1]
        img_name = image.split('	')[0]
        img = io.imread(path + '/images/' + img_name, grayscale=False)
        if img.shape != (64, 64, 3):
            img = np.stack((img,) * 3, -1)
        X_val.append(img)
        y_val.append(label)

    X_val = np.array(X_val)
    y_label = np.load('imagenetTY_train.npz')['label']
    y_val = [y_label.item()[x] for x in y_val]
    y_val = np.array(y_val)
    print('Validation (Testing) data')
    print(X_val.shape)
    print(y_val.shape)
    np.savez_compressed('imagenetTY_val', X=X_val, y=y_val, label=y_label)

if __name__ == '__main__':
    np.random.seed(12227)
    #validation_data()
    #train_data()
    tmp = np.load('imagenetTY_train.npz')
    X_train, y_train, label = (tmp['X'], tmp['y'], tmp['label'])
    tmp = np.load('imagenetTY_val.npz')
    X_test, y_test = (tmp['X'], tmp['y'])
    np.savez_compressed('imageNetTiny', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, label=label)