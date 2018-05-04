import numpy as np
import sys
from skimage import io, transform

def label_generator(names=None):
    y = {}
    label_idx = -1
    for name in names:
        label = name.split('/')[0]
        if label not in y.keys():
            label_idx = label_idx + 1
            y[label] = label_idx
    return y

def load_data(file_name=''):
    tmp = np.load(file_name)
    X_train, y_train, X_test, y_test = (tmp['X_train'], tmp['y_train'], tmp['X_test'], tmp['y_test'])

    return X_train, y_train, X_test, y_test

def generate_single_file(root_path='E:/datasets/food-101', version=(128, 128)):
    f = open(root_path + '/meta/train.txt')
    train = f.readlines()

    f = open(root_path + '/meta/test.txt')
    test = f.readlines()

    y = label_generator(train)

    X_train = []
    y_train = []
    for img_file in train:
        path = root_path+'/images/'+img_file.rstrip()+'.jpg'
        img = io.imread(path, grayscale=False)
        img = transform.resize(img, version, preserve_range=True).astype('uint8')
        X_train.append(img)
        y_ = y[img_file.split('/')[0]]
        y_train.append(y_)

    X_test = []
    y_test = []
    for img_file in test:
        path = root_path+'/images/'+img_file.rstrip()+'.jpg'
        img = io.imread(path,grayscale=False)
        img = transform.resize(img, version, preserve_range=True).astype('uint8')
        X_test.append(img)
                y_ = y[img_file.split('/')[0]]
        y_test.append(y_)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    np.savez_compressed('food-101_{}x{}'.format(version[0], version[1]), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    np.random.seed(12227)
    #generate_single_file()
    f = open('C:/Users/ARTUR/Desktop/Residuals/lulu/train.txt')
    train = f.readlines()
    y = label_generator(train)
    print('done')