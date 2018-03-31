import numpy as np
import cv2
import os
from six.moves import range

try:
    import cPickle as pickle
except ImportError:
    import pickle


def ytf_rescale(frame, debug=False):
    import cv2

    rescaled = cv2.resize(frame, (200, 200))
    rescaled = rescaled[50:150, 50:150, :]
    rescaled = np.asarray(rescaled, dtype=np.float32)

    if debug:
        cv2.imshow('rescaled ytf', rescaled)
        cv2.waitKey(0)

    # if center_data:
    #     # Zero-center by mean pixel
    #     rescaled[:, :, 2] -= 93.5940  # Blue
    #     rescaled[:, :, 1] -= 104.7624  # Green
    #     rescaled[:, :, 0] -= 129.1863  # Red
    #
    # if scale_data:
    #     rescaled /= 255

    return rescaled


def fetch_ytf_pairs(ytf_root='', file_path='', img_net_format=False):
    import os

    X = [[[None]] * 2] * 5000
    y = np.array([[0] * 250 + [1] * 250] * 10)

    with open(file_path, 'r') as splits_f:
        splits_f.readline()

        for i, line in enumerate(splits_f):
            # split number, pair number in split, first name, second name, is same:
            names = [None] * 2
            split_num, pair_num, names[0], names[1], same = line.strip().split(',')
            for pair_idx in range(2):
                for root, dirs, files in os.walk('{}/{}/'.format(ytf_root, names[pair_idx].strip())):
                    for file in files:
                        frame = cv2.imread(os.path.join(root, file))
                        frame = ytf_rescale(frame, False)
                        if img_net_format:
                            frame = cv2.resize(frame, (224, 224))
                        frame = frame[:, :, ::-1]
                        X[i][pair_idx].append(frame)

    return X, y


def extract_vgg_faces_features(path='e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\',
                               img_net_format=False, center_data=True,
                               scale_data=True, vgg_layer='pool5'):
    from keras_vggface.vggface import VGGFace
    from keras.models import Model
    import os
    import cv2

    ytf_root = path
    file_path = path + 'splits.txt'

    import keras.backend as K
    import tensorflow as tf
    K.clear_session()

    if img_net_format:
        rows, cols, channels = 224, 224, 3
    else:
        rows, cols, channels = 160, 160, 3

    conv = VGGFace(include_top=img_net_format, input_shape=(rows, cols, channels))
    extractor = Model(conv.input, conv.get_layer(vgg_layer).output)
    if len(extractor.output_shape) > 2:
        _, frows, fcols, fchannels = extractor.output_shape
        feats = np.zeros((5000, 2, frows, fcols, fchannels), dtype=np.float32)
    else:
        _, fchannels = extractor.output_shape
        feats = np.zeros((5000, 2, fchannels), dtype=np.float32)

    with open(file_path, 'r') as splits_f:
        from tqdm import tqdm

        splits_f.readline()
        lines = splits_f.readlines()

    names = [None] * 2
    for i in tqdm(range(5000)):
        line = lines[i]
        # split number, pair number in split, first name, second name, is same:
        split_num, pair_num, names[0], names[1], same = line.strip().split(',')
        for pair_idx in range(2):
            for root, dirs, files in os.walk('{}/{}/'.format(ytf_root, names[pair_idx].strip())):
                num_files = len(files)
                if img_net_format:
                    frames = np.zeros((num_files, 224, 224, 3), dtype=np.float32)
                else:
                    frames = np.zeros((num_files, rows, cols, 3), dtype=np.float32)

                for file_it, file in enumerate(files):
                    frame = cv2.imread(os.path.join(root, file))
                    frame = ytf_rescale(frame, debug=False)
                    if img_net_format:
                        frame = cv2.resize(frame, (224, 224))
                    else:
                        frame = cv2.resize(frame, (rows, cols))
                    frame = frame[:, :, ::-1]
                    frames[file_it] = frame

            if center_data:
                # Zero-center by mean pixel
                frames[:, :, :, 2] -= 93.5940  # Blue
                frames[:, :, :, 1] -= 104.7624  # Green
                frames[:, :, :, 0] -= 129.1863  # Red

            if scale_data:
                frames /= 255
            frame_feats = extractor.predict(frames, batch_size=32)
            feats[i, pair_idx] = np.mean(frame_feats, 0)

    np.savez_compressed(file='ytf_vggf_larger_{}_{}'.format('_imgnet' if img_net_format else '', vgg_layer),
                        feats=feats)


def extract_lbp_features(path='e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\',
                         img_net_format=False, center_data=True,
                         scale_data=True):
    # from sklearn.datasets import fetch_lfw_pairs
    from ytf import fetch_ytf_pairs
    from skimage.feature import local_binary_pattern
    from keras_vggface.vggface import VGGFace
    from keras.models import Model
    import os
    import cv2

    ytf_root = path
    file_path = path + 'splits.txt'

    import keras.backend as K
    import tensorflow as tf
    K.clear_session()

    if img_net_format:
        rows, cols, channels = 224, 224, 3
    else:
        rows, cols, channels = 100, 100, 3

    n_bins = 256
    mode = 'default'
    feats = np.zeros((5000, 2, n_bins * 4), dtype=np.float32)

    with open(file_path, 'r') as splits_f:
        from tqdm import tqdm

        splits_f.readline()
        lines = splits_f.readlines()

    names = [None] * 2
    for i in tqdm(range(5000)):
        line = lines[i]
        # split number, pair number in split, first name, second name, is same:
        split_num, pair_num, names[0], names[1], same = line.strip().split(',')
        for pair_idx in range(2):
            for root, dirs, files in os.walk('{}/{}/'.format(ytf_root, names[pair_idx].strip())):
                num_files = len(files)
                if img_net_format:
                    frames = np.zeros((num_files, 224, 224, 3), dtype=np.float32)
                else:
                    frames = np.zeros((num_files, 100, 100, 3), dtype=np.float32)

                for file_it, file in enumerate(files):
                    frame = cv2.imread(os.path.join(root, file))
                    frame = ytf_rescale(frame, debug=False)
                    if img_net_format:
                        frame = cv2.resize(frame, (224, 224))
                    frame = frame[:, :, ::-1]
                    frames[file_it] = frame

            if center_data:
                # Zero-center by mean pixel
                frames[:, :, :, 2] -= 93.5940  # Blue
                frames[:, :, :, 1] -= 104.7624  # Green
                frames[:, :, :, 0] -= 129.1863  # Red

            if scale_data:
                frames /= 255

            frame_feats = np.zeros((frames.shape[0], n_bins * 4), dtype=np.float32)
            for j, img in enumerate(frames):
                sub_rows, sub_cols = rows // 2, cols // 2
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                lbp = local_binary_pattern(gray[0:sub_rows, 0:sub_cols], 8, 1, mode)
                lbp = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))[0]
                frame_feats[j, :n_bins] = lbp

                lbp = local_binary_pattern(gray[:sub_rows, sub_cols:], 8, 1, mode)
                lbp = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))[0]
                frame_feats[j, n_bins:n_bins * 2] = lbp

                lbp = local_binary_pattern(gray[sub_rows:, :sub_cols], 8, 1, mode)
                lbp = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))[0]
                frame_feats[j, n_bins * 2:n_bins * 3] = lbp

                lbp = local_binary_pattern(gray[sub_rows:, sub_cols:], 8, 1, mode)
                lbp = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))[0]
                frame_feats[j, n_bins * 3:] = lbp

            if np.isnan(feats).any():
                raise ValueError("Something went wrong NAN")

            feats[i, pair_idx] = np.mean(frame_feats, 0)

    np.savez_compressed(file='ytf_lbp.npz', feats=feats)


def extract_hsv_features(path='e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\',
                         img_net_format=False, center_data=True,
                         scale_data=True):
    # from sklearn.datasets import fetch_lfw_pairs
    from ytf import fetch_ytf_pairs
    from skimage.feature import local_binary_pattern
    from keras_vggface.vggface import VGGFace
    from keras.models import Model
    import os
    import cv2

    ytf_root = path
    file_path = path + 'splits.txt'

    import keras.backend as K
    import tensorflow as tf
    K.clear_session()

    if img_net_format:
        rows, cols, channels = 224, 224, 3
    else:
        rows, cols, channels = 100, 100, 3

    n_bins = 90
    feats = np.zeros((5000, 2, n_bins * 4), dtype=np.float32)

    with open(file_path, 'r') as splits_f:
        from tqdm import tqdm

        splits_f.readline()
        lines = splits_f.readlines()

    names = [None] * 2
    for i in tqdm(range(5000)):
        line = lines[i]
        # split number, pair number in split, first name, second name, is same:
        split_num, pair_num, names[0], names[1], same = line.strip().split(',')
        for pair_idx in range(2):
            for root, dirs, files in os.walk('{}/{}/'.format(ytf_root, names[pair_idx].strip())):
                num_files = len(files)
                if img_net_format:
                    frames = np.zeros((num_files, 224, 224, 3), dtype=np.float32)
                else:
                    frames = np.zeros((num_files, 100, 100, 3), dtype=np.float32)

                for file_it, file in enumerate(files):
                    frame = cv2.imread(os.path.join(root, file))
                    frame = ytf_rescale(frame, debug=False)
                    if img_net_format:
                        frame = cv2.resize(frame, (224, 224))
                    frame = frame[:, :, ::-1]
                    frames[file_it] = frame

            if center_data:
                # Zero-center by mean pixel
                frames[:, :, :, 2] -= 93.5940  # Blue
                frames[:, :, :, 1] -= 104.7624  # Green
                frames[:, :, :, 0] -= 129.1863  # Red

            if scale_data:
                frames /= 255

            frame_feats = np.zeros((frames.shape[0], n_bins * 4), dtype=np.float32)

            for j, img in enumerate(frames):
                sub_rows, sub_cols = rows // 2, cols // 2
                h_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                h_img = h_img[:, :, 0]

                hsv = np.histogram(h_img[0:sub_rows, 0:sub_cols], normed=True, bins=n_bins, range=(0, 360))[0]
                frame_feats[j, :n_bins] = hsv

                hsv = np.histogram(h_img[:sub_rows, sub_cols:], normed=True, bins=n_bins, range=(0, 360))[0]
                frame_feats[j, n_bins:n_bins * 2] = hsv

                hsv = np.histogram(h_img[sub_rows:, :sub_cols], normed=True, bins=n_bins, range=(0, 360))[0]
                frame_feats[j, n_bins * 2:n_bins * 3] = hsv

                hsv = np.histogram(h_img[sub_rows:, sub_cols:], normed=True, bins=n_bins, range=(0, 360))[0]
                frame_feats[j, n_bins * 3:] = hsv

            if np.isnan(feats).any():
                raise ValueError("Something went wrong NAN")

            feats[i, pair_idx] = np.mean(frame_feats, 0)

    np.savez_compressed(file='ytf_lbp.npz', feats=feats)

    # from sklearn.datasets import fetch_lfw_pairs
    import cv2

    # from sklearn.datasets import fetch_lfw_pairs
    from ytf import fetch_ytf_pairs

    X, y = fetch_ytf_pairs('e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\',
                           'e:\\Development\\Datasets\\YouTubeFaces\\splits.txt', img_net_format)

    if center_data:
        # Zero-center by mean pixel
        X[:, :, :, :, 2] -= 93.5940  # Blue
        X[:, :, :, :, 1] -= 104.7624  # Green
        X[:, :, :, :, 0] -= 129.1863  # Red

    if scale_data:
        X /= 255

    _, _, rows, cols, channels = X.shape

    n_bins = 90
    feats = np.zeros((X.shape[0], 2, n_bins * 4), dtype=np.float32)
    for i, pair in enumerate(X):
        for j, img in enumerate(pair):
            sub_rows, sub_cols = rows // 2, cols // 2
            h_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h_img = h_img[:, :, 0]

            hsv = np.histogram(h_img[0:sub_rows, 0:sub_cols], normed=True, bins=n_bins, range=(0, 360))[0]
            feats[i, j, :n_bins] = hsv

            hsv = np.histogram(h_img[:sub_rows, sub_cols:], normed=True, bins=n_bins, range=(0, 360))[0]
            feats[i, j, n_bins:n_bins * 2] = hsv

            hsv = np.histogram(h_img[sub_rows:, :sub_cols], normed=True, bins=n_bins, range=(0, 360))[0]
            feats[i, j, n_bins * 2:n_bins * 3] = hsv

            hsv = np.histogram(h_img[sub_rows:, sub_cols:], normed=True, bins=n_bins, range=(0, 360))[0]
            feats[i, j, n_bins * 3:] = hsv

    if np.isnan(feats).any():
        raise ValueError("Something went wrong NAN")
    np.savez_compressed(file='ytf_hsv_hist', feats=feats)


if __name__ == '__main__':
    from keras_vggface.vggface import VGGFace

    # VGGFace(False, input_shape=(192, 144, 3)).summary()

    extract_vgg_faces_features('e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\', img_net_format=True,
                               vgg_layer='fc6')
    # extract_vgg_faces_features('e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\', img_net_format=True,
    #                            vgg_layer='fc7')
    extract_vgg_faces_features('e:\\Development\\Datasets\\YouTubeFaces\\aligned_images_DB\\', img_net_format=False,
                               vgg_layer='pool5')
