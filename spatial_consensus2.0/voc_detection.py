import numpy as np
from chainercv import utils

from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.datasets import voc_detection_label_names
from chainercv.datasets.voc import voc_detection_dataset
from chainer import iterators
from chainercv.evaluations import eval_detection_voc

def load_detection_windows(file_name):
    bboxes = []
    labels = []
    scores = []
    with open(file_name) as file:
        for line in file:
            det_win = line.strip()
            det_win = det_win.split(' ')
            x = int(float(det_win[0]))
            y = int(float(det_win[1]))
            w = int(float(det_win[2]))
            h = int(float(det_win[3]))

            bboxes.append(np.array([x, y, w, h]))
            labels.append(int(det_win[4]))
            scores.append(float(det_win[5]))

    bboxes = np.array(bboxes)
    labels = np.array(labels)
    scores = np.array(scores)
    return [bboxes], [labels], [scores]

if __name__ == '__main__':
    input = 'C:/Users/Artur Jordao/Desktop/Residuals/tete/SC'
    dataset = voc_detection_dataset.VOCDetectionDataset(
        year='2007', split='test', use_difficult=True, return_difficult=True)

    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    gt_bboxes = []
    gt_labels = []
    gt_difficults = []

    pred_bboxes = []
    pred_labels = []
    pred_scores = []

    img_idx = 0
    for batch in iterator:

        imgs = list()
        gt_values = list()
        for sample in batch:
            if isinstance(sample, tuple):
                imgs.append(sample[0])
                gt_values.append(sample[1:])
            else:
                imgs.append(sample)
                gt_values.append(tuple())

        gt_values = tuple(list(v) for v in zip(*gt_values))

        gt_bboxes += [tmp for tmp in gt_values[0]]
        gt_labels += [tmp for tmp in gt_values[1]]
        gt_difficults += [tmp for tmp in gt_values[2]]

        full_path = '{}/{}.txt'.format(input, img_idx)
        pred = load_detection_windows(file_name=full_path)
        img_idx = img_idx + 1

        pred_bboxes += [tmp for tmp in pred[0]]
        pred_labels += [tmp for tmp in pred[1]]
        pred_scores += [tmp for tmp in pred[2]]

    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, use_07_metric=True)
    print(result)