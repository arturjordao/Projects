import numpy as np
import sys
import os
from chainercv import utils

from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.datasets.voc import voc_detection_dataset
from chainer import iterators
from chainercv.evaluations import eval_detection_voc

def save_detection_windows(file_name, detections, labels=None, scores=None):
    file_out = open(file_name, 'w')
    line = ''
    if labels is not None:
        n_detections = detections[0].shape[0]
        for i in range(0, n_detections):
            detwin = detections[0][i]
            label = labels[0][i]

            if scores is not None:
                score = scores[0][i]
            else:
                score = 1.0

            line += '{} {} {} {} {} {}\n'.format(str(detwin[0]), str(detwin[1]), str(detwin[2]),
                                              str(detwin[3]), str(label), str(score))
    else:
        for i in range(0, len(detections)):
            detwin = detections[i][0:4]
            label = int(detections[i][4])
            score = detections[i][5]
            line += '{} {} {} {} {} {}\n'.format(str(detwin[0]), str(detwin[1]), str(detwin[2]),
                                                 str(detwin[3]), str(label), str(score))

    file_out.write(line)
    file_out.close()

def load_detection_windows(file_name):
    dts = []
    with open(file_name) as file:
        for line in file:
            det_win = line.strip()
            det_win = det_win.split(' ')
            x = int(float(det_win[0]))
            y = int(float(det_win[1]))
            w = int(float(det_win[2]))
            h = int(float(det_win[3]))
            label = int(det_win[4])
            score = float(det_win[5])
            dts.append(np.array([x, y, w, h, label, score]))

    return dts

def detect(detector='Faster', output=''):
    if detector == 'Faster':
        model = FasterRCNNVGG16(pretrained_model='voc07')
    elif detector == 'SSD300':
        model = SSD300(pretrained_model='voc0712')
    elif detector == 'SSD512':
        model = SSD512(pretrained_model='voc0712')

    model.score_thresh = 0.1

    dataset = voc_detection_dataset.VOCDetectionDataset(
        year='2007', split='test', use_difficult=True, return_difficult=True)

    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    img_idx =0
    for batch in iterator:

        imgs = list()
        for sample in batch:
            if isinstance(sample, tuple):
                imgs.append(sample[0])
            else:
                imgs.append(sample)


        pred = model.predict(imgs)
        pred_bboxes = pred[0]
        pred_labels = pred[1]
        pred_scores = pred[2]

        full_path = '{}/{}.txt'.format(output, img_idx)
        save_detection_windows(full_path, pred_bboxes, pred_labels, pred_scores)
        img_idx = img_idx +1

def save_gt(output=''):
    dataset = voc_detection_dataset.VOCDetectionDataset(
        year='2007', split='test', use_difficult=True, return_difficult=True)

    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

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

        gt_bboxes = gt_values[0]
        gt_labels = gt_values[1]

        full_path = '{}/{}.txt'.format(output, img_idx)
        save_detection_windows(full_path, gt_bboxes, gt_labels)
        img_idx = img_idx + 1

def union(a,b):
   x = min(a[0], b[0])
   y = min(a[1], b[1])
   w = max(a[0]+a[2], b[0]+b[2]) - x
   h = max(a[1]+a[3], b[1]+b[3]) - y
   return (x, y, w, h)

def intersection(a,b):
   x = max(a[0], b[0])
   y = max(a[1], b[1])
   w = min(a[0]+a[2], b[0]+b[2]) - x
   h = min(a[1]+a[3], b[1]+b[3]) - y
   if (w<0 or h<0): return (0, 0, 0, 0)
   return (x, y, w, h)

def area(r):
    return r[2]*r[3]

def jaccard(a, b):
    U = union(a, b)
    I = intersection(a, b)
    return area(I)/float(area(U))
######################################################
def fx(x,a,b):
    return a*x + b

def sc(dts=[], projection=None, jaccard_th=0.6):
#Inputs:
# dts - List of bouding-boxes from detectors
#projections - Parameters, a and b, of the projection to perform y = ax+b

        U = []
        #Map the det_i response to det0 space
        if projection is not None:

            for i in range(1, len(dts)):
                for idxDetWin in range(0, len(dts[i])):
                    score = dts[i][idxDetWin][4]
                    score = fx(score, projection[i][0], projection[i][1])
                    dts[i][idxDetWin][4] = score

        for h in range(0, len(dts[0])):

            ph = dts[0][h]
            matched = False
            for i in range(1,len(dts)):
                for l in range(0, len(dts[i])):

                    pl = dts[i][l]
                    if ph[4] == pl[4]:
                        whl = jaccard(ph, pl)
                        if whl > jaccard_th:
                            matched = True
                            sh = ph[5] + whl * pl[5]
                            dts[0][h][5] = sh
                            ph = dts[0][h]
            if matched:
                U.append(ph)

        return U

def execute_folder(detection_folders='',output_dir=''):

    for i in range(0, 4952):
        dts = []

        for detector_path in detection_folders:
            full_path = '{}/{}.txt'.format(detector_path, i)
            dts.append(load_detection_windows(full_path))

        output = sc(dts)
        full_path = '{}/{}.txt'.format(output_dir, i)
        save_detection_windows(full_path, output)

if __name__ == '__main__':
    #save_gt(output='C:/Users/Artur Jordao/Desktop/Residuals/tete')
    #detect(detector='Faster', output='Faster')
    #detect(detector='SSD300', output='C:/Users/Artur Jordao/Desktop/Residuals/tete/SSD30')
    #detect(detector='SSD512', output='SSD512')

    detector1='C:/Users/Artur Jordao/Desktop/Residuals/tete/Faster'
    detector2='C:/Users/Artur Jordao/Desktop/Residuals/tete/SSD300'
    detector3='C:/Users/Artur Jordao/Desktop/Residuals/tete/SSD512'
    detectors = [detector1, detector2, detector3]

    output_dir = 'C:/Users/Artur Jordao/Desktop/Residuals/tete/SC'
    execute_folder(detection_folders=detectors,
                   output_dir=output_dir)

    #pred_bboxes = pred[0]
    #pred_labels = pred[1]
    #pred_scores = pred[2]

    #gt_bboxes = gt[0]
    #gt_labels = gt[1]
    #result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, use_07_metric=True)
    #print(result)