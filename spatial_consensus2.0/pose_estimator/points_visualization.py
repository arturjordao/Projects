import cv2
import numpy as np
import glob

def load_detection_windows(file_name):
    dts = []
    with open(file_name) as file:
        for line in file:
            det_win = line.strip()
            det_win = det_win.split(' ')
            x = int(det_win[0])
            y = int(det_win[1])
            score = float(det_win[3])
            part_idx = int(det_win[2])
            dts.append([x, y, score, part_idx])

    return dts

if __name__ == '__main__':
    points_path = 'C:/Users/ARTUR/Desktop/Residuals/tete/Pose'
    images_path = 'E:/datasets/pedestrian_detection/INRIAPerson/Test/pos'
    img_ext = '.png'

    images_names = glob.glob(points_path + '/*.Body.txt')

    for file_name in images_names:
        img_name = file_name.split('\\')
        img_name = img_name[len(img_name) - 1]
        img_name = img_name.replace('.Body.txt', img_ext)

        img = cv2.imread(images_path+'/'+img_name)
        points = load_detection_windows(file_name)
        for point in points:
            x, y, score, id = point
            img = cv2.circle(img, (x, y), 2, (255, 0, 0), 3)
        cv2.imshow('Draw01', img)
        cv2.waitKey(0)