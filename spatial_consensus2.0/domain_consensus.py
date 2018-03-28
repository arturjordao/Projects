import numpy as np
import cv2
import copy

def load_detection(file_name, path):
    dts = []
    with open(path + '/' + file_name) as file:
        for line in file:
            det_win = line.strip()
            det_win = det_win.split(' ')
            x = int(det_win[0])
            y = int(det_win[1])
            w = int(det_win[2])
            h = int(det_win[3])
            score = float(det_win[4])
            dts.append([x, y, w, h, score])

    return dts

def load_pose(file_name, path):
    dts = []
    with open(path+'/'+file_name) as file:
        for line in file:
            det_win = line.strip()
            det_win = det_win.split(' ')
            x = int(det_win[0])
            y = int(det_win[1])
            score = float(det_win[3])
            part_idx = int(det_win[2])
            dts.append([x, y, score, part_idx])

    return dts

def load_segmentation(file_name, path):
    seg_img = []
    with open(path+'/'+file_name) as file:
        for line in file:
            seg_img.append(line.split(' '))

    seg_img = np.array(seg_img)
    return seg_img

def save_detection_windows(file_name, detections):
    file_out = open(file_name, 'w')
    line = ''
    for detwin in detections:
        line+= '{} {} {} {} {}\n'.format(str(detwin[0]), str(detwin[1]), str(detwin[2]), str(detwin[3]), str(detwin[4]))
    file_out.write(line)
    file_out.close()

def points_inside_bb(bb, points):
    energy = []
    x_bb, y_bb, w_bb, h_bb = bb

    # img_ori = np.zeros((512, 512, 3), np.uint8)
    # cv2.rectangle(img_ori, (x_bb, (y_bb+h_bb)), ((x_bb+w_bb), y_bb), (0, 255, 0), 3)

    for point in points:
        x_p, y_p, score_p, _ = point
        if x_p >= x_bb and x_p <= (x_bb+w_bb):
            if y_p>= y_bb and y_p <= (y_bb+h_bb):
                energy.append(score_p)

                # img = copy.deepcopy(img_ori)
                # img = cv2.circle(img, (x_p, y_p), 2, (255, 0, 0), 3)
                # cv2.imshow('Draw01', img)
                # cv2.waitKey(0)
        # img = img_ori
    return np.sum(energy)/18.0
    if len(energy)==0:
        return 0
    return np.mean(energy)

def segm_inside_bb(bb, seg, type='9'):
    human_seg = 0
    seg_class = []
    x_bb, y_bb, w_bb, h_bb = bb
    crop = seg[y_bb:(y_bb+h_bb), x_bb:(x_bb+w_bb)]
    for row in range(0, h_bb):
        for col in range(0, w_bb):
            if crop[row][col] == type:
                human_seg+=1

    energy = human_seg/(crop.shape[0]*crop.shape[1])
    return energy

def dc(dt=None, seg=None, pose=None):
    U = []
    for bb in dt:
        x, y, w, h, score = bb
        energy_pose = points_inside_bb([x, y, w, h], pose)

        if seg is not None:
            energy_segmentation = segm_inside_bb([x, y, w, h], seg)

        score = score + energy_pose + energy_segmentation
        U.append([x, y, w, h, score])

    return U

def execute_folder(imgs_path='', detection_folder='',segmentation_folder='',pose_folder='', output_dir='', img_ext='.png'):
    import glob
    images_names = glob.glob(imgs_path+'/*{}'.format(img_ext))

    for img_name in images_names:
        dts = []
        file_name = img_name.split('\\')
        file_name = file_name[len(file_name)-1]
        file_name = file_name.replace(img_ext, '.Body.txt')

        dt = load_detection(file_name, detection_folder)

        if(segmentation_folder!=''):
            seg = load_segmentation(file_name, segmentation_folder)
        else:
            seg = None

        if(pose_folder!=''):
            pose = load_pose(file_name, pose_folder)
        else:
            pose = None

        detections = dc(dt, seg=seg, pose=pose)
        save_detection_windows(output_dir+'/'+file_name, detections)

if __name__ == '__main__':
    imgs_path = 'E:/datasets/pedestrian_detection/INRIAPerson/Test/pos'
    output_dir = ''
    img_ext='.png'
    #
    detector=''
    segmentation = ''
    pose = ''
    execute_folder(imgs_path=imgs_path,
                   detection_folder=detector,
                   segmentation_folder=segmentation,
                   pose_folder=pose,
                   output_dir=output_dir,
                   img_ext=img_ext)