import numpy as np

#####Rect functions####################################
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

def jaccard (a, b):
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
                    whl = jaccard(ph, pl)
                    if whl > jaccard_th:
                        matched = True
                        sh = ph[4] + whl * pl[4]
                        dts[0][h][4] = sh
                        ph = dts[0][h]
            if matched:
                U.append(ph)

        return U

def load_detection_windows(file_name, detectionPath):
    dts = []
    with open(detectionPath + '/' + file_name) as file:
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

def save_detection_windows(file_name, detections):
    file_out = open(file_name, 'w')
    line = ''
    for detwin in detections:
        line+= '{} {} {} {} {}\n'.format(str(detwin[0]), str(detwin[1]), str(detwin[2]), str(detwin[3]), str(detwin[4]))
    file_out.write(line)
    file_out.close()

def execute_folder(img_path='', detection_folders='',output_dir='', img_ext='.png', dt_ext='.Body.txt'):
    import glob
    images_names = glob.glob(img_path+'/*{}'.format(img_ext))

    for img_name in images_names:
        dts = []
        img_name = img_name.split(img_ext)[0] + dt_ext
        img_name = img_name.split('\\')
        img_name = img_name[len(img_name) - 1]

        for detector_path in detection_folders:
            dts.append(load_detection_windows(img_name, detector_path))

        output = sc(dts)
        save_detection_windows(img_name, output)

if __name__ == '__main__':
    detector1 = [[123, 44, 220, 592, 0.98]]
    detector2 = [[123, 54, 221, 585, 0.99]]
    detector3 = [[129, 70, 194, 583, 0.9987], [226, 72, 109, 316, 0.2823] ]
    output = sc(dts=[detector1, detector2, detector3])

    #
    detector1='C:/Users/Artur Jordao/Desktop/Residuals/Faster'
    detector2='C:/Users/Artur Jordao/Desktop/Residuals/SSD300'
    detector3='C:/Users/Artur Jordao/Desktop/Residuals/SSD550'
    detectors = [detector1, detector2, detector3]
    img_path = 'D:/datasets/pedestrian_detection/INRIAPerson/Test/pos'
    outpu_dir = 'C:/Users/Artur Jordao/Desktop/Residuals/SC'
    execute_folder(img_path=img_path, detection_folders=detectors,
                   output_dir=outpu_dir)