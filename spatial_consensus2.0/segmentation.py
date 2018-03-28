import numpy as np
from chainercv import utils
from chainercv.datasets import camvid_label_names
from chainercv.links import SegNetBasic
import glob

def segmentation(img):
    model = SegNetBasic(n_class=len(camvid_label_names),pretrained_model='camvid')
    labels = model.predict([img])[0]
    return labels

if __name__ == '__main__':
    images_path = 'E:/datasets/pedestrian_detection/INRIAPerson/Test/pos'
    output_dir = ''
    img_ext = '.png'
    images_names = glob.glob(images_path + '/*{}'.format(img_ext))

    for img_name in images_names:
        img = utils.read_image(img_name, color=True)
        file_name = img_name.split('\\')
        file_name = file_name[len(file_name)-1]
        file_name = file_name.replace(img_ext, '.Body.txt')

        seg = segmentation(img)
        np.savetxt(X=seg, fmt='%s', fname=output_dir+'/'+file_name, delimiter=' ', )
        #print('tete')