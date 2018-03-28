import cv2
import numpy as np
import glob
from chainercv.datasets import camvid_label_colors
from chainercv.datasets import camvid_label_names
from chainercv.visualizations import vis_image
from chainercv import utils
import matplotlib.pyplot as plot


def _default_cmap(label):
    """Color map used in PASCAL VOC"""
    r, g, b = 0, 0, 0
    i = label
    for j in range(8):
        if i & (1 << 0):
            r |= 1 << (7 - j)
        if i & (1 << 1):
            g |= 1 << (7 - j)
        if i & (1 << 2):
            b |= 1 << (7 - j)
        i >>= 3
    return r, g, b


def vis_semantic_segmentation(
        label, label_names=None,
        label_colors=None, ignore_label_color=(0, 0, 0), alpha=1,
        all_label_names_in_legend=False, ax=None):
    """Visualize a semantic segmentation.

    Example:

        >>> from chainercv.datasets import VOCSemanticSegmentationDataset
        >>> from chainercv.datasets \
        ...     import voc_semantic_segmentation_label_colors
        >>> from chainercv.datasets \
        ...     import voc_semantic_segmentation_label_names
        >>> from chainercv.visualizations import vis_image
        >>> from chainercv.visualizations import vis_semantic_segmentation
        >>> import matplotlib.pyplot as plot
        >>> dataset = VOCSemanticSegmentationDataset()
        >>> img, label = dataset[60]
        >>> ax = vis_image(img)
        >>> _, legend_handles = vis_semantic_segmentation(
        ...     label,
        ...     label_names=voc_semantic_segmentation_label_names,
        ...     label_colors=voc_semantic_segmentation_label_colors,
        ...     alpha=0.9, ax=ax)
        >>> ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        >>> plot.show()

    Args:
        label (~numpy.ndarray): An integer array of shape
            :math:`(height, width)`.
            The values correspond to id for label names stored in
            :obj:`label_names`.
        label_names (iterable of strings): Name of labels ordered according
            to label ids.
        label_colors: (iterable of tuple): An iterable of colors for regular
            labels.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`.
            If :obj:`colors` is :obj:`None`, the default color map used.
        ignore_label_color (tuple): Color for ignored label.
            This is RGB format and the range of its values is :math:`[0, 255]`.
            The default value is :obj:`(0, 0, 0)`.
        alpha (float): The value which determines transparency of the figure.
            The range of this value is :math:`[0, 1]`. If this
            value is :obj:`0`, the figure will be completely transparent.
            The default value is :obj:`1`. This option is useful for
            overlaying the label on the source image.
        all_label_names_in_legend (bool): Determines whether to include
            all label names in a legend. If this is :obj:`False`,
            the legend does not contain the names of unused labels.
            An unused label is defined as a label that does not appear in
            :obj:`label`.
            The default value is :obj:`False`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        matploblib.axes.Axes and list of matplotlib.patches.Patch:
        Returns :obj:`ax` and :obj:`legend_handles`.
        :obj:`ax` is an :class:`matploblib.axes.Axes` with the plot.
        It can be used for further tweaking.
        :obj:`legend_handles` is a list of legends. It can be passed
        :func:`matploblib.pyplot.legend` to show a legend.

    """
    import matplotlib
    from matplotlib.patches import Patch
    from matplotlib import pyplot as plot

    if label_names is not None:
        n_class = len(label_names)
    elif label_colors is not None:
        n_class = len(label_colors)
    else:
        n_class = label.max() + 1

    if label_colors is not None and not len(label_colors) == n_class:
        raise ValueError(
            'The size of label_colors is not same as the number of classes')
    if label.max() >= n_class:
        raise ValueError('The values of label exceed the number of classes')

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    if label_colors is None:
        label_colors = [_default_cmap(l) for l in range(n_class)]
    # [0, 255] -> [0, 1]
    label_colors = np.array(label_colors) / 255
    cmap = matplotlib.colors.ListedColormap(label_colors)

    img = cmap(label / (n_class - 1), alpha=alpha)

    # [0, 255] -> [0, 1]
    ignore_label_color = np.array(ignore_label_color) / 255,
    img[label < 0, :3] = ignore_label_color

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.imshow(img)

    legend_handles = list()
    if all_label_names_in_legend:
        legend_labels = [l for l in np.unique(label) if l >= 0]
    else:
        legend_labels = range(n_class)
    for l in legend_labels:
        legend_handles.append(
            Patch(color=cmap(l / (n_class - 1)), label=label_names[l]))

    return ax, legend_handles

def load_segmentation(file_name, path):
    seg_img = []
    with open(path+'/'+file_name) as file:
        for line in file:
            seg_img.append(line.split(' '))

    seg_img = np.array(seg_img)
    return seg_img

if __name__ == '__main__':
    seg_path = 'C:/Users/ARTUR/Desktop/Residuals/tete/Segmentation'
    images_path = 'E:/datasets/pedestrian_detection/INRIAPerson/Test/pos'
    img_ext = '.png'
    color = [(255, 0, 0), (0, 255, 255)]

    images_names = glob.glob(seg_path + '/*.Body.txt')

    for file_name in images_names:
        img_name = file_name.split('\\')
        img_name = img_name[len(img_name) - 1]
        file_name = img_name
        img_name = img_name.replace('.Body.txt', img_ext)


        seg = load_segmentation(file_name, seg_path)
        img = utils.read_image(images_path+'/'+img_name, color=True)
        fig = plot.figure()
        #ax1 = fig.add_subplot(1, 2, 1)
        #vis_image(img, ax=ax1)
        ax2 = fig.add_subplot(1, 1, 1)

        tmp = seg.astype(int)
        vis_semantic_segmentation(tmp, camvid_label_names, camvid_label_colors, ax=ax2)
        plot.show()

        # img = np.zeros([seg.shape[0], seg.shape[1], 3])
        # for row in range(0,img.shape[0]):
        #     for col in range(0, img.shape[1]):
        #         label = int(seg[row][col])
        #         rgb = camvid_label_colors[label]
        #         img[row,col,0] = rgb[2]
        #         img[row, col, 1] = rgb[1]
        #         img[row, col, 2] = rgb[0]
        # cv2.imshow('Draw01', img)
        # cv2.waitKey(0)
