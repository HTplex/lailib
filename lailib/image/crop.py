import numpy as np
from lailib.image.binarize import otsu_thresh

def crop_boundary_and_padding(im, padding=0, binarized=None):
    '''
    crop and padding objects and text imgs with pure black border, then padding.
    ex:
    for input
        0   255 0   0
        0   255 255 0
        0   255 0   0
        0   0   0   0
    output is
        255 0
        255 255
        255 0
    :param image_cv(ndarray): input image, uint8 numpy array, assumed to be gray scale
    :param padding(int or list of ints): if padding is a scalar, padding will be applied to left, right, upside and downside of an image.
                    if padding is a list of size 4, each entry in the list corresponds to left, right, upside and
                    downside of an image.
    :return: cropped image (uint8)
    '''
    if len(im.shape) != 2 or im.dtype != np.uint8:
        raise TypeError('input image must be gray scale image as uint8 ndarray')
    if not binarized is None:
        if binarized.shape != im.shape:
            raise ValueError('binarized image shape {} must meet in image shape {}'.format(binarized.shape, im.shape))
        if binarized.dtype != np.uint8:
            raise TypeError('binarized mask must be uint8 ndarray')
    else:
        binarized = otsu_thresh(im)
    if np.sum(binarized) == 0:
        raise ValueError('In image for crop function is all zero')

    if isinstance(padding, int):
        padding = [padding] * 4
    elif not (isinstance(padding, list) and len(padding) == 4):
        raise TypeError('padding in crop function must be int or list of size 4')

    # use binarized image to get vertical and horizontal boundaries
    col_sums = np.sum(binarized, axis=1)
    row_start = np.where(col_sums)[0][0]
    rev_col_sum = np.flip(col_sums, axis=0)
    rev_end = np.where(rev_col_sum)[0][0]
    row_end = col_sums.shape[0] - rev_end

    row_sums = np.sum(binarized, axis=0)
    col_start = np.where(row_sums)[0][0]
    rev_row_sum = np.flip(row_sums, axis=0)
    rev_end = np.where(rev_row_sum)[0][0]
    col_end = row_sums.shape[0] - rev_end

    cropped = im[row_start:row_end, col_start:col_end]

    cropped_height, cropped_width = cropped.shape
    # TODO add rand noise to height
    final_out = np.zeros((cropped_height + padding[2] + padding[3],
                          cropped_width + padding[0] + padding[1]), dtype = np.uint8)

    out_height, out_width = final_out.shape
    # do not directly use -padding as the value for end side because it can't deal with padding = 0
    final_out[padding[2]: out_height - padding[3], padding[0]: out_width - padding[1]] = cropped
    return final_out
