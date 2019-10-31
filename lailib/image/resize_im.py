import cv2

def resize_height_keep_ratio(im, new_height, **kwargs):
    '''
    resize image with given height, keep ratio, kwargs are passed to
    cv2.resize function.
    :param image_cv(ndarray): image to be resized
    :param new_height(int): new height of the output image
    :return: resized image
    '''

    (height, width) = im.shape
    new_width = int(float(width) * new_height / float(height))
    im = cv2.resize(im, (new_width, new_height), **kwargs)
    return im
