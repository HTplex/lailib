import cv2

def resize_height_keep_ratio(image_cv, new_height, **kwargs):
    '''
    resize image with given height, keep ratio, kwargs are passed to
    cv2.resize function.
    :param image_cv(ndarray): image to be resized
    :param new_height(int): new height of the output image
    :return: resized image
    '''

    (height, width) = image_cv.shape
    new_width = int(float(width) * new_height / float(height))
    image_cv = cv2.resize(image_cv, (new_width, new_height), **kwargs)
    return image_cv