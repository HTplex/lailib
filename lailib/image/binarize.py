import numpy as np
import cv2

def otsu_thresh(im):
    '''
    binarize image with otsu algorithm
    :param im(ndarray): uint8 numpy array
    :return: binarized image as uint8 numpy array
    '''
    _, binarized = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized

def vanilla_thresh(im):
    '''
    binarize image with a fixed threshold (170)
    :param im(ndarray): uint8 numpy array
    :return: binarized image as uint8 numpy array
    '''
    _, binarized = cv2.threshold(im, 0, 170, cv2.THRESH_BINARY)
    return binarized
