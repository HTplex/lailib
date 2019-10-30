import pytest
import numpy as np
from lailib.image.crop import crop_boundary_and_padding

class TestCropBoundaryAndPad:
    @staticmethod
    def binarized_case0():
        res = np.zeros((3, 3), dtype=np.uint8)
        res[0, 0] = 255
        res[2, 2] = 255
        padded_in = np.pad(res, ((3, 2), (2, 3)), constant_values=(0, 0))
        return padded_in, res

    @staticmethod
    def binarized_case1():
        res = np.zeros((4, 5), dtype=np.uint8)
        res[0, 0] = 255
        res[3, 4] = 255
        res[2, 1] = 255
        padded_in = np.pad(res, ((23, 2), (2, 44)), constant_values=(0, 0))
        return padded_in, res

    @staticmethod
    def gray_case0():
        '''not binarized image without mask generated outside the crop function'''
        res = np.zeros((4, 5), dtype=np.uint8)
        res[0, 0] = 128
        res[3, 4] = 255
        res[2, 1] = 128
        padded_in = np.pad(res, ((23, 2), (2, 44)), constant_values=(0, 0))
        return padded_in, res

    @staticmethod
    def gray_case1():
        '''gray image with mask generated outside the crop function'''
        res = np.zeros((4, 5), dtype=np.uint8)
        res[0, 0] = 128
        res[3, 4] = 255
        res[2, 1] = 64

        binarized = np.copy(res)
        binarized[0, 0] = 255
        binarized[3, 4] = 255
        binarized[2, 1] = 255
        padded_in = np.pad(res, ((23, 2), (2, 44)), constant_values=(0, 0))
        padded_binarized = np.pad(binarized, ((23, 2), (2, 44)), constant_values=(0, 0))
        return padded_in, padded_binarized, res

    def test_out_type(self):
        x = np.ones((3, 3), dtype=np.uint8)
        out = crop_boundary_and_padding(x)
        assert out.dtype == np.uint8

    def test_binarized(self):
        padded_in0, res_0 = self.binarized_case0()
        padded_in1, res_1 = self.binarized_case1()
        assert(np.array_equal(crop_boundary_and_padding(padded_in0), res_0))
        assert(np.array_equal(crop_boundary_and_padding(padded_in1), res_1))

    def test_gray(self):
        padded_in0, res_0 = self.gray_case0()
        padded_in1, padded_binarized, res_1 = self.gray_case1()
        assert (np.array_equal(crop_boundary_and_padding(padded_in0), res_0)) #without binarize input
        assert (np.array_equal(crop_boundary_and_padding(padded_in1, binarized = padded_binarized), res_1)) #with binarize map

    def test_with_pad(self):
        padded_in_0 = np.ones((3, 3), dtype = np.uint8)
        res_0 = np.zeros((10,20), dtype = np.uint8)
        res_0[2:5, 10:13] = padded_in_0

        padded_in_1 = np.ones((3, 3), dtype = np.uint8)
        res_1 = np.zeros((9,9), dtype = np.uint8)
        res_1[3:6, 3:6] = padded_in_1

        assert(np.array_equal(crop_boundary_and_padding(padded_in_0, [10,7,2,5]), res_0))
        assert(np.array_equal(crop_boundary_and_padding(padded_in_1, 3), res_1))

    def test_all_zero_im(self):
        in_im = np.zeros((100, 200), dtype = np.uint8) #All zero input image with whatever shape
        with pytest.raises(ValueError, match='In image for crop function is all zero'):
            crop_boundary_and_padding(in_im)

    def test_invalid_pad(self):
        wrong_pad0 = [0,0,0,0,0]
        wrong_pad1 = 15432514.315345
        wrong_pad2 = [0]
        in_im = np.zeros((100, 200), dtype = np.uint8)
        in_im[30, 30] = 255
        with pytest.raises(TypeError, match='padding in crop function must be int or list of size 4'):
            crop_boundary_and_padding(in_im, padding = wrong_pad0)
        with pytest.raises(TypeError, match='padding in crop function must be int or list of size 4'):
            crop_boundary_and_padding(in_im, padding = wrong_pad1)
        with pytest.raises(TypeError, match='padding in crop function must be int or list of size 4'):
            crop_boundary_and_padding(in_im, padding = wrong_pad2)

    def test_wrong_input_type(self):
        wrong_dtype = np.ones((3, 3))*255 #image with wrong dtype(int64)
        not_gray_scale = np.ones((5, 5, 3), dtype = np.uint8) * 255 #image with 3 channels
        with pytest.raises(TypeError, match='input image must be gray scale image as uint8 ndarray'):
            crop_boundary_and_padding(wrong_dtype)
        with pytest.raises(TypeError, match='input image must be gray scale image as uint8 ndarray'):
            crop_boundary_and_padding(not_gray_scale)

    def test_wrong_binarized_input(self):
        in_im = np.ones((3,3), dtype=np.uint8)
        big_mask = np.ones((4,4), dtype=np.uint8) * 255
        small_mask = np.ones((2,2), dtype=np.uint8) * 255
        wrong_type = np.ones((3,3)) * 255

        with pytest.raises(ValueError, match='binarized image shape .* must meet in image shape .*'):
            crop_boundary_and_padding(in_im, binarized=big_mask)
        with pytest.raises(ValueError, match='binarized image shape .* must meet in image shape .*'):
            crop_boundary_and_padding(in_im, binarized=small_mask)
        with pytest.raises(TypeError, match='binarized mask must be uint8 ndarray'):
            crop_boundary_and_padding(in_im, binarized=wrong_type)
