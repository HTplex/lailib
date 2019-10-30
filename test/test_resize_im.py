import numpy as np
import pytest
from lailib.image.resize_im import resize_height_keep_ratio

@pytest.mark.parametrize('original_height,original_width,goal_height,goal_width',
                        [(1, 1, 100, 100), (1, 1, 30, 30 ), (3, 5, 50, 83), (5, 3, 50, 30), (100, 250, 30, 75)])
def test_resize_height_keep_ratio(original_height, original_width, goal_height, goal_width):
    fake_im = np.zeros((original_height, original_width))
    ret_im = resize_height_keep_ratio(fake_im, goal_height)
    _, w = ret_im.shape
    assert(abs(goal_width - w) < 2)
