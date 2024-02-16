import torch
import cv2
from albumentations import *
import numpy as np
from itertools import compress, product
from skimage.util.shape import view_as_windows
from typing import Tuple
import scipy.signal

import numpy as np




def norm(original):# normalize to [0,1]
    d_min=original.min()
    if d_min<0:
        original+=torch.abs(d_min)
        d_min=original.min()
    d_max=original.max()
    dst=d_max-d_min
    norm_data=(original-d_min).true_divide(dst)
    return norm_data



def faultseg_augumentation(p=1):
    return Compose([
    OneOf([
        HorizontalFlip(p=p),
        VerticalFlip(p=p),
        Compose([VerticalFlip(p=p), HorizontalFlip(p=p)]),
    ]),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30,interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5),

    ])


def crop2(variable, th, tw):  # this is for crop center when outputs are 96*96
    h, w = variable.shape[-2], variable.shape[-1]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]

cached_2d_windows = dict()



def window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

def spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

# to split big image to small size image
def split_Image(bigImage, isMask, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize, vertical_splits_number,
                horizontal_splits_number):
    #     print(bigImage.shape)
    if isMask == True:
        arr = np.pad(bigImage, ((top_pad, bottom_pad), (left_pad, right_pad)), "reflect")
        splits = view_as_windows(arr, (splitsize, splitsize), step=stepsize)#(66, 270, 58, 58)
        splits = splits.reshape((vertical_splits_number * horizontal_splits_number, splitsize, splitsize))
    else:
        arr = np.pad(bigImage, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), "reflect")
        splits = view_as_windows(arr, (splitsize, splitsize, 3), step=stepsize)
        splits = splits.reshape((vertical_splits_number * horizontal_splits_number, splitsize, splitsize, 3))
    return splits  # return list of arrays.

# to recover small patches to big image
# idea from https://github.com/dovahcrow/patchify.py
def recover_Image(patches: np.ndarray, imsize: Tuple[int, int, int], left_pad, right_pad, top_pad, bottom_pad,
                  overlapsize):
    #     patches = np.squeeze(patches)
    assert len(patches.shape) == 5

    i_h, i_w, i_chan = imsize
    image = np.zeros((i_h + top_pad + bottom_pad, i_w + left_pad + right_pad, i_chan), dtype=patches.dtype)
    divisor = np.zeros((i_h + top_pad + bottom_pad, i_w + left_pad + right_pad, i_chan), dtype=patches.dtype)

    n_h, n_w, p_h, p_w, _ = patches.shape

    o_w = overlapsize
    o_h = overlapsize

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i, j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    recover = image / divisor
    return recover[top_pad:top_pad + i_h, left_pad:left_pad + i_w]

