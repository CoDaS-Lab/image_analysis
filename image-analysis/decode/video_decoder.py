import os
import skvideo.io
import numpy as np
from skimage.io import *


def pad_batch(batch, batch_size, frame):
    """
    INPUTS
    frame:      a numpy array extracted from an MPEG (length x width x channels)
    batch:      list of frames
    batch_size: number of frames per batch (integer >= 1)

    OUTPUT
    returns padded batch: list of batch_size frames, each an ndarray:(L x W x C)

    DESCRIPTION: takes in a batch, pads it with 0s if necessary, and returns
                    appended batch
    """
    if len(batch) > batch_size:
        raise ValueError("len(batch) should be <= batch_size!")
    elif batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    elif len(batch) == batch_size:
        return batch
    elif len(batch) < batch_size:
        pad_list = [np.zeros(frame.shape)] * (batch_size - len(batch))
        batch += pad_list
        return batch
    else:
        raise ValueError("Something is wrong with the pad_batch function")


def decode_mpeg(v_path, *, batch_size=1, stride=1, start_idx=0, end_idx=-1, out_frame_ext=".jpg", out_frame_dir=""):
    """
    INPUTS
    v_path:     Path to MPEG video (i.e. include the video's name & extension)
    batch_size: Number of frames in each batch
    stride:     Stride indicates beginning of batches, i.e. every stride'th
                frame (integer > 1)
    start_idx:  Index of first frame for first batch (integer >= 0)
    end_idx:    Index of last frame in the range of interest (integer >= 0)
    out_frame_ext: Extension you want to save the frames with (e.g. jpg)
    out_frame_dir: Directory you want to save extracted frames to

    OUTPUTS
    RETURNS: LIST of NUMPY batches of frames (length x width x channels), and
                if the last batch is not full, it is padded with frames of:
                    np.zeros((frame.shape))
    #TODO create optional saving, incase someone wants to visualize the frames

    Description: creates a list of batches of frames from an MPEG file
    """

    if start_idx < 0 or end_idx < -1:
        raise ValueError("Cannot use start_idx < 0 or end_idx < -1")
    elif end_idx < start_idx:
        raise ValueError("Cannot use end_idx < start_idx")

    if batch_size < 1 or stride < 1:
        raise ValueError("Cannot use batch_size or stride < 1")

    batch_list = []
    count = 0
    temp = start_idx
    batch = []

    # build batch_list
    for frame in skvideo.io.vreader(v_path):
        if count == end_idx:
            if len(batch) == batch_size:
                batch_list.append(np.array(batch))
                batch = []
            idx = count - ((count - start_idx) % stride)
            if count >= idx and count < (idx + batch_size):
                batch.append(frame)
                batch_list.append(np.array(
                    pad_batch(batch, batch_size, frame)))
            if stride > batch_size:
                return batch_list[1:]
            elif batch_size >= stride:
                return batch_list
        elif count >= start_idx and stride > batch_size:
            if (count - start_idx) % stride == 0:
                temp = count
                batch_list.append(np.array(batch))
                batch = []
                batch.append(frame)
            elif temp < count and count < (temp + batch_size):
                batch.append(frame)
        elif count >= start_idx and batch_size >= stride:
            if count - start_idx < batch_size:
                batch.append(frame)
            elif len(batch) == batch_size:
                batch_list.append(np.array(batch))
                batch = []
                for x in range(stride - batch_size, 0):
                    batch.append(batch_list[-1][x])
                batch.append(frame)
            elif len(batch) < batch_size:
                batch.append(frame)
        count += 1
    if end_idx == -1:
        if len(batch) == batch_size:
            batch_list.append(np.array(batch))
            batch = []
        idx = count - ((count - start_idx) % stride)
        if count >= idx and count < (idx + batch_size):
            batch.append(frame)
            batch_list.append(np.array(
                pad_batch(batch, batch_size, frame)))
        if stride > batch_size:
            return batch_list[1:]
        elif batch_size >= stride:
            return batch_list
