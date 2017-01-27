# REQUIRED PACKAGES
# sci-kit-image
# ffmpeg
# skvideo

import os
import skvideo.io
import numpy as np
from skimage.io import *
from skimage.color import rgb2gray


def decode_mpeg(v_path,*, batch_size=1, stride=1, start_idx=0, end_idx=-1,
        out_frame_ext=".jpg", out_frame_dir=""):
    # incomplete, ravi is comign back here later today to fix and add comments.
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
    if start_idx < 0 or end_idx < 0:
        raise ValueError("Cannot use negative start or end indices")
    elif end_idx < start_idx:
        raise ValueError("Cannot use end_idx < start_idx")
    elif batch_size < 1 or stride < 1:
        raise ValueError("Cannot use batch_size or stride < 1")
    
    batch_list = []
    count = 0
    temp = start_idx
    
#    initialize batch
#    if stride >= batch_size:
#        batch = []
#    elif batch_size > stride:
#        for frame in skvideo.io.reader(v_path):
#            if count - start_idx  == batch_size:
#                break
#            elif count >= start_idx:
#                batch.append(frame)
#            count += 1

# build batch_list
    count = 0
    for frame in skvideo.io.vreader(v_path):
        if count >= start_idx:
            if stride > batch_size:
                if count % stride  == 0:
                    temp = count
                    batch_list.append(np.array(batch))
                    batch = []
                    batch.append(frame)
                elif (temp < count and count < (temp + batch_size)):
                    batch.append(frame)
                else print("problem")
                if count == end_idx:
                    if len(batch) == batch_size:
                        batch_list.append(np.array(batch))
                    elif len(batch) < batch_size:
                        pad_list = [[np.zeros(frame.shape)]] * \
                                (batch_size - len(batch))
                        batch += pad_list
                        batch_list.append(np.array(batch))
                    else print("problem!!")
            elif batch_size >= stride:
                if len(batch) < batch_size - stride:
                    batch.append(frame)
                elif len(batch) % batch_size == 0:
                    batch_list.append(np.array(batch))
                    batch = []
                    for x in range(stride - batch_size, 0):
                        batch.append(batch_list[-1][x])
                elif len(batch) < batch_size:
                    batch.append(frame)
                if count == end_idx:
                    if len(batch) == batch_size:
                        batch_list.append(np.array(batch))
                    elif len(batch) < batch_size:
                        pad_list = [[np.zeros(frame.shape)]] * \
                                (batch_size - len(batch))
                        batch += pad_list
                        batch_list.append(np.array(batch))
                    else print("problem!!")
        count += 1
    return batch_list # list of numpy arrays
