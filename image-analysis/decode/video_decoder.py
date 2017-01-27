import os
import skvideo.io
import numpy as np
from skimage.io import *
from skimage.color import rgb2gray

def pad_batch(batch, batch_size, frame):
    """
    INPUTS
    frame:      a numpy array extracted from an MPEG (length x width x channels)
    batch:      list of frames
    batch_size: number of frames per batch (integer >= 1)

    OUTPUT
    returns padded batch: list of batch_size frames, each an ndarray:(L x W x C)

    DESCRIPTION: takes in a batch, pads it with 0s if necessary, and returns 
                    appended batch_list
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
    batch = []
#   build batch_list
    count = 0
    for frame in skvideo.io.vreader(v_path):
        if count >= start_idx:
            if stride > batch_size:
                if count == end_idx:
                    if len(batch) == batch_size:
                        batch_list.append(np.array(batch))
                        batch = []
                    batch.append(frame)
                    batch_list.append(np.array(pad_batch(batch, batch_size, frame)))
                    return batch_list[1:]
                elif count % stride  == 0:
                    temp = count
                    batch_list.append(np.array(batch))
                    batch = []
                    batch.append(frame)
                elif temp < count and count < (temp + batch_size):
                    batch.append(frame)
            elif batch_size >= stride:
                print("batch_size >= stride")
                if len(batch) < batch_size - stride:
                    batch.append(frame)
                elif len(batch) % batch_size == 0:
                    print("len(batch) % batch_size == 0")
                    batch_list.append(np.array(batch))
                    batch = []
                    for x in range(stride - batch_size, 0):
                        batch.append(batch_list[-1][x])
                elif len(batch) < batch_size:
                    batch.append(frame)
                else: print("problem")
                if count == end_idx:
                    if len(batch) == batch_size:
                        batch_list.append(np.array(batch))
                    elif len(batch) < batch_size:
                        pad_list = [[np.zeros(frame.shape)]] * \
                                (batch_size - len(batch))
                        batch += pad_list
                        batch_list.append(np.array(batch))
                    else: print("problem!!")
        count += 1
    return batch_list # list of numpy arrays


curr_dir = os.getcwd()
v_path = curr_dir + "/1.E.E.1.mp4"
a = decode_mpeg(v_path, batch_size=2, stride=4, start_idx = 0, end_idx = 20)

print(len(a))
print(a[-1].shape)
print(a[-1][-1,-1,-1])
