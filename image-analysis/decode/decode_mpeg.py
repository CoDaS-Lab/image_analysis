# REQUIRED PACKAGES
# sci-kit-image
# ffmpeg
# skvideo

import os
import skvideo.io
import skvideo.datasets
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
    start_idx   index of first frame for first batch (integer >= 0)
    end_idx     index of last frame in the range of interest (integer >= 0)
    out_frame_ext: extension you want to save the frames with (e.g. jpg)
    out_frame_dir: directory you want to save extracted frames to

    OUTPUTS
    RETURNS a LIST of NUMPY batches of frames
    #TODO create optional saving, incase someone wants to visualize the frames

    DESCRIPTION: extracts frames from MPEG and saves them to a directory
    """
    if start_idx < 0 or end_idx < 0:
        raise ValueError("Cannot use negative start or end indices")
    elif batch_size < 1 or stride < 1:
        raise ValueError("Cannot use batch_size or stride < 1")
    
    batch_list = []
    batch = []
    count = 0
    temp = start_idx
    
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
                elif count == end_idx:
                    if len(batch) == batch_size:
                        batch_list.append(batch)
                    elif len(batch) < batch_size:
                        pad_list = [[np.zeros(frame.shape)]] * \
                                (batch_size - len(batch))
                        batch += pad_list
                    else print("problem!!")
            elif batch_size >= stride:

            return batch_list.append(np.array(batch))
            
            if len(batch) % batch_size == 0:
                batch_list.append(np.array(batch))
                batch = []
                count -= (batch_size - stride -1)
                if (batch_size - stride -1 > 0 )
                    batch.append(batch_list[-1][-(batch_size-stride - 1):])
            batch.append(frame)
        if count2 == end_idx:
            if batch_size - len(batch) > 0:
                pad_list = [[np.zeros(frame.shape)]] * (batch_size - len(batch))
                batch += pad_list
            break
        count += 1
        count2 += 1
    
    return batch_list # list of numpy arrays


#def extract_frames_from_MPEG(v_path,*, out_frame_dir, out_frame_ext=".jpg",\
#        out_data_dir="", start_idx=0, end_idx=-1):
#    """
#    INPUTS
#    v_path: path to MPEG video (i.e. include the video's name & extension)
#    out_frame_ext: extension you want to save the frames with (e.g. jpg)
#    out_frame_dir: directory you want to save extracted frames to
#    stride_length: extract frame on every stride_length'd frame
#    start_idx: index of first frame you want to save (integer)
#    end_idx: index of last frame you want to save (integer)
#
#    OUTPUTS
#    RETURNS batches of frames; list of numpy arrays
#    SAVES frames if you want
#
#    DESCRIPTION: extracts frames from MPEG and saves them to a directory
#    """
#    v_name = os.path.basename(v_path)
#    video = skvideo.io.vreader(v_path)
#    count = 0
#    
#    if out_frame_dir == out_data_dir == "":
#        raise ValueError('No output directory was specified.')
#    elif out_frame_dir != "":
#        for frame in video:
#            count += 1
#            if count > start_idx:
#                imsave(out_frame_dir + "/" + v_name + "_frame" + \
#                        str(count) + out_frame_ext, frame)
#            if count == end_idx:
#                break
#
#curr_dir = os.getcwd()
#path_to_MPEG_video = curr_dir + "/1.E.E.1.mp4"
#extract_frames_from_MPEG(path_to_MPEG_video, curr_dir,end_idx=5)
