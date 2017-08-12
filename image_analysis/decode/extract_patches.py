# Copyright 2017 Codas Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import skvideo.io
import os


def get_mpeg_dims(vid_fname):
    """
    ARGS:
        :vid_fname: file name for an mpeg video with one color channel

    RETURNS:
        :(tuple): containing number of rows, columns, and frames of data
    """
    meta_data = skvideo.io.ffprobe(vid_fname)
    height = meta_data['video']['@height']
    width = meta_data['video']['@width']
    n_frames = meta_data['video']['@nb_frames']

    return (int(height), int(width), int(n_frames))


def extract_patches(mpeg_path, patch_dims, save_dir, *, n_frames=None,
                    batch_size=250000):
    """
    DESCRIPTION:
        extract patches (neighborhoods) of pixels from videos in batches and
        save them to disk

    ARGS:
        :save_dir: directory to save extracted patches
        :mpeg_path: file name for an mpeg video with one color channel
        :patch_dims: size of patches to extract
        :n_frames: number of frames to extract
        :batch_size: size limit for each batch
    """
    N, M, O = get_mpeg_dims(mpeg_path)
    n, m, o = patch_dims
    ppr = (M - m + 1)
    ppc = (N - n + 1)
    ppf = (ppr)(N - n + 1)

    vid_gen = skvideo.io.vreader(mpeg_path)

    if n_frames is None:
        end = O - o + 1
    elif n_frames > (O - o + 1) and n_frames < o:
        raise ValueError('End frame index is out of range.')
    else:
        end = n_frames

    count = 0
    frames = []
    batches = []
    for frame in vid_gen:
        if count < end and count < o:
            frames.append(frame)
        elif count < end:
            arr = np.array(frames)
            for i in range(ppr):
                for j in range(ppc):
                    batches.append(arr[:, i: i + n, j: j + m].flatten())
            del frames[0]
            frames.append(frame)
            batches = np.array(batches)
            np.save(batches, save_dir + 'batch_' + str(count))
            batches = []
        else:
            break
        count += 1

# def main():
#     vid_path = os.getcwd() + '/../data/input/test_video.mp4'
#     # '/../data/input/subj5-4hr.avi'
#     patch_dims = (3, 3, 2)
#     n_frames = 100    # 104322
#     save_dir = os.getcwd() + '/../data/working/motion_patches/'
#     patches = extract_patches(vid_path, patch_dims,
#                               save_dir, n_frames=n_frames)


# if __name__ == '__main__':
#     main()
