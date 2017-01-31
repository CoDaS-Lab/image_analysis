
from pathlib import Path
from decode import video_decoder as vd

import os
import shutil
import unittest
import skvideo
import skimage
import math
import warnings


class TestVideoDecoder(unittest.TestCase):
    def setUp(self):
        self.video_path = os.getcwd() + \
                "/test/decode/testing_files/IASample.mpeg"
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3

    def test_mpeg_dimensions(self):
        video = vd.decode_mpeg(self.video_path, end_idx=0)
        batch = video[0]  # grab a batch
        frame = batch[0]

        self.assertEqual(batch.shape[0], 1)  # only one frame in batch
        self.assertEqual(frame.shape[0], self.frames_height)
        self.assertEqual(frame.shape[1], self.frames_width)
        self.assertEqual(frame.shape[2], self.frames_channels)

    def test_decode_mpeg(self):
        # Ignore weird resource warnings for now
        warnings.simplefilter("ignore")

        # Load video file in batches = strides of 29 frames
        batch_size = stride = 29
        video_batches = vd.decode_mpeg(self.video_path,
                                       batch_size=batch_size, stride=stride)
        self.assertEqual(len(video_batches), 43,
                         'len(video_batches) was not 43')
        self.assertEqual(video_batches[0].shape[0], 29,
                         'batch did not contain 29 frames')
        self.assertEqual(video_batches[0][-1, -1, -1, -1], 231,
                         'last element of last batch is not 231')

        # load video with batch_size > stride with padding
        batch_size = 20
        stride = 10
        video_batches = vd.decode_mpeg(self.video_path, batch_size=batch_size,
                                       stride=stride, end_idx=1200)
        self.assertEqual(len(video_batches), 120,
                         'length of video_batches is not 120')
        self.assertEqual(video_batches[-1][-1, -1, -1, -1], 0,
                         'last element in last batch was not 0. not padded 
                         properly')

        # load video with batch_size > stride without padding
        batch_size = 20
        stride = 10
        video_batches = vd.decode_mpeg(self.video_path, 
                                       batch_size=batch_size, stride=stride, 
                                       end_idx=1119)
        self.assertEqual(len(video_batches), 111,
                         'length of batch_list is not 111')
        self.assertEqual(video_batches[-1][-1, -1, -1, -1], 238,
                         'last element was not 238')

        # load video with stride > batch_size with padding
        batch_size = 10
        stride = 20
        video_batches = vd.decode_mpeg(self.video_path, batch_size=batch_size,
                                       stride=stride, end_idx=-1)
        self.assertEqual(len(video_batches), 63,
                         'total number of batches was not equal to 63')
        self.assertEqual(video_batches[-1][-1, -1, -1, -1], 0,
                         'last element of last batch was not 0; 
                          not padded properly')

        # load video with stride > batch_size without padding
        batch_size = 10
        stride = 29
        video_batches = vd.decode_mpeg(self.video_path,
                                       batch_size=batch_size,
                                       stride=stride, end_idx=-1)
        self.assertEqual(len(video_batches), 43,
                         'total number of batches was not equal to 43')
        self.assertEqual(video_batches[-1][-1, -1, -1, -1], 229,
                         'last element of last batch was not 229')


if __name__ == "__name__":
    unittest.main()
