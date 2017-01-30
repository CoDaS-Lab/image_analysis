
from pathlib import Path
from decode import video_decoder as vd

import os
import shutil
import unittest
import skvideo
import skimage
import math


class TestVideoDecoder(unittest.TestCase):

    def setUp(self):
        self.video_path = os.getcwd() + "/test/decode/testing_files/IASample.mpeg"
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3

    def test_load_frame(self):
        frames = vd.decode_mpeg(self.video_path, end_idx=0)
        batch = frames[0]  # grab a batch
        frame = batch[0]

        self.assertEqual(batch.shape[0], 1)  # only one frame in batch
        self.assertEqual(frame.shape[1], self.frames_height)
        self.assertEqual(frame.shape[2], self.frames_width)
        self.assertEqual(frame.shape[3], self.frames_channels)

    def test_decode_mpeg(self):
        # Load video file in batches of 30 frames
        batch_size = 30
        num_batches = math.ceil(self.total_frames / batch_size)
        frames_batch = vd.decode_mpeg(self.video_path, end_idx=self.total_frames - 1, batch_size=batch_size, stride=batch_size)

        # test batch sizes
        test_batch = frames_batch[0]
        self.assertEqual(len(frames_batch), num_batches)
        self.assertEqual(len(test_batch), batch_size)


if __name__ == "__name__":
    unittest.main()
