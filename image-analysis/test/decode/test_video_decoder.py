
from pathlib import Path
from decode import video_decoder as vd

import os
import shutil
import unittest
import skvideo
import skimage


class TestVideoDecoder(unittest.TestCase):

    # TODO:

    def setUp(self):
        self.video_path = os.getcwd() + "/test/decode/testing_files/IASample.mpeg"
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3

    def test_load_end(self):
        start = 1200
        frames_end = vd.decode_mpeg(self.video_path, start_idx=start)
        self.assertEqual(len(frames_end), self.total_frames - start)

    def test_load_start(self):
        end = 10
        frames_start = vd.decode_mpeg(self.video_path, end_idx=end)
        self.assertEqual(len(frames_start), 10)

    def test_load_stride(self):
        stride = 2
        frames_stride = vd.decode_mpeg(self.video_path, end_idx=10, stride=stride)

    def test_load_batches(self):
        batch_size = 2
        frames_batch = vd.decode_mpeg(self.video_path, end_idx=10, batch_size=batch_size, stride=batch_size)
        # test batch sizes
        test_batch = frames_batch[0]
        self.assertEqual(len(frames_batch), 5)
        self.assertEqual(len(test_batch), batch_size)  # frames per batch

    def test_load_frame(self):
        frames = vd.decode_mpeg(self.video_path, end_idx=1)
        frame = frames[0]  # grab a frame
        self.assertEqual(frame.shape[0], self.frames_height)
        self.assertEqual(frame.shape[1], self.frames_width)
        self.assertEqual(frame.shape[2], self.frames_channels)

    def test_decode_mpeg(self):
        frames_all = vd.decode_mpeg(self.video_path)
        self.assertEqual(len(frames_all), self.total_frames)


if __name__ == "__name__":
    unittest.main()
