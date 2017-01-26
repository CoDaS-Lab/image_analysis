
from pathlib import Path
from importer import video_decoder as vd

import os
import shutil
import unittest
import skvideo
import skimage


class TestVideoDecoder(unittest.TestCase):

    # TODO:

    def setUp(self):
        self.video_path = "testing_files/IASample.mpeg"
        self.video_ext = "mpeg"
        self.video_out_frames_dir = "testing_files/frames"
        self.frame_out_ext = ".jpg"
        self.start_idx = 10
        self.end_idx = 20
        self.stride = 5
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3
        self.batch_size = 10 # means we will extract 100 frames in batches of 10 so 100/10 means 10 total batches


    def tearDown(self):
        shutil.rmtree(self.video_out_frames_dir)


    def test_decode_mpeg(self):
        # creates array in different ways of starts, ends, strides, and batches
        frames_start = vd.decode_mpeg(self.video_path, end_idx=self.end_idx)
        frames_end = vd.decode_mpeg(self.video_path, start_idx=self.start_idx)
        frames_stride = vd.decode_mpeg(self.video_path, stride=self.stride)
        frames_batch = vd.decode_mpeg(self.video_path, end_idx=99, batch_size=self.batch_size)
        frames_all = vd.decode_mpeg(self.video_path)

        # test whether your're getting correct number of frames
        self.assertEqual(len(frames_all), self.total_frames)
        self.assertEqual((len(frames_start), self.end_idx + 1)
        self.assertEqual(len(frames_end), self.end_idx - self.start_idx + 1)
        self.assertLess(len(frames_stride), len(frames_all))
        # test batch sizes
        self.assertEqual(len(frames_batch), 10) # 10 batches
        self.assertEqual(frames_batch[0].shape[0], self.batch_size) # 10 frames per batch

        # test dimension, frames are numpy arrays so we can just use numpy's shape function
        frame = frames_all[0] # grab a frame
        self.assertEqual(frame.shape[0], self.frames_height)
        self.assertEqual(frame.shape[1], self.frames_width)
        self.assertEqual(frame.shape[2], self.frames_channels)


    def test_decode_and_save(self):
        try:
            fail_test = vd.decode_mpeg()
        except ValueError:
            print("ValueError caught: No input Directory")

        os.makedirs(self.video_out_frames_dir)
        vd.decode_mpeg(self.video_path, out_frame_dir=self.video_out_frames_dir)

        # test whether frames were created
        test_frame = Path(self.video_out_frames_dir + "/frame0.jpg")
        assertTrue(test_frame.is_file())


if __name__ == "__name__":
    unittest.main()
