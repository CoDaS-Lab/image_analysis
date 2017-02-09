
from pathlib import Path
import os
import shutil
import unittest

import skvideo

from extract import feature_extractor as fe
from decode import video_decoder as vd


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.video_path = os.getcwd() + "#"
        self.video_frames = vd.decode_mpeg(self.video_path, end_idx=0,
                                           batch_size=1)
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3

    def test_make_frames_dict(self):
        # TODO test metadata
        frame_dict = fe.gen_features(self.video_frames,
                                     {"grayscale": fe.grayscale})

        for frame in frame_dict:
            self.assertIsNotNone(frame["input"]["frame"])
            self.assertIsNotNone(frame["input"]["grayscale"])
            self.assertIsNotNone(frame["metadata"]["frame_num"])
            self.assertIsNotNone(frame["metadata"]["batch_num"])
if __name__ == "__name__":
    unittest.main()