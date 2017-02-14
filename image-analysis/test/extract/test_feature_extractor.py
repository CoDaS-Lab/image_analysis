
import os
import shutil
import unittest
from pathlib import Path

import skvideo

from decode import video_decoder as vd
from extract import feature_extractor as fe
from extract import features


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.total_frames = 1247
        self.frames_width = 352
        self.frames_height = 288
        self.frames_channels = 3

    def test_gen_frame_features(self):
        # load 9 frames
        video = vd.decode_mpeg("../test_data/video.mpeg", end_idx=9)
        grayscale = features.RGBToGray()
        frames = fe.gen_frame_features(video, [grayscale])

        for frame in frame_features:
            self.assertIsNotNone(frame_features["input"])
            self.assertIsNotNone(frame_features[grayscale.key_name])

        del video

    def test_gen_batch_features(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg("../test_data/video.mpeg", batch_size=3,
                               stride=3, end_idx=9)
        batch_op = features.BatchOP()  # TODO change to proper name
        batch_features = fe.gen_batch_features(video, [batch_op])

        for feature in batch_features:
            self.assertIsNotNone(batch_features[feature])

        del video

    def test_batch_to_frame_dictionaries(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg("../test_data/video.mpeg", batch_size=3,
                               stride=3, end_idx=9)
        batch_op = features.BatchOP()  # TODO change to proper name
        frames = fe.batch_to_frame_dictionaries(video, [batch_op])

        for frame frames:
            self.assertIsNotNone(frames["input"])
            # check features are ok
            self.assertIsNotNone(frames["input"][batch_op.key_name])

        del video

    def test_extract_features(self):
        video = vd.decode_mpeg("../test_data/video.mpeg", batch_size=3,
                               stride=3, end_idx=9)
        grayscale = features.RGBToGray()
        scale = features.ImageScale()

        frames = fe.extract_features(video, [grayscale, scale])

        for frame in frames:
            self.assertIsNotNone(frame["input"])
            for feature in frame["input"]:
                self.assertIsNotNone(frame["input"][feature])

        del video

if __name__ == "__name__":
    unittest.main()
