import unittest
import os
import sys
top_dir = os.path.dirname(__file__) + '/../../'
sys.path.append(top_dir)
from decode import video_decoder as vd
from extract import feature_extractor as fe
from extract import features


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.video_path = '../test_data/test_video.mp4'

    def test_gen_frame_features(self):
        # load 9 frames
        video = vd.decode_mpeg(self.video_path, end_idx=9)
        frames = fe.gen_frame_features(video, [features.RGBToGray()])

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            self.assertIsNotNone(frame['input']['grayscale'])

    def test_gen_batch_features(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg(self.video_path, batch_size=3,
                               stride=3, end_idx=9)
        batches = fe.gen_batch_features(video, [features.BatchOP()])

        for batch in batches:
            self.assertIsNotNone(batch['input'])
            self.assertIsNotNone(batch['input']['batch_length'])

    def test_batch_to_frame_dictionaries(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg(self.video_path, batch_size=3,
                               stride=3, end_idx=8)
        batch_features = fe.gen_batch_features(video, [features.BatchOP()])
        frames = fe.batch_to_frame_dictionaries(batch_features)

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            # check features are ok
            self.assertIsNotNone(frame['input']['batch_length'])

        del video

    def test_extract_features(self):
        video = vd.decode_mpeg(self.video_path, end_idx=8)
        grayscale = features.RGBToGray

        frames = fe.extract_features(video, [features.RGBToGray])

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            for feature in frame['input']:
                self.assertIsNotNone(frame['input'][feature])


if __name__ == '__name__':
    unittest.main()
