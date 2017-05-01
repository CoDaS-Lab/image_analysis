import unittest
import os
import wget
import time
import sys
from decode import video_decoder as vd
from extract import feature_extractor as fe
from test.extract import test_features as features


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        data_dir = 'test/test_data/'
        vid_link = 'https://s3.amazonaws.com/codasimageanalysis/test_video.mp4'
        if not os.path.exists(data_dir + 'test_video.mp4'):
            wget.download(vid_link, data_dir)

        self.video_path = data_dir + 'test_video.mp4'
        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

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

        frames = fe.extract_features(video, [features.RGBToGray,
                                             features.BatchOP])

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            self.assertIsNotNone(frame['input']['grayscale'])
            self.assertIsNotNone(frame['input']['batch_length'])


if __name__ == '__main__':
    unittest.main()
