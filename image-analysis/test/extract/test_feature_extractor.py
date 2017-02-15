
import unittest

import skvideo.io

from decode import video_decoder as vd
from extract import feature_extractor as fe
from extract import features


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.video_path = 'test/test_data/video.mp4'
        metadata = skvideo.io.ffprobe(self.video_path)
        self.total_frames = int(metadata['video']['@nb_frames'])
        self.frames_width = int(metadata['video']['@width'])
        self.frames_height = int(metadata['video']['@height'])
        self.frames_channels = 3

    def test_gen_frame_features(self):
        # load 9 frames
        video = vd.decode_mpeg(self.video_path, end_idx=9)
        grayscale = features.RGBToGray()
        frames = fe.gen_frame_features(video, [grayscale])

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            self.assertIsNotNone(frame['input'][grayscale.key_name])

        del video

    def test_gen_batch_features(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg(self.video_path, batch_size=3,
                               stride=3, end_idx=9)
        batch_op = features.BatchOP()  # TODO change to proper name
        batches = fe.gen_batch_features(video, [batch_op])

        for batch in batches:
            self.assertIsNotNone(batch['input'])
            self.assertIsNotNone(batch['input'][batch_op.key_name])

        del video

    def test_batch_to_frame_dictionaries(self):
        # load 3 batches of 3 frames each
        video = vd.decode_mpeg(self.video_path, batch_size=3,
                               stride=3, end_idx=8)
        batch_op = features.BatchOP()  # TODO change to proper name
        batch_features = fe.gen_batch_features(video, [batch_op])
        frames = fe.batch_to_frame_dictionaries(batch_features)

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            # check features are ok
            self.assertIsNotNone(frame['input'][batch_op.key_name])

        del video

    def test_extract_features(self):
        video = vd.decode_mpeg(self.video_path, end_idx=8)
        grayscale = features.RGBToGray()

        frames = fe.extract_features(video, [grayscale])

        for frame in frames:
            self.assertIsNotNone(frame['input'])
            for feature in frame['input']:
                self.assertIsNotNone(frame['input'][feature])

        del video

if __name__ == '__name__':
    unittest.main()
