import unittest
from sklearn.pipeline import Pipeline
from decode import video_decoder as vd
from extract.pixel_transforms import Grayscale


class TestPixelTransforms(unittest.TestCase):
    def setUp(self):
        self.video_path = 'test/test_data/video.mp4'

    def test_grayscale_transform(self):
        video = vd.decode_mpeg(self.video_path, end_idx=9)

        basic_transform = Grayscale()

        basic_pipeline = Pipeline(steps=[
            ('grayscale', basic_transform)
        ])

        frames = basic_pipeline.transform(video)
        self.assertEqual(len(frames), len(video))

        for frame in frames:
            self.assertIsNotNone(frame['input']['frame'])
            self.assertIsNotNone(frame['input']['grayscale'])

        del video

if __name__ == '__main__':
    unittest.main()
