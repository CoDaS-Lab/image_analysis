import unittest
import os
import wget
import time
import sys
import numpy as np
from decode import video_decoder as vd
from extract.orientation_filter import OrientationFilter


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

    def test_bowtie(self):
        # Test the bowtie filter with sine waves
        wDeg = 3
        nPix = 257
        sf = 1
        orientation = 0

        x = y = np.linspace(-wDeg // 2, wDeg // 2, nPix + 1)
        u, v = np.meshgrid(x, y)
        # u = u[:-2, :-2]
        # v = v[:-2, :-2]

        ramp = np.sin(orientation * np.pi / 180) * u
        ramp -= np.cos(orientation * np.pi / 180) * v
        sinwaves_grating = np.sin(2 * np.pi * sf * ramp)

        mask_filter = OrientationFilter(mask='bowtie', falloff='triangle')
        filt = mask_filter.bowtie(90, 20, nPix, .1, nPix + 1, 'triangle')

        # TODO finish this multiply filt by the sinwaves_grating and it should
        # make the sine wave disappear


if __name__ == '__name__':
    unittest.main()
