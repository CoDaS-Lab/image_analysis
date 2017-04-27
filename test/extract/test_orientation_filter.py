import unittest
import os
import wget
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from decode import video_decoder as vd
from pyfftw.interfaces.numpy_fft import fftshift
from pyfftw.interfaces.numpy_fft import fft2
from pyfftw.interfaces.numpy_fft import ifft2
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

        ramp = np.sin(orientation * np.pi / 180) * u
        ramp -= np.cos(orientation * np.pi / 180) * v
        sinwaves_grating = np.sin(2 * np.pi * sf * ramp)

        mask_filter = OrientationFilter(mask='bowtie', falloff='triangle')
        filt = mask_filter.bowtie(90, 42, nPix, .1, nPix + 1, 'triangle')

        fft_grating = fft2(sinwaves_grating)
        filt_shift = fftshift(filt)

        out = fft_grating * filt_shift
        altered = ifft2(out).real
        skimage.io.imshow_collection([sinwaves_grating, altered])
        # plt.imshow(altered, cmap='gray')
        plt.show()


if __name__ == '__name__':
    unittest.main()
