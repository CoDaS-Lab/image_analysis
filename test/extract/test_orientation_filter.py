import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
from decode import video_decoder as vd
from pyfftw.interfaces.numpy_fft import fftshift
from pyfftw.interfaces.numpy_fft import fft2
from pyfftw.interfaces.numpy_fft import ifft2
from extract.orientation_filter import OrientationFilter


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
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
        img = np.sin(2 * np.pi * sf * ramp)
        fimg = fft2(img)

        mask_filter = OrientationFilter(mask='bowtie', falloff='triangle')
        filt = mask_filter.bowtie(90, 20, nPix, .2, nPix + 1, 'triangle')
        filt = 1 - filt
        filt = fftshift(filt)
        out = ifft2(fimg * filt).real.astype(int)

        skimage.io.imshow_collection([img, out])
        plt.show()

if __name__ == '__main__':
    unittest.main()
