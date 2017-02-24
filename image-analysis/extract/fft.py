import numpy as np
import pyfftw
import math
from feature import Feature


class FFT(Feature):
    def __init__(self):
        Feature.__init__(self, 'FastFourierTransform', frame_op=True)

    def filter(self, center_orientation, width, high, low, sz, falloff):
        """
        INPUTS:
        center_orientation: int for the center orientation (0-180).
        width: int for the orientation width of the filter.
        high: int high spatial frequency cutoff.
        low: int low spatial frequency cutoff.
        sz: int total size.
        falloff: 'triangle' or 'rectangle' shape of the filter falloff from 
        the center.

        OUTPUT:
        the bowtie shaped filter.

        DESCRIPTION:
        Creates a filter that can be multiplied by the amplitude
        spectrum of an image to increase/decrease specific orientations/spatial 
        frequencies.
        """

        nx = sz
        ny = sz

        x_intervals = np.linspace(0, nx // 2, (nx // 2) + 1)
        y_intervals = np.linspace(0, ny // 2, (ny // 2) + 1)

        xv, yv = np.meshgrid(x_intervals, y_intervals)
        theta = np.arctan2(yv, xv) * (180 / math.pi)  # in degrees
        radius = np.sqrt(xv**2 + yv**2)

        del xv
        del yv

        flipped_radius = np.fliplr(radius[0:, 1:(ny // 2)])
        radius = np.concatenate((radius, flipped_radius), axis=1)
        flipped_radius = np.flipud(radius[1:(nx // 2), 0:])
        radius = np.concatenate((radius, flipped_radius), axis=0)
        radius = pyfftw.interfaces.numpy_fft.fftshift(radius)
        radius1 = radius

        del flipped_radius

        flipped_theta = np.fliplr((theta[1:(nx // 2) + 1, 0:]).T) + 90
        theta = np.concatenate((flipped_theta, theta), axis=1)
        flipped_theta = np.flipud(np.fliplr(theta[1:, 0:])) + 180
        theta = np.concatenate((flipped_theta, theta), axis=0)
        theta1 = theta

        del flipped_theta

        # parameters
        centor1 = center_orientation
        centor2 = cantor1 + 180
        wid = width
        half_wid = np.divide(wid, 2.0)
        hi_sf = high
        lo_sf = low
        rectangle = 0
        triangle = 0
        exponent = 4

        if falloff == 'rectangle':
            rectangle = 1
        elif falloff == 'triangle:
            triangle = 1

        centor_width1 = centor1 + half_wid
        c_centor_width1 = centor1 - half_wid
        centor_width2 = centor2 + half_wid
        c_centor_width2 = centor2 - half_wid

        if c_centor_width1 < 0:
            theta = np.fliplr(theta).T
            centor1 += 90
            centor2 += 90
            centor_width1 += 90
            c_centor_width1 += 90
            centor_width2 += 90
            c_centor_width2 += 90

        theta1 = theta1[0:sz, 0:sz]
        # spatial frequency filter
        sf_filter = np.zeros(raidus1.shape)
        # angle filter
        an_filter = np.zeros(theta1.shape)

        for index, val in np.ndenumerate(radius1):
            if lo_sf <= val <= hi_sf:
                sf_filter[index] = 1
            else:
                sf_filter[index] = 0

        if rectangle == 1:
            for index, val in np.ndenumerate(theta1):
                if c_centor_width1 <= val <= centor_width1 or c_centor_width2 \
                        <= val <= centor_width2:
                        an_filter[index] = 1
                else:
                    an_filter[index] = 0
            filt = sf_filter * an_filter
        elif triangle == 1:
            for index, val in np.ndenumerate(theta1):
                if c_centor_width1 <= val <= centor_width1 and val <= centor1:
                    an_filter[index] = np.divide(val - centor1 + half_wid,
                                                 half_wid)
                elif c_centor_width1 <= val <= centor_width1 and val > centor1:
                    an_filter[index] = np.divide(-val + centor2 + half_wid,
                                                 half_wid)
                elif c_centor_width2 <= val <= centor_width2 and val <= centor2:
                    an_filter[index] = np.divide(val - centor2 + half_wid,
                                                 half_wid)
                elif c_centor_width2 <= val <= centor_width2 and val > centor2:
                    an_filter[index] = np.divide(-val + centor2 + half_wid,
                                                 half_wid)
                else:
                    an_filter[index] = 0
        else:
            ang_filter1 = np.zeros(theta1.shape)
            ang_filter2 = np.zeros(theta1.shape)

            for index, val in np.ndenumerate(theta1):
                ang_filter1[index] = math.e**(-(np.divide(val - centor1,
                                              .5 * wid)) ** exponent)
                ang_filter2[index] = math.e**(-(np.divide(val - centor2,
                                              .5 * wid)) ** exponent)

            an_filter = ang_filter1 + ang_filter2
        filt = sf_filter * an_filter

        return filt

    # TODO: implement using scikit-cuda
    def fft_gpu(self, input_frame, filter_mask):
        """
        INPUT:
        input_frame: numpy array of pixel values
        filter_mask: int determining the type of filter to implement 1 = iso,
        2 = horizontal decrement

        OUTPUT:
        Return the transformed and processed frame.

        DESCRIPTION:
        Transforms a matrix using fft using gpu, multiplies the
        result by a mask, and then transforms the matrix back using ifft.
        """
        pass

    # TODO: implement using pyfftw
    def fft_cpu(self, input_frame, filter_mask):
        """
        INPUT:
        input_frame: numpy array of pixel values
        filter_mask: int determining the type of filter to implement 1 = iso,
        2 = horizontal decrement

        OUTPUT:
        Return the transformed and processed frame.

        DESCRIPTION:
        Transforms a matrix using fft using cpu (parallelized), multiplies the
        result by a mask, and then transforms the matrix back using ifft.
        """
        pass

    def extract(serlf, input_frame, filter_mask, gpu=False):
        if gpu:
            return self.fft_gpu(input_frame, filter_mask)
        else:
            return self.fft_cpu(input_frame, filter_mask)

