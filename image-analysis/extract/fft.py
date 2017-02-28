import numpy as np
import pyfftw
import math
from feature import Feature


class FFT(Feature):
    def __init__(self):
        Feature.__init__(self, 'FastFourierTransform', frame_op=True)

    def filter(center_orientation, orientation_width, high_cutoff, low_cutoff,
               target_size, falloff=''):

        """
        DESCRIPTION:
            Creates a filter that can be multiplied by the amplitude spectrum of an
            image to increase/decrease specific orientations/spatial frequencies.

        INPUTS:
            center_orientation: int for the center orientation (0-180).
            orientation_width: int for the orientation width of the filter.
            high_cutoff: int high spatial frequency cutoff.
            low_cutoff: int low spatial frequency cutoff.
            target_size: int total size.
            falloff: string 'triangle' or 'rectangle' shape of the filter
                     falloff from the center.

        OUTPUTS:
            filt: return the bowtie shaped filter.
        """
        if (target_size % 2) != 0:
            raise ValueError('Target_size should be even!')

        x = y = np.linspace(0, target_size / 2, target_size / 2 + 1)
        u, v = np.meshgrid(x, y)

        # derive polar coordinates: (theta, radius), where theta is in degrees
        theta = np.arctan2(v, u) * 180 / np.pi
        radii = (u**2 + v**2) ** 0.5
        del u, v

        # using radii for one quadrant, build the other 3 quadrants
        flipped_radii = np.fliplr(radii[:, 1:target_size / 2])
        radii = np.concatenate((radii, flipped_lr_radii), axis=1)
        flipped_radii = np.flipud(radii[1:target_size / 2, :])
        radii = np.concatenate((radii, flipped_radii), axis=0)
        radii = np.fft.fftshift(radii)  # come back and GPU optimize FFT
        # note: the right-most column and bottom-most row were sliced off
        del flipped_radii

        # using theta for one quadrant, build the other 3 quadrants
        flipped_theta = 90 + np.fliplr((theta[1:target_size / 2 + 1, :].T))
        # note: +1 is done for theta, but not for radii
        # note: transpose is done for theta, but not for radii
        theta = np.concatenate((flipped_theta, theta), axis=1)
        flipped_theta = 180 + np.flipud(np.fliplr(theta[1:, :]))
        # might be able to optimize by transposing and then flipping
        # instead of flip and then flip
        theta = np.concatenate((flipped, theta), axis=0)
        del flipped_theta

        center_orientation_2 = 180 + center_orientation
        # The 2D frequency spectrum is mirror symmetric, orientations must be
        # represented on both sides. All orientation functions below must be
        # repeated using both center_orientation's

        # clockwise orientation cutoff, from center_orientation
        cwb1 = center_orientation + width / 2
        # counterclockwise orientation cutoff, from center_orientation
        ccwb1 = center_orientation - width / 2
        # clockwise orientation cutoff, from center_orientation_2
        cwb2 = center_orientation_2 + width / 2
        # counterclockwise orientation cutoff, from center_orientation_2
        ccwb2 = center_orientation_2 - width / 2

        if ccwb1 < 0:
            theta = np.fliplr(theta).T
            center_orientation += 90
            center_orientation_2 += 90
            cwb1 += 90
            ccwb1 += 90
            cwb2 += 90
            ccwb2 += 90

        # theta = theta[0:target_size,0:target_size]; only need this to check 
        # dim's
        # sffilter = np.zeros(radii.shape)
        # anfilter = np.zeros(theta.shape)

        sffilter = (low_cutoff <= radii) & (radii <= high_cutoff)

        if falloff == 'rectangle':
            anfilter = ((ccwb1 <= theta) & (theta <= cwb1)) | (
                (ccwb2 <= theta) & (theta <= csb2))
            # filt = sffiler*anfilter
        elif falloff == 'triangle':
            for idx, val in np.ndenumerate(theta):
                if ccwb1 <= val <= cwb1 and val <= center_orientation:
                    anfilter[idx] = (val - center_orientation + width / 2) \
                        * 2 / width
                elif ccwb1 <= val <= cwb1 and val > center_orientation:
                    anfilter[idx] = (-val + center_orientation_2 + width / 2) \
                        * 2 / width
                elif ccwb2 <= val <= cwb2 and val <= center_orientation_2:
                    anfilter[idx] = (val - center_orientation_2 + width / 2) \
                        * 2 / width
                elif ccwb2 <= val <= cwb2 and val > center_orientation_2:
                    anfilter[idx] = (-val + center_orientation_2 + width / 2) \
                        * 2 / width
                else:
                    anfilter[idx] = 0
        else:
            angfilter1 = np.exp(-((theta - center_orientation) / (.5 * width))\
                                ** 4)
            angfilter1 = np.exp(-((theta - center_orientation_2) / (.5 * width))\
                                ** 4)
            anfilter = angfilter1 + angfilter2

        return(sffilter * anfilter)

    def noise_amp(sz):
        """
        DESCRIPTION:
            Creates a sz * sz matrix of randomly generated noise with amplitude
            1/f slope

        INPUT:
            sz: size of matrix

        OUTPUT:
            returns the amplitude
        """
        x = y = np.linspace(1, sz, sz)
        u, v = np.meshgrid(x, y)
        u -= sz / 2
        v -= sz / 2

        amplitude = np.flipud(np.fliplr(np.fft.fftshift(((u**2 + v**2) ** 0.5) /
                                        sz * (2)**.5)))
        amplitude[0, 0] = 1
        amplitude = 1 / amplitude
        amplitude[0, 0] = 0
        return amplitude

    # TODO: implement using scikit-cuda
    def fft_gpu(self, input_frame, filter_mask):
        """
        DESCRIPTION:
            Transforms a matrix using fft using gpu, multiplies the
            result by a mask, and then transforms the matrix back using ifft.

        INPUT:
            input_frame: numpy array of pixel values
            filter_mask: int determining the type of filter to implement 1=iso,
            2 = horizontal decrement

        OUTPUT:
            Return the transformed and processed frame.
        """
        pass

    # TODO: implement using pyfftw
    def fft_cpu(self, input_frame, filter_mask):
        """
        DESCRIPTION:
            Transforms a matrix using fft using cpu (parallelized), multiplies
            the result by a mask, and then transforms the matrix back using
            ifft

        INPUT:
            input_frame: numpy array of pixel values
            filter_mask: int determining the type of filter to implement 1=iso,
            2 = horizontal decrement

        OUTPUT:
            Return the transformed and processed frame.
        """
        pass

    def extract(serlf, input_frame, filter_mask, gpu=False):
        if gpu:
            return self.fft_gpu(input_frame, filter_mask)
        else:
            return self.fft_cpu(input_frame, filter_mask)

