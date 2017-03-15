import numpy as np
import cv2
import math
from feature import Feature


class FFT(Feature):
    def __init__(self):
        Feature.__init__(self, 'FastFourierTransform', frame_op=True)

    def filter_func(center_orientation, orientation_width, high_cutoff,
                    low_cutoff, target_size, falloff=''):
        """
        DESCRIPTION:
            Creates a filter that can be multiplied by the amplitude spectrum
            of an image to increase/decrease specific orientations/spatial
            frequencies.

        PARMS:
            center_orientation: int for the center orientation (0-180).
            orientation_width: int for the orientation width of the filter.
            high_cutoff: int high spatial frequency cutoff.
            low_cutoff: int low spatial frequency cutoff.
            target_size: int total size.
            falloff: string 'triangle' or 'rectangle' shape of the filter
                     falloff from the center.

        RETURNS:
            filt: return the bowtie shaped filter.
        """
        if (target_size % 2) != 0:
            raise ValueError('Target_size should be even!')

        x = y = np.arange(0, target_size / 2 + 1).astype(float32)
        u, v = np.meshgrid(x, y)

        # derive polar coordinates: (theta, radius), where theta is in degrees
        theta = np.arctan2(v, u) * 180 / np.pi
        radii = (u**2 + v**2) ** 0.5

        # using radii for one quadrant, build the other 3 quadrants
        flipped_radii = np.fliplr(radii[:, 1:target_size / 2])
        radii = np.concatenate((radii, flipped_lr_radii), axis=1)
        flipped_radii = np.flipud(radii[1:target_size / 2, :])
        radii = np.concatenate((radii, flipped_radii), axis=0)
        radii = np.fft.fftshift(radii)
        # note: the right-most column and bottom-most row were sliced off

        # using theta for one quadrant, build the other 3 quadrants
        flipped_theta = 90 + np.fliplr((theta[1:target_size / 2 + 1, :].T))
        # note: +1 is done for theta, but not for radii
        # note: transpose is done for theta, but not for radii
        theta = np.concatenate((flipped_theta, theta), axis=1)
        flipped_theta = 180 + np.flipud(np.fliplr(theta[1:, :]))
        # might be able to optimize by transposing and then flipping
        # instead of flip and then flip
        theta = np.concatenate((flipped, theta), axis=0)

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

    def noise_amp(size):
        """
        DESCRIPTION:
            Creates a size x size matrix of randomly generated noise with
            amplitude values with1/f slope

        PARM:
            size: size of matrix

        RETURN:
            returns the amplitudes with noise added
        """
        x = y = np.arange(1, size).astype(float32)
        u, v = np.meshgrid(x, y)  # coordinates for a square grid
        u -= size / 2
        v -= size / 2

        amplitude = np.flipud(np.fliplr(np.fft.fftshift(
                                        (((u**2 + v**2) ** 0.5) / size) \
                                        * (2)**.5)))
        amplitude[0, 0] = 1
        amplitude = 1 / amplitude
        amplitude[0, 0] = 0

        return amplitude

    def fft_mask(input_frame, mask, plan_inverse):
        """
        DESCRIPTION:
            Transforms a matrix using FFT, multiplies the result by a mask, and
            then transforms the matrix back using Inverse FFT.

        PARMS:
            input_frame: (m x n) numpy array
            mask: int determining the type of filter to implement, where
                  1 = iso and 2 = horizontal decrement, etc.
            plan_inverse: skcuda.fft.Plan object

        RETURN:
            return the transformed and processed frame
        """
        if input_frame is None:
            return ValueError('Frame is invalid: {0}'.format(input_frame))

        if mask not in [1, 2]:
            return ValueError('Invalid mask: {0}'.format(mask))

        if plan_inverse is None:
            return ValueError('Invalid plan_inverse: {0}'.format(plan_inverse))

        # this is what requires GPU optimization
        # perform discrete Fourier transform on input frame
        dft_frame = cv3.dft(np.float32(input_frame),
                            flags=cv3.DFT_COMPLEx_OUTPUT)
        gpu_phase = gpuarray.to_gpu(cv3.phase(dft_frame[:, :, 0],
                                              dft_frame[:, :, 1]))
        output = gpu.array.empty_like(gpu_phase)
        size = np.shape(dft_frame)[1]
        if mask == 1:
            amp = noise_amp(size)
        elif mask == 2:
            amp = np.abs(dft_frame) * (1 - filter_func(90, 20, size / 2,
                                                       .1, size,
                                                       falloff='triangle'))
            # replaced "t" with " falloff='triangle'": is this oK?
        cu_fft.ifft(
            cumath.exp(gpu_phase * 1j) * amp,
            output, plan_inverse, True)

        # altimg = amp*g_complex.get()
        iframe = gpuarray.to_gpu(amp * g_complex.get())
        iframe_gpu = gpuarray.empty_like(iframe)
        cu_fft.ifft(iframe, iframe_gpu, plan_inverse, True)
        # normalize the image for display
        # should this be one by numpy instead of the gpu?
        output_frame = output.get().real
        output_frame -= output_frame.min()
        output_frame /= output_frame.max

        return output_frame

        def extract(self):
            return self
