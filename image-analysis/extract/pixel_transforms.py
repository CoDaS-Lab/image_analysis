# This file contains all prixel operations like grayscale, color invertion ect..

import skimage.color


class Grayscale:
    """
    Converts frames to grayscale
    """

    def transform(self, frames, **transform_params):
        # dont edit original frames
        for frame in frames:
            frame['input']['grayscale'] = skimage.color.rgb2gray(frame['input']['frame'])

        return frames

    def fit(self, X, y=None, **fit_params):
        # fits data returned from transfrom since this grayscale doesn't fit
        # anything we just return nothing or self
        return self
