import skimage.color
import skimage.io
import skvideo.io


class Feature:
    """
    user defined subclasses should set (1) EITHER self.batch_op OR
    self.frame_op to True, (2) set a key_name and (3) implement extract()
    """
    def __init__(self, key_name):
        self.batch_op = False
        self.frame_op = False
        self.key_name = key_name
        raise NotImplementedError

    def is_batch_op(self):
        return self.batch_op

    def is_frame_op(self):
        return self.frame_op

    def extract(self, *inputs_for_feature_extraction):
        raise NotImplementedError


class RGBToGray(Feature):
    # requires skimage.color
    def __init__(self):
        self.batch_op = False
        self.frame_op = True
        self.key_name = "Grayscale"

    def extract(self, RGB_frame):
        return skimage.color.rgb2gray(RGB_frame)


class ImageScale(Feature):
    def __init__(self):
        self.batch_op = False
        self.frame_op = True
        self.key_name = "imgscale"

    def extract(self, frame):
        returns skimage.transform.rescale(frame, 10)
"""
class FFT(Featur):
    def __init__(self):
        self.bath_op = True
        self.frame_op = False
        self.key_name = "FFT"

        def extract(self, RGB_frame):
"""
