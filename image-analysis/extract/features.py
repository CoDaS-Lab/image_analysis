import skimage.color
import skimage.transform
import skimage.io
import skvideo.io


class Feature:
    """
    user defined subclasses should set (1) EITHER self.batch_op OR
    self.frame_op to True, (2) set a key_name and (3) implement extract()
    """
    def __init__(self, key_name, batch_op, frame_op):
        self.batch_op = batch_op
        self.frame_op = frame_op
        self.key_name = key_name
        # raise NotImplementedError

    def is_batch_op(self):
        return self.batch_op

    def is_frame_op(self):
        return self.frame_op

    def extract(self, **inputs_for_feature_extraction):
        raise NotImplementedError


class RGBToGray(Feature):
    # requires skimage.color
    def __init__(self):
        Feature.__init__(self, 'Grayscale', False, True)

    def extract(self, RGB_frame):
        return skimage.color.rgb2gray(RGB_frame)


class BatchOP(Feature):
    def __init__(self):
        Feature.__init__(self, 'batch_length', True, False)

    def extract(self, batch):
        return len(batch)
"""
class FFT(Featur):
    def __init__(self):
        self.bath_op = True
        self.frame_op = False
        self.key_name = 'FFT'

        def extract(self, RGB_frame):
"""
