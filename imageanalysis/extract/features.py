import skimage.color
import skimage.transform
import skimage.io


class Feature:
    """
    user defined subclasses should set (1) EITHER self.batch_op OR
    self.frame_op to True, (2) set a key_name and (3) implement extract()
    """
    def __init__(self, key_name, batch_op=False, frame_op=False):
        self.batch_op = batch_op
        self.frame_op = frame_op
        self.key_name = key_name

    def extract(self, **inputs_for_feature_extraction):
        raise NotImplementedError


class RGBToGray(Feature):
    def __init__(self):
        Feature.__init__(self, 'grayscale', frame_op=True)

    def extract(self, RGB_frame):
        return skimage.color.rgb2gray(RGB_frame)


class BatchOP(Feature):
    def __init__(self):
        Feature.__init__(self, 'batch_length', batch_op=True)

    def extract(self, batch):
        return len(batch)
