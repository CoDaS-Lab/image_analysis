import skimage.color
import skimage.transform
import skimage.io


class Feature:
    """
    user defined subclasses should set (1) EITHER self.batch_op OR
    self.frame_op to True, (2) set a key_name and (3) implement extract().

    This can also be used as a machine learning model
    """
    def __init__(self, key_name, batch_op=False, frame_op=False, **args):
        self.batch_op = batch_op
        self.frame_op = frame_op
        self.key_name = key_name

    def extract(self, **args):
        """
        DESCRIPTION:
            extract features from the data
        """
        raise NotImplementedError

    def train_model(self, **args):
        raise NotImplementedError

    def predict(self, Y):
        raise NotImplementedError
