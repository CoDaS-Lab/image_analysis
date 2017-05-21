class Feature:
    """
    DESCRIPTION:
        Base class for features we want to extract or transformations we
        want to apply to data

    PARAMS:
        :batch_op: boolean to say the feature runs on batches of frames
        :frame_op: boolean to say the feature runs on each frame
        :save: boolean check to save feature in output dict
    """

    def __init__(self, key_name, batch_op=False, frame_op=False, save=False):
        self.batch_op = batch_op
        self.frame_op = frame_op
        self.key_name = key_name
        self.save = save

    def extract(self, **args):
        """
        DESCRIPTION:
            extract features from the data
        """
        raise NotImplementedError

    def train_model(self, **args):
        """
        DESCRIPTION:
            train models on images
        """
        raise NotImplementedError

    def predict(self, Y):
        """
        DESCRIPTION:
            predict new points
        """
        raise NotImplementedError
