import numpy as np


class Pipeline:
    """
    This class is the glue that binds the extracting of features together with
    running the statistical models. It keeps the frames as dictionaries by
    default but there is a function to transform them to numpy arrays.
    """

    def __init__(self, data=None, parallel=False, save=False, operations=None,
                 models=None):
        if operations is None:
            self.operations = []
        else:
            self.operations = operations

        if models is None:
            self.models = []
        else:
            self.models = models

        if data is not None:
            self.data = list(data)
            self.nbatches = len(self.data)
            self.nframes_per_batch = len(self.data[0])
        else:
            self.data = data

        self.batch_operations = []
        self.frame_operations = []
        self.save = save
        self.parallel = parallel

        for op in self.operations:
            if op.batch_op:
                self.batch_operations.append(op)
            elif op.frame_op:
                self.frame_operations.append(op)
            else:
                raise ValueError('{0} is not a valid feature'.format(op))

    def transform(self):
        """
        DESCRIPTION:
            extracts or transforms the data

        RETURN:
            transformed_data
        """
        if self.data is None:
            raise ValueError('No data in the pipeline')

        if self.parallel and self.save:
            return self.transform_regular_save()
        elif self.parallel and not self.save:
            raise ValueError('Can\'t extract all features and not save at\
                             the same time')
        elif not self.parallel and not self.save:
            return self.transform_sequential()
        elif not self.parallel and self.save:
            return self.transform_sequential_save()

    # TODO: implement this function
    def transform_sequential_save(self):
        """
        DESCRIPTION:
            extracts information in sequential order of features in the lists.
            Taking each output of the operations as input to the next but
            saves  the  data

        RETURN:
            transformed_data
        """
        pass

    def transform_sequential(self):
        """
        DESCRIPTION:
            extracts information in sequential order of features in the lists.
            Taking each output of the operations as input to the next but
            does not save the transformed data

        RETURN:
            transformed_data
        """
        frame_count = 0
        batch_count = 0
        self.data_dict = []

        for batch in self.data:
            batch_dict = []
            batch_transforms = {}
            # temp data will get fed to next operation for each operation
            # create copy of batch don't change batch
            last_op_name = ''
            temp_batch = batch

            for op in self.batch_operations:
                # remove the last data added and we'll pass to next
                # operation in pipeline
                temp_batch = op.extract(temp_batch)
                last_op_name = op.key_name

            # add extracted stuff to batch_transforms
            # if last_op_name isn't empty then we ran batch ops
            if last_op_name != '':
                batch_transforms.update({last_op_name: temp_batch})

            for frame in batch:
                # since we running sequentialy only last operation is added
                # well keep track of  key of last operation
                last_op_name = 'original'
                frame_transforms = {}
                temp = frame

                for op in self.frame_operations:
                    # extract next feature based on current feature(sequential)
                    temp = op.extract(temp)
                    last_op_name = op.key_name

                # add batch features, if any
                frame_transforms.update(batch_transforms)
                # add frame features, if any
                frame_transforms.update({last_op_name: temp})

                # create dict of frame features
                metadata = {
                    'frame_num': frame_count,
                    'batch_num': batch_count
                }
                frame_dict = self.create_dict(transforms=frame_transforms,
                                              metadata=metadata)
                batch_dict.append(frame_dict)
                frame_count += 1

            batch_count += 1
            self.data_dict.append(batch_dict)

        return self.data_dict

    def transform_regular_save(self):
        """
        DESCRIPTION:
            extracts all information in operations list. By regular we mean
            that the transformations have no dependencies between each other.
            This one saves all extracted features.

        RETURN:
            transformed_data
        """
        frame_count = 0
        batch_count = 0
        self.data_dict = []

        for batch in self.data:
            batch_transforms = {}
            batch_dict = []

            # extract batch stuff
            for op in self.batch_operations:
                batch_transforms.update({op.key_name: op.extract(batch)})

            for frame in batch:
                frame_transforms = {
                    'original': np.copy(frame)
                }
                # add batch stuff to frame_transforms
                frame_transforms.update(batch_transforms)

                # extract frame stuff
                for op in self.frame_operations:
                    frame_transforms.update({op.key_name: op.extract(frame)})

                metadata = {
                    'frame_num': frame_count,
                    'batch_num': batch_count
                }

                # create frame dictionary with all the extracted stuff
                # and metadata
                frame_dict = self.create_dict(transforms=frame_transforms,
                                              metadata=metadata)
                batch_dict.append(frame_dict)
                frame_count += 1

            batch_count += 1
            self.data_dict.append(batch_dict)

        return self.data_dict

    def create_dict(self, transforms=None, metadata=None):
        """
        DESCRIPTION:
            creates the dictionary data structure for each frame

        RETURN:
            frame as dictionary
        """
        if transforms is None:
            transforms = {}

        if metadata is None:
            metadata = {}

        frame_dict = {
            'input': {},
            'metadata': {}
        }

        for key, value in transforms.items():
            frame_dict['input'].update({key: value})

        for key, value in metadata.items():
            frame_dict['metadata'].update({key: value})

        return frame_dict

    def data_as_nparray(self, data=None):
        # TODO: write test
        """
        DESCRIPTION:
            returns the data as numpy arrays

        PARAMS:
            data: data to return as np array. If none is passed the the
                  pipeline's data is used

        RETURN:
            data in numpy array form
        """
        data_dict = data
        if data_dict is None:
            data_dict = self.data_dict

        output = []
        for batch in data_dict:
            temp_batch = []
            for frame in batch:
                feature_maps = []
                # create the feature map at each frame
                for key, val in frame['input'].items():
                    feature_maps.append(val)

                temp_batch.append(np.array(feature_maps))

            output.append(np.array(temp_batch))
        return np.array(output)

    def train_models(self):
        pass

    def predict(self):
        pass
