
def gen_batch_features(batch_list, batch_op_list):
    """
    DESCRIPTION:
        return list of dictionaries, with each dictionary
        representing a batch and containing features
        extracted according to batch_op_list

    INPUTS:
        batch_list: list of batches (ndarrays)
        batch_op_list: list of batch_op objects that extract features
                    from batches

    OUTPUTS:
        batch_dictionaries: list of dictionaries, each representing a batch
    """
    batch_dictionaries = []
    for batch in batch_list:
        batch_ds = {
            'input': {
                'batch': batch
            },
            'metadata': {

            }
        }
        batch_dictionaries.append(batch_ds)
        for op in batch_op_list:
            batch_ds['input'].update({op.key_name: op.extract(batch)})
    return batch_dictionaries


def gen_frame_features(batch_list, frame_op_list):
    """
    DESCRIPTION:
        return list of dictionaries, with each dictionary
        representing a frame and containing features extracted
        according to frame_op_list

    INPUTS:
        batch_list: list of batches (numpy arrays)
        batch_op_list: list of batch_op objects that extract features from
                    batches

    OUTPUTS:
        frame_dictionaries: list of dictionaries, each representing a frame
    """
    frame_dictionaries = []
    for batch in batch_list:
        for frame in batch:
            frame_ds = {
                'input': {
                    'frame': frame

                },
                'metadata': {
                }
            }

            frame_dictionaries.append(frame_ds)
            for op in frame_op_list:
                frame_ds['input'].update({op.key_name: op.extract(frame)})
    return frame_dictionaries


def batch_to_frame_dictionaries(batch_dictionaries):
    """
    DESCRIPTION:
        take in list of batch dictionaries and output list of
        frame dictionaries

    INPUTS:
        batch_dictionary: list of dictionaries representing batches (numpy
                        arrays) of frames; each dictionary should have same
                        exact keys, and each batch should be an ndarry

    OUTPUTS:
        frame_dictionary: list of dictionaries representing frames
    """
    frame_dictionaries = []
    batch_index = 0
    for batch_ds in batch_dictionaries:
        for frame in batch_ds['input']['batch']:
            frame_ds = {
                'input': {
                    'frame': frame

                },
                'metadata': {
                    'batch_index': batch_index
                }
            }

            frame_dictionaries.append(frame_ds)
            for batch_op in batch_ds['input']:
                if batch_op != 'batch':
                    frame_ds['input'][batch_op] = batch_ds['input'][batch_op]
        batch_index += 1

    return frame_dictionaries


def extract_features(batch_list, op_list):
    """
    DESCRIPTION:
        takes in lists of (1) batches and (2) features to be
        extracted, and then outputs a list of dictionaries,
        each corresponding to a frame and containing all the
        specified features

    INPUTS:
        batch_list: list of batches (numpy arrays)
        op_list: list of objects (of type 'feature')

    OUTPUTS:
        frame_dictioanries: list of dictionaries, each representing a frame
    """
    batch_ops = []
    frame_ops = []
    for op in op_list:
        feature = op()
        if feature.batch_op is True:
            batch_ops.append(feature)
        elif feature.frame_op is True:
            frame_ops.append(feature)
        else:
            raise ValueError('{0} is not a valid feature'.format(feature))

    frame_dictionaries = []
    count = 0
    for batch in batch_list:
        batch_dictionary = gen_batch_features([batch], batch_ops)
        frame_dictionaries += (batch_to_frame_dictionaries(batch_dictionary))

        if len(frame_ops) != 0:
            for frame in batch:
                frame_dictionary = gen_frame_features([[frame]], frame_ops)
                frame_dictionaries[count].update(frame_dictionary[0])
                count += 1
    return frame_dictionaries
