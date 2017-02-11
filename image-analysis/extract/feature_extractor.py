def gen_batch_features(batch_list, batch_op_list):
    """
    INPUTS:
    batch_list:     list of batches (numpy arrays)
    batch_op_list:  list of batch_op objects that extract features from batches

    OUTPUTS:
    batch_dictionaries: list of dictionaries, each representing a batch

    DESCRIPTION: return list of dictionaries, with each dictionary representing
                    a batch and containing features extracted according to
                    batch_op_list
    """
    for op in batch_op_list:

    return batch_dictionaries

def gen_frame_features(batch_list, frame_op_list):
    """
    INPUTS:
    batch_list:     list of batches (numpy arrays)
    batch_op_list:  list of batch_op objects that extract features from batches

    OUTPUTS:
    frame_dictionaries: list of dictionaries, each representing a frame

    DESCRIPTION: return list of dictionaries, with each dictionary representing
                    a frame and containing features extracted according to 
                    frame_op_list
    """

    return frame_dictionaries

def batch_to_frame_dictionaries(batch_dictionaries):
    """
    INPUTS: 
    batch_dictionary: list of dictionaries representing batches (numpy arrays) 
                        of frames

    OUTPUTS: 
    frame_dictionary: list of dictionaries representing frames

    DESCRIPTION: take in list of batch dictionaries and output list of frame 
                    dictionaries
    """
    
    return frame_dictionaries


def extract_features(batch_list, op_list, feature_dictionary):
    """
    INPUTS:
        batch_list: list of batches (numpy arrays)
        op_list: list of objects (of type "feature")
        feature_dictionary:

    OUTPUTS: 
        frame_dictioanries: list of dictionaries, each representing a frame

    DESCRIPTION: takes in lists of (1) batches and (2) features to be extracted,
                    and then outputs a list of dictionaries, each corresponding
                    to a frame and containing all the specified features
    """
    batch_ops = []
    frame_ops = []
    for op in op_list:
        if op.is_batch_op() == True:
            batch_ops.append(op)
        elif op.is_frame_op() == True:
            frame_ops.append(op)
        else: print("at least one op is neither a batch_op or frame_op"))
    frame_dictionaries = []
    count = 0
    for batch in batch_list:
        batch_dictionary = gen_batch_features([batch], batch_ops)
        frame_dictionaries += (batch_to_frame_dictionaries(batch_dictionary))
        if len(frame_ops) != 0:
            for frame in batch:
                frame_dictionary = gen_frame_features([frame], frame_ops)
                frame_dictioanries[count].update(frame_dictionary[0])
                count += 1
    return frame_dictionaries


#import skimage.color as skcolor
#
#
#def gen_features(batch_list, feature_dict, metadata={}):
#    """
#    INPUTS:
#    batch_list: takes list of numpy arrays as input
#    feature_dict: takes objects of type feature extractor
#
#    OUTPUT: list of dictionaries
#
#    """
#
#    # TODO implement user metadata param in the frame dict like (follow same
#    # way as feature_dict)
#
#    frame_dictionary = []
#    count_batch = 0
#    count_frame = 0
#
#    for batch in batch_list:
#        for frame in batch:
#            frame_features = {}
#            # apply feature extraction functions in feature_dict
#            for feat in feature_dict.keys():
#                # call the feature extraction method and add the result
#                # to frame_dictionary
#                frame_features[feat] = feature_dict[feat](frame)
#
#            frame_features["frame"] = frame
#            frame_dictionary.append({
#                "input": frame_features,
#                "metadata": {
#                    "frame_num": count_frame,
#                    "batch_num": count_batch,
#                }
#            })
#            count_frame += 1
#        count_batch += 1
#
#    return frame_dictionary
#
#
#def grayscale(frame):
#    """
#    INPUTS:
#    frame: numpy array of frame pixel values
#
#    OUTPUTS: grayscale of frame
#    """
#    return skcolor.rgb2gray(frame)
#
#
#def grayscale_collection(frames):
#    """
#    INPUTS:
#    frames: batch of frames
#
#    OUTPUTS: batch grayscale of frame
#    """
#
#    gray_frames = []
#    for frame in frames:
#        gray_frames.append(grayscale(frame))
#    return gray_frames
