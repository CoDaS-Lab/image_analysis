import skimage.color as skcolor


def gen_features(batch_list, feature_dict, metadata={}):
    """
    INPUTS:
    batch_list: takes list of numpy arrays as input
    feature_dict: takes objects of type feature extractor

    OUTPUT: list of dictionaries

    """

    # TODO implement user metadata param in the frame dict like (follow same
    # way as feature_dict)

    frame_dictionary = []
    count_batch = 0
    count_frame = 0

    for batch in batch_list:
        for frame in batch:
            frame_features = {}
            # apply feature extraction functions in feature_dict
            for feat in feature_dict.keys():
                # call the feature extraction method and add the result
                # to frame_dictionary
                frame_features[feat] = feature_dict[feat](frame)

            frame_features["frame"] = frame
            frame_dictionary.append({
                "input": frame_features,
                "metadata": {
                    "frame_num": count_frame,
                    "batch_num": count_batch,
                }
            })
            count_frame += 1
        count_batch += 1

    return frame_dictionary


def grayscale(frame):
    """
    INPUTS:
    frame: numpy array of frame pixel values

    OUTPUTS: grayscale of frame
    """
    return skcolor.rgb2gray(frame)


def grayscale_collection(frames):
    """
    INPUTS:
    frames: batch of frames

    OUTPUTS: batch grayscale of frame
    """

    gray_frames = []
    for frame in frames:
        gray_frames.append(grayscale(frame))
    return gray_frames
