from features import RGBToGray
import sys
import os
import skvideo
sys.path.append(os.getcwd() + "/../decode")
import video_decoder as vd
# need to fix imports

def gen_batch_features(batch_list, batch_op_list):
    """
    INPUTS:
    batch_list:     list of batches (ndarrays)
    batch_op_list:  list of batch_op objects that extract features
                    from batches

    OUTPUTS:
    batch_dictionaries: list of dictionaries, each representing a batch

    DESCRIPTION: return list of dictionaries, with each dictionary 
                    representing a batch and containing features 
                    extracted according to batch_op_list
    """
    for batch in batch_list:
        for op in batch_op_list:
            batch_dictionaries.update({up.key_name:op.extract(frame)})
    return batch_dictionaries

def gen_frame_features(batch_list, frame_op_list):
    """
    INPUTS:
    batch_list:     list of batches (numpy arrays)
    batch_op_list:  list of batch_op objects that extract features from
                    batches

    OUTPUTS:
    frame_dictionaries: list of dictionaries, each representing a frame

    DESCRIPTION: return list of dictionaries, with each dictionary 
                    representing a frame and containing features extracted
                    according to frame_op_list
    """
    frame_dictionaries = []
    for batch in batch_list:
        for frame in batch:
            frame_dictionaries.append({})
            for op in frame_op_list:
                frame_dictionaries[-1].update({op.key_name:op.extract(frame)})
    return frame_dictionaries

def batch_to_frame_dictionaries(batch_dictionaries):
    """
    INPUTS: 
    batch_dictionary: list of dictionaries representing batches (numpy 
                        arrays) of frames

    OUTPUTS: 
    frame_dictionary: list of dictionaries representing frames

    DESCRIPTION: take in list of batch dictionaries and output list of 
                    frame dictionaries
    """
    
    return frame_dictionaries


def extract_features(batch_list, op_list, feature_dictionary):
    """
    INPUTS:
        batch_list:         list of batches (numpy arrays)
        op_list:            list of objects (of type "feature")
        feature_dictionary:

    OUTPUTS: 
        frame_dictioanries: list of dictionaries, each representing a frame

    DESCRIPTION: takes in lists of (1) batches and (2) features to be 
                    extracted, and then outputs a list of dictionaries, 
                    each corresponding to a frame and containing all the 
                    specified features
    """
    batch_ops = []
    frame_ops = []
    for op in op_list:
        if op.is_batch_op() == True:
            batch_ops.append(op)
        elif op.is_frame_op() == True:
            frame_ops.append(op)
        else: print("at least one op is neither a batch_op or frame_op")
    frame_dictionaries = []
    count = 0
    for batch in batch_list:
        batch_dictionary = gen_batch_features([batch], batch_ops)
        frame_dictionaries += (batch_to_frame_dictionaries(batch_dictionary))
        if len(frame_ops) != 0:
            for frame in batch:
                frame_dictionary = gen_frame_features([[frame]], frame_ops)
                frame_dictioanries[count].update(frame_dictionary[0])
                count += 1
    return frame_dictionaries

my_batch_list = vd.decode_mpeg('/Users/Ravis/Desktop/Shafto_Lab/'
				r'Image_Analysis_Project/IASample.mpeg',
				batch_size=2, stride = 5, start_idx=1)
Gray_scale = RGBToGray()
x = gen_frame_features(my_batch_list, [Gray_scale])
print(len(x))
print(x[5]['Grayscale'].shape)
