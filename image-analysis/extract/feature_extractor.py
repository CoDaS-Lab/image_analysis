def mk_features(batch_list, feature_list):
    """
    INPUTS:
    batch_list: takes list of numpy arrays as input
    feature_list: takes objects of type feature extractor

    OUTPUT: list of dictionaries

    """
    frame_dictionary = []
    count_batch = 0
    count_frame = 0

    for batch in batch_list:
        for frame in batch:
            frame_dictionary.append({
                "input":{
                    "frame": frame
                    "fft_output": 1
                    },
                "metadata": {
                    "frame_num": count_frame,
                    "batch_num": count_batch
                    }  
                })
            count_frame +=1
        count_batch += 1
