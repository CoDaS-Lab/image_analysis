import skvideo.io
import skimage.io
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from image_analysis import *

video = video_dec.decode_mpeg(os.getcwd() + '/demo/city.mp4', end_idx=100)
grayFeat = features.RGBToGray()
grayscale_frames = feature_extractor.gen_frame_features(video, [grayFeat])

out = []
for i in range(len(grayscale_frames)):
    out.append(grayscale_frames[i]['input'][grayFeat.key_name])

out = np.array(out)
print(out.shape)
skvideo.io.vwrite('demo/out.mp4', out)
