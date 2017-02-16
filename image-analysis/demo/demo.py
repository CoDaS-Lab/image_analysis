import sys, os
# to find modules
sys.path.append(os.getcwd() + "/../")

import numpy as np
import image_analysis as codas

# install matplotlib if you want to display!
from matplotlib import pyplot as plt


video = codas.video_dec.decode_mpeg(os.getcwd() + '/city.mp4', end_idx=2)
gray_feat = codas.features.RGBToGray()
grayscale_frames = codas.feature_extractor.gen_frame_features(video, [gray_feat])

frame = grayscale_frames[0]
# display a gray frame to make sure it worked
codas.skimage.io.imshow(frame["input"][gray_feat.key_name])
plt.show()
