import os
import sys
sys.path.append(os.getcwd() + "/../")
import skimage.io
import wget
from matplotlib import pyplot as plt
from decode import video_decoder as vd
from extract import feature_extractor as fe
from demo_features import RGBToGray
from demo_features import BatchOP

# Meeting Notes
# Before presenting demo, review the 2 specified data structures
# Make everyone cd into directory containing this demo.py file
# Make everyone open this demo.py file in his/her/their favorite editor
# Make everyone run each of the commands below in terminal

# Figure out if you already have test_video.mp4 in the required directory
vid_path = os.getcwd() + '/../test/test_data/'

# Download test_video.mp4 into directory if it doesn't already exist
if not os.path.exists(vid_path + 'test_video.mp4'):
    wget.download("https://s3.amazonaws.com/codasimageanalysis/test_video.mp4", 
                  vid_path)

# Decode video to build dataset
batch_list = vd.decode_mpeg(vid_path + 'test_video.mp4',
                            batch_size=2, end_idx=10)

print('Batch_list contains {0} batches!'.format(len(batch_list)))

# Extract features (e.g. Grayscale and "BatchOP") from data
pipeline_output = fe.extract_features(batch_list,
                                     [RGBToGray, BatchOP])

# Now, let's access  some of the features extracted from frames
imgs = [data_structure[0]['input']['frame'],
        data_structure[0]['input']['grayscale']]
skimage.io.imshow_collection(imgs)
plt.show()

# You can access particular features and keys in the data structure, too!
for key, thing in data_structure[0]['input'].items():
    print(key)
