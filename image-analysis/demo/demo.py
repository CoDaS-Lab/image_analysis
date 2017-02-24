import os, sys
# to find modules
sys.path.append(os.getcwd() + "/../")
import image_analysis as codas
import skimage.io
from matplotlib import pyplot as plt
import wget

# Meeting Notes
# Before presenting demo, re-+view the 2 specified data structures
# Make everyone cd into directory containing this demo.py file
# Make everyone open this demo.py file in his/her/their favorite editor
# Make everyone run each of the commands below in terminal

# load 10 images so it runs fast
vid_path = os.getcwd() + '/../test/test_data'
already_have = False
for x in os.listdir(os.getcwd() + '/../test/test_data/'):
    if x == 'test_video.mp4':
        already_have = True
if already_have == False:
    wget.download("https://s3.amazonaws.com/testcodas/test_video.mp4", vid_path)

batch_list = codas.vd.decode_mpeg(os.getcwd() + '/../test/test_data/test_video.mp4', end_idx=10)

print('Batch_list contains {0} batches!'.format(len(batch_list)))

data_structure = codas.fe.extract_features(batch_list,
                                           [codas.RGBToGray, codas.BatchOP])

imgs = [data_structure[0]['input']['frame'],
        data_structure[0]['input']['Grayscale']]
# Now, let's access  some of the features extracted from frames
skimage.io.imshow_collection(imgs)
plt.show()


for key, thing in data_structure[0]['input'].items():
    print(key)
