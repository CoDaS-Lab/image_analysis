import os
import sys
sys.path.append(os.getcwd() + "/../")
import skimage.io
import numpy as np
from matplotlib import pyplot as plt
from extract.orientation_filter import OrientationFilter
from extract.pipeline import Pipeline
from skimage.color import rgb2gray

# Test the bowtie filter with sine waves
# img = skimage.io.imread('../test1.jpg')[:426, :426, :]
img = skimage.io.imread('bricks.jpg')[:1000, :1000, :]

motion_analysis = Pipeline(data=[[img]],
                           ops=[OrientationFilter('bowtie', 90, 42, 1000, .2,
                                                  1000, 'triangle')],
                           save_all=True)

pipeline_output = motion_analysis.extract()
# print(pipeline_output[-1]['seq_features'])
print(pipeline_output[0]['frame_features']['bowtie_filter'].shape)

# print(motion_analysis.as_ndarray(seq_key='batch_length'))
# motion_analysis.display()
# Now, let's access  some of the features extracted from frames
# img_filtered_amp_spectrum = np.load('filtered_img_amp_spectrum.npy')

imgs = [pipeline_output[0]['input'],
        pipeline_output[0]['frame_features']['bowtie_filter']]
# filtered_minus_original = imgs[1] - rgb2gray(imgs[0])
# skimage.io.imshow_collection(imgs)
# plt.imshow(rgb2gray(pipeline_output[0]['input']), cmap='gray')
plt.imshow(pipeline_output[0]['frame_features']['bowtie_filter'], cmap='gray')
# plt.imshow(filtered_minus_original)
plt.show()
# plt.imshow(img_filtered_amp_spectrum)
# plt.show()
