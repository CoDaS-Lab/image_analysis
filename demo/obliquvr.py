import os
import sys
sys.path.append(os.getcwd() + "/../")
import skimage.io
import numpy as np
from matplotlib import pyplot as plt
from decode import video_decoder as vd
from extract.orientation_filter import OrientationFilter
from extract.pipeline import Pipeline
from skimage.color import rgb2gray

# Test the bowtie filter with sine waves
img = skimage.io.imread('../test1.jpg')[:426, :426, :]

motion_analysis = Pipeline(data=[[img]],
                           ops=[OrientationFilter('bowtie', 90, 42, 426, .2,
                                                  426, 'triangle')],
                           save_all=True)

pipeline_output = motion_analysis.extract()
# print(pipeline_output[-1]['seq_features'])
print(pipeline_output[0]['frame_features']['bowtie_filter'].shape)

# print(motion_analysis.as_ndarray(seq_key='batch_length'))
# motion_analysis.display()
# Now, let's access  some of the features extracted from frames
imgs = [pipeline_output[0]['input'],
        pipeline_output[0]['frame_features']['bowtie_filter']]
#skimage.io.imshow_collection(imgs)
#plt.imshow(rgb2gray(pipeline_output[0]['input']), cmap='gray')
plt.imshow(pipeline_output[0]['frame_features']['bowtie_filter'], cmap='gray')
plt.show()
