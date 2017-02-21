import os, sys
# to find modules
sys.path.append(os.getcwd() + "/../")

import image_analysis as codas
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

""" Steps to create a basic pipeline """

# Step 1: load 9 frames
video = codas.video_dec.decode_mpeg(os.getcwd() + '/city.mp4', end_idx=9)

# Step 2: create a pipeline object with defined steps already
# pipeline with a single step to extract grayscale from images
basic_pipeline = Pipeline(steps=[
    ("grayscale", codas.pixel_transforms.Grayscale())
    # more steps can go here
])

# Step 3: call transform functions only for all pipeline steps
# you can also call fit(...) only or fit_transform(...) for both
# these functions return the transformed data
# if you are fitting data then after you get the output of pipeline
# you can call pipeline_output.predict(unseen_data)
transformed_frames = basic_pipeline.transform(list(video))

# Step 4: do what you want with the transformed data
# grab one frame to display, just for sugar on the cake this is not needed
frame = transformed_frames[0]
# display a gray frame to make sure it worked
codas.skimage.io.imshow(frame['input']['grayscale'])
plt.show()

# TODO: create a demo that uses the fit method
