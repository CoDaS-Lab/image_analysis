import os
import sys
sys.path.append(os.getcwd() + "/../")
import numpy as np
import warnings
import cv2
import time
import skimage
from decode import video_decoder as vd
from pipeline.orientation_filter import OrientationFilter
from pipeline.pipeline import Pipeline
from pipeline.fps import FPS


cap = cv2.VideoCapture(0)
# image size
# we need to know the widths of the image in advance for the filter
vidwidth = 640
imgshape = (480, 640)
bowtie = OrientationFilter('bowtie', 90, 42, vidwidth // 2, .2,
                           vidwidth, 'triangle')
pipe = Pipeline(ops=[bowtie], save_all=True)

fps = FPS()
fps.start()
while True:
        warnings.simplefilter("ignore")

        # Capture frame-by-frame
        ret, frame = cap.read()
        pipe.data = [[frame]]
        altframe = pipe.extract()[0]['frame_features']['bowtie_filter']
        cv2.imshow('Real-Time', altframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        fps.update()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
