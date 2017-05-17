import os
import sys
sys.path.append(os.getcwd() + "/../")
import numpy as np
import warnings
import cv2
import time
import skimage
from decode import video_decoder as vd
from extract.orientation_filter import OrientationFilter
from extract.pipeline import Pipeline


cap = cv2.VideoCapture(0)
# image size
# we need to know the widths of the image in advance for the filter
vidwidth = 848
bowtie = OrientationFilter('bowtie', 90, 42, vidwidth // 2, .2, vidwidth,
                           'triangle')
pipe = Pipeline(ops=[bowtie], save_all=True)
bowtie.filter = bowtie.filter[:480, :]

lasttime = time.time()
while True:
        warnings.simplefilter("ignore")

        # Capture frame-by-frame
        ret, frame = cap.read()
        pipe.data = [[frame]]
        altframe = pipe.extract()[0]['frame_features']['bowtie_filter']
        cv2.imshow('Real-Time', altframe)
        # calculate how long we spend
        elapsed = time.time() - lasttime
        print('secs/frame: {0}'.format(elapsed))
        lasttime = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()