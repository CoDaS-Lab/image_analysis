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
vidwidth = 640
bowtie = OrientationFilter('bowtie', 90, 42, vidwidth // 2, .2, vidwidth,
                           'triangle')
pipe = Pipeline(ops=[bowtie], save_all=True)


def run():
    # center window on screen
    # it was showing up in weird places on mine (Anderson)
    # remove if you need to
    # cv2.namedWindow('Real-Time')
    # cv2.moveWindow('Real-Time', (1920 // 2) - imgsz, (1080 // 2) - imgsz)\
    lasttime = time.time()
    nframes = 0
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
run()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
