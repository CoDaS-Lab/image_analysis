import os
import sys
sys.path.append(os.getcwd() + "/../")
import skimage.io
import numpy as np
import warnings
from matplotlib import pyplot as plt
from decode import video_decoder as vd
from extract.orientation_filter import OrientationFilter
from extract.pipeline import Pipeline
from skimage.color import rgb2gray
import cv2

cap = cv2.VideoCapture(0)

# image size
# we need to know the widths of the image in advance for the filter
imgsz = 600
bowtie = OrientationFilter('bowtie', 90, 42, imgsz, .2, imgsz, 'triangle')
pipe = Pipeline(ops=[bowtie], save_all=True)

# center window on screen
# it was showing up in weird places on mine (Anderson)
# remove if you need to
# cv2.namedWindow('Real-Time')
# cv2.moveWindow('Real-Time', (1920 // 2) - imgsz, (1080 // 2) - imgsz)

while True:
        warnings.simplefilter("ignore")
        # Capture frame-by-frame
        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]
        # resize image
        frame = skimage.transform.resize(frame, (imgsz, imgsz), mode='reflect')
        pipe.data = [[frame]]
        altframe = pipe.extract()[0]['frame_features']['bowtie_filter']
        altframe = skimage.transform.resize(altframe, (height, width), mode='reflect')
        cv2.imshow('Real-Time', altframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
