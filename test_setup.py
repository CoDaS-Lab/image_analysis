import wget
import os

test_files_path = os.getcwd() + '/image-analysis/test/test_data/'

# test files will be here whether is data, images, videos ect.
test_files = ["https://s3.amazonaws.com/testcodas/test_video.mp4"]

for file_path in test_files:
  wget.download(file_path, test_files_path)
