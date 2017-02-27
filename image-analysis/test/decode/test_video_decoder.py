import unittest
import warnings
import skvideo.io
from decode import video_decoder as vd
import os
import numpy as np

""" Do not delete these comments.
Data for test_video.mp4 is saved in test_video_data.npy.

You do not want to write tests using an skvideo function because if it breaks
or changes in functionality, you will waste time trying to figure out whether
the issue was video_decoder.py, skvideo, or the test(s).
"""


class TestVideoDecoder(unittest.TestCase):
    def setUp(self):
        self.vid_path = 'test/test_data/test_video.mp4'
        self.correct_data = np.load('test/test_data/test_video_data.npy')
        self.nframes = self.correct_data.shape[0]
        self.width = self.correct_data.shape[2]
        self.height = self.correct_data.shape[1]
        self.nchannels = self.correct_data.shape[3]

    def test_decode_mpeg(self):
        # Ignore weird resource warnings for now.
        warnings.simplefilter('ignore')

        # Verifies dimensions: nframes x height x width, nchannels.
        def test_mpeg_dimensions(self, message='', *,
                                 nframes=self.nframes,
                                 height=self.height,
                                 width=self.width,
                                 nchannels=self.nchannels):
            if vd_batch is None:
                vd_nframes = self.nframes
                vd_height = self.height
                vd_width = self.width
                vd_nchannels = self.nchannels
            else:
                vd_nframes = vd_batch.shape[0]
                vd_height = vd_batch.shape[1]
                vd_width = vd_batch.shape[2]
                vd_nchannels = vd_batch.shape[3]

            self.assertEqual(nframes, vd_nframes, message +
                             '\nnframes = {0}, vd_nframes = {1}'.format(
                              nframes, vd_nframes))
            self.assertEqual(height, vd_height, message +
                             '\nheight = {0}, vd_height = {1}'.format(
                              height, vd_height))
            self.assertEqual(width, vd_width, message +
                             '\nwidth = {0}, vd_width = {1}'.format(
                              width, vd_width))
            self.assertEqual(nchannels, vd_nchannels, message +
                             '\nnchannels = {0}, vd_channels = {1}'.format(
                              nchannels, vd_nchannels))

        # Verifies that the frames are the same.
        def test_mpeg_frame(self, message='', *, frame=0, vd_frame=0):
            self.assertEqual(np.array_equal(frame, vd_frame), True, message +
                             '\nDecoded frame does not match.')

        # Checks for the number of correct batches..
        def test_mpeg_nbatches(self, message='', *,
                               nbatches=0, vd_nbatches=0):
            self.assertEqual(nbatches, vd_nbatches, message +
                             '\nnbatches = {0}, vd_nbatches = {1}'.format(
                                 nbatches, vd_nbatches)

        # Test default decode_mpeg settings.
        numframes = 1
        numbatches = self.nframes
        batch_list = vd.decode_mpeg(self.vid_path)
 
        test_mpeg_nbatches('Default settings test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))
        
        batch = batch_list[0]
        prompt = 'Default settings test: check first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[0])
        
        batch = batch_list[-1]
        prompt = 'Default settings test: check last frame'
        test_mpeg_dimensions(prompt, nframes=numframes,  vd_batch=batch)
        test_mpeg_frame(prompt,  vd_frame=batch[0],
                        frame=self.correct_data[-1])

        # Test indices, with each frame as a separate batch: end > start.
        start = 13
        end = 88
        numbatches = end - start + 1
        numframes = 1
        batch_list = vd.decode_mpeg(self.vid_path, start_idx=start,
                                    end_idx=end)
        
        test_mpeg_nbatches('end > start test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'end > start test: check first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame = self.correct_data[start])

        batch = batch_list[-1]
        prompt = 'end > start test: check last frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame = self.correct_data[end])
        
        # Test each frame as a separate batch, using indices: start > end.
            # should raise valueerror

        # Test each frame as a separate batch, using indices start = end != 1.
        start = end = 14
        numbatches = 1
        numframes = 1
        
        test_mpeg_nbatches('end == start test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'end == start test: check frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])

        # Test batch_size = stride, with padding on.
        start = 1
        end = 99
        numframes = b_stride = 10
        numbatches = (end - start + 1) // numframes +
                      ((end - start + 1) % numframes > 0)
        batch_list = vd.decode_mpeg(self.vid_path, 
                                    start_idx=start, end_idx=end,
                                    batch_size=numframes, stride=b_stride)
        
        test_mpeg_nbatches('batch_size = stride test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))
        
        batch = batch_list[0]
        prompt = 'batch_size = stride test: check first batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0], 
                        frame=self.correct_data[start])
        prompt = 'batch_size = stride test: check first batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + numframes - 1])

        batch = batch_list[-1]
        prompt = 'batch_size = stride test: check last batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0], 
                        frame=self.correct_data[start + 
                            numframes * (end - start + 1 // numframes)])
        prompt = 'batch_size = stride test: check last batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=np.zeros(self.correct_data[0].shape))
        # Test batch_size = stride, with padding off.

        # Test batch_size > stride, with padding on.
        start = 0
        end = 100
        numframes = 15
        b_stride = 10
        numbatches = ((end - start + 1) // b_stride) +
                      (((end - start + 1) % b_stride) >(numframes - b_stride))
        batch_list = vd.decode_mpeg(self.vid_path, 
                                    start_idx=start, end_idx=end,
                                    batch_size = numframes, stride = b_stride)
        
        test_mpeg_nbatches('batch_size > stride test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'batch_size > stride test: check first batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0], 
                        frame=self.correct_data[start])
        prompt = 'batch_size > stride test: check first batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + numframes - 1])

        batch = batch_list[1]
        prompt = 'batch_size > stride test: check second batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame = self.correct_data[start + b_stride])
        prompt = 'batch_size > stride test: check second batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + b_stride +
                                                  numframes - 1])

        batch = batch_list[-1]
        prompt = 'batch_size>stride test: check last batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                frame=self.correct_data[start + 
                    b_stride * (numbatches -
                     (((end - start + 1) % b_stride) < (
                      numframes - b_stride))))])
        # TODO: add mini test to check last "real" frame in padded batch
        prompt = 'batch_size > stride test: check last batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=np.zeros(self.correct_data[0].shape))

        # Test batch_size > stride, with padding off.

        # Test batch_size < stride, with padding on.

        # Test batch_size < stride, with padding off.
####################
        batch_size = stride = 13
        expected_batches = (self.nframes // stride)
        video_batches = vd.decode_mpeg(self.vid_path,
                                       batch_size=batch_size, stride=stride)
        self.assertEqual(len(video_batches), expected_batches,
                         'len(video_batches) was not {0}'.format(
                             expected_batches))
        self.assertEqual(video_batches[0].shape[0], batch_size,
                         'batch did not contain {0} frames'.format(batch_size))

        # Load video with batch_size > stride with padding
        batch_size = 20
        stride = 10
        expected_batches = 10
        video_batches = vd.decode_mpeg(self.vid_path, batch_size=batch_size,
                                       stride=stride, end_idx=101)
        self.assertEqual(len(video_batches), expected_batches,
                         'length of video_batches is not {0}'.format(
                             expected_batches))

        last_frame = video_batches[-1][-1]['input']['frame']
        self.assertEqual(last_frame[-1, -1, -1], 0,
                         'last element in last batch was not 0. not padded \
                           properly')

        # load video with batch_size > stride without padding
        # TODO if the batch size is always greater than the stride there will
        # always be padding
        # expected_batches = 99 // stride
        # video_batches = vd.decode_mpeg(self.vid_path,
        #                                batch_size=batch_size, stride=stride,
        #                                end_idx=99)
        # self.assertEqual(len(video_batches), expected_batches,
        #                  'length of batch_list is not 111')

        # load video with stride > batch_size with padding
        batch_size = 10
        stride = 11
        expected_batches = 15
        video_batches = vd.decode_mpeg(self.vid_path, batch_size=batch_size,
                                       stride=stride)
        self.assertEqual(len(video_batches), expected_batches,
                         'total number of batches was not equal to' +
                          '{0}'.format(expected_batches))

        last_frame = video_batches[-1][-1]['input']['frame']
        self.assertEqual(last_frame[-1, -1, -1], 0,
                         'last element of last batch was not 0; \
                          not padded properly')

        # Load video with stride > batch_size without padding.
        batch_size = 10
        stride = 20
        expected_batches = 5
        video_batches = vd.decode_mpeg(self.vid_path,
                                       batch_size=batch_size,
                                       stride=stride, end_idx=99)
        self.assertEqual(len(video_batches), expected_batches,
                         'total number of batches was not equal to {0}'.format(
                             expected_batches))

        del video_batches


if __name__ == '__main__':
    unittest.main()
