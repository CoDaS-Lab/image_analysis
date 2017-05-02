import unittest
import os
import wget
import time
import sys
from decode import video_decoder as vd
from extract.pipeline import Pipeline
from test.extract import test_features as features


class TestPipeline(unittest.TestCase):

    def setUp(self):
        data_dir = 'test/test_data/'
        self.vid_path = data_dir + 'test_video.mp4'
        vid_link = 'https://s3.amazonaws.com/codasimageanalysis/test_video.mp4'
        if not os.path.exists(self.vid_path):
            wget.download(vid_link, data_dir)

        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_operations_save(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            ops=[rgb2gray, maxPixel],
                            save_all=True)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 2)
            self.assertIsNotNone(frame['frame_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['frame_features'][rgb2gray.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_sequential_save(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            seq=[rgb2gray, maxPixel],
                            save_all=True)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 0)
            self.assertIsNotNone(frame['seq_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['seq_features'][rgb2gray.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_sequential_nonsave(self):

        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()
        maxPixel.save = True

        testpipe = Pipeline(data=data,
                            seq=[rgb2gray, maxPixel],
                            save_all=False)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 0)
            self.assertIsNotNone(frame['seq_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_as_ndarray(self):
        # TODO implement this function
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        frame_width = data[0][0].shape[0]
        frame_height = data[0][0].shape[1]
        n = len(data) * len(data[0])  # n = numbatches * framesperbatch

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            ops=[rgb2gray, maxPixel],
                            save_all=True)

        pipeline_ouput = testpipe.extract()
        gray_frames = testpipe.as_ndarray(frame_key=rgb2gray.key_name)
        self.assertTupleEqual(gray_frames.shape, (n, frame_width,
                                                  frame_height))

if __name__ == '__main__':
    unittest.main()
