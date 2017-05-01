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
        maxPixel = features.MaxPixel()

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
        maxPixel = features.MaxPixel()

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
        maxPixel = features.MaxPixel()
        maxPixel.save = True

        testpipe = Pipeline(data=data,
                            seq=[rgb2gray, maxPixel],
                            save_all=False)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 0)
            print(frame)
            self.assertIsNotNone(frame['seq_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_create_dict(self):
        fake_transformations = {'test1': 123456}
        fake_metadata = {'fake_metadata': 7890}
        testpipe = Pipeline()
        fake_dict = testpipe.create_dict(transforms=fake_transformations,
                                         metadata=fake_metadata)

        keys = list(fake_dict.keys())
        # only input and metadata in dict
        self.assertEqual(len(keys), 2)
        self.assertIsNotNone(fake_dict['input']['test1'])
        self.assertIsNotNone(fake_dict['metadata']['fake_metadata'])

    def test_data_as_nparray(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.MaxPixel()
        batchNum = features.BatchOP()

        testpipe = Pipeline(data=data,
                            save=True,
                            parallel=True,
                            operations=[rgb2gray, maxPixel, batchNum],
                            models=None)
        # extract information or transform data by calling:
        testpipe.transform()
        # get data as np arrays
        pipeline_ouput = testpipe.data_as_nparray()
        # shape should be (5, 2, 4)-> 5 batches with 2 frames each and 4
        # feature maps per frame (3 extracted features above + original frame)
        self.assertTupleEqual(pipeline_ouput.shape, (5, 2, 4))

if __name__ == '__main__':
    unittest.main()
