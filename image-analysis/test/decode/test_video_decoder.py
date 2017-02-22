import unittest, warnings
import skvideo.io
from decode import video_decoder as vd


class TestVideoDecoder(unittest.TestCase):
    def setUp(self):
        self.video_path = 'test/test_data/video.mp4'
        metadata = skvideo.io.ffprobe(self.video_path)
        self.total_frames = int(metadata['video']['@nb_frames'])
        self.frames_width = int(metadata['video']['@width'])
        self.frames_height = int(metadata['video']['@height'])
        self.frames_channels = 3

    def test_mpeg_dimensions(self):
        video = vd.decode_mpeg(self.video_path, end_idx=0)
        frame = video[0]

        self.assertEqual(len(video), 1)  # only one frame in batch
        self.assertEqual(frame['input']['frame'].shape[0], self.frames_height)
        self.assertEqual(frame['input']['frame'].shape[1], self.frames_width)
        self.assertEqual(frame['input']['frame'].shape[2],
                         self.frames_channels)

    def test_decode_mpeg(self):
        # Ignore weird resource warnings for now
        warnings.simplefilter('ignore')

        # Load video file in batches = strides of 13 frames
        batch_size = stride = 13
        expected_batches = (self.total_frames // stride)
        video_batches = vd.decode_mpeg(self.video_path,
                                       batch_size=batch_size, stride=stride)
        self.assertEqual(len(video_batches), expected_batches,
                         'len(video_batches) was not {0}'.format(
                             expected_batches))
        self.assertEqual(video_batches[0].shape[0], batch_size,
                         'batch did not contain {0} frames'.format(batch_size))

        # load video with batch_size > stride with padding
        batch_size = 20
        stride = 10
        expected_batches = 10
        video_batches = vd.decode_mpeg(self.video_path, batch_size=batch_size,
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
        # video_batches = vd.decode_mpeg(self.video_path,
        #                                batch_size=batch_size, stride=stride,
        #                                end_idx=99)
        # self.assertEqual(len(video_batches), expected_batches,
        #                  'length of batch_list is not 111')

        # load video with stride > batch_size with padding
        batch_size = 10
        stride = 11
        expected_batches = 15
        video_batches = vd.decode_mpeg(self.video_path, batch_size=batch_size,
                                       stride=stride)
        self.assertEqual(len(video_batches), expected_batches,
                         'total number of batches was not equal to {0}'.format(
                             expected_batches))

        last_frame = video_batches[-1][-1]['input']['frame']
        self.assertEqual(last_frame[-1, -1, -1], 0,
                         'last element of last batch was not 0; \
                          not padded properly')

        # load video with stride > batch_size without padding
        batch_size = 10
        stride = 20
        expected_batches = 5
        video_batches = vd.decode_mpeg(self.video_path,
                                       batch_size=batch_size,
                                       stride=stride, end_idx=99)
        self.assertEqual(len(video_batches), expected_batches,
                         'total number of batches was not equal to {0}'.format(
                             expected_batches))

        del video_batches


if __name__ == '__main__':
    unittest.main()