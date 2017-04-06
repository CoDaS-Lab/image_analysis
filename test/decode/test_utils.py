import unittest
import os
import wget
import time
import numpy as np
import skimage.io
from decode import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.img_dir = 'test/test_data/testimgs/'
        self.testimg1_path = self.img_dir + 'test1.jpg'
        testimg1_link = 'https://s3.amazonaws.com/codasimageanalysis/test1.jpg'
        self.testimg2_path = self.img_dir + 'test2.jpg'
        testimg2_link = 'https://s3.amazonaws.com/codasimageanalysis/test2.jpg'

        if not os.path.exists(self.testimg1_path):
            if not os.path.exists(self.img_dir):
                os.mkdir(self.img_dir)

            wget.download(testimg1_link, self.img_dir)

        if not os.path.exists(self.testimg2_path):
            if not os.path.exists(self.img_dir):
                os.mkdir(self.img_dir)

            wget.download(testimg2_link, self.img_dir)

        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_load_mult_imgs(self):
        testimg1 = skimage.io.imread(self.testimg1_path)
        testimg2 = skimage.io.imread(self.testimg2_path)
        imgs = utils.load_mult_images(dirname=self.img_dir)

        self.assertTrue(np.array_equal(testimg1, imgs[0][0]))
        self.assertTrue(np.array_equal(testimg2, imgs[1][0]))

    def test_load_mult_imgs_batches(self):
        batchsize = 2
        imgs = utils.load_mult_images(dirname=self.img_dir,
                                      batchsize=batchsize)

        # 2 batches each with two images inside. Last batch should be padded
        self.assertEqual(len(imgs), 2)
        self.assertEqual(len(imgs[0]), batchsize)
        self.assertEqual(len(imgs[1]), batchsize)

if __name__ == '__name__':
    unittest.main()
