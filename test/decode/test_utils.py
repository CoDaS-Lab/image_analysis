import unittest
import os
import numpy as np
import skimage.io
from decode import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_mult_imgs(self):
        img_dir = os.getcwd() + '/test/test_data/testimgs/'
        testimg1 = skimage.io.imread(img_dir + 'test1.jpg')
        testimg2 = skimage.io.imread(img_dir + 'test2.jpg')
        imgs = utils.load_mult_images(dirname=img_dir)

        self.assertTrue(np.array_equal(testimg1, imgs[0][0]))
        self.assertTrue(np.array_equal(testimg2, imgs[1][0]))

    def test_load_mult_imgs_batches(self):
        batchsize = 2
        img_dir = os.getcwd() + '/test/test_data/testimgs/'
        imgs = utils.load_mult_images(dirname=img_dir, batchsize=batchsize)

        # 2 batches each with two images inside. Last batch should be padded
        self.assertEqual(len(imgs), 2)
        self.assertEqual(len(imgs[0]), batchsize)
        self.assertEqual(len(imgs[1]), batchsize)

if __name__ == '__name__':
    unittest.main()
