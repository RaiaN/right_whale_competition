#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import cv2
import os
import numpy as np
import sys
from multiprocessing import Pool
from skimage.io import imread
from skimage.exposure import histogram
from skimage import color

PRE_PROCESSING_DIR = 'pre_processing'


class ImagesReader(object):
    """
    ImagesReader reads images from specified directory
    """
    def __init__(self, images_dir_path):
        self.dir_path = os.path.abspath(images_dir_path)
        assert os.path.exists(self.dir_path) and os.path.isdir(self.dir_path)

        for _, _, image_names in os.walk(self.dir_path):            
            self.image_ids = [self.get_image_id(image_name) 
                              for image_name in image_names
                              if image_name.startswith('w_') and image_name.endswith('.jpg')]
        self.image_ids.sort()
        self.pre_processed_with = None    

    def pre_process(self, image_processor, rewrite=False, threads=8):
        """
        Makes pre processing for every image in image dir using image_processor
        :param image_processor: function from ndarray RGB image NxMx3 to ndarray image
        """
        processor_name = image_processor.__name__
        sys.stdout.write('Pre processing images with "' + processor_name + '" processor\n')

        pre_processing_dir = self.pre_processor_dir(processor_name)
        if not os.path.exists(pre_processing_dir):
            os.makedirs(pre_processing_dir)
        
        process_pool = Pool(threads)        
        futures = []
        for idx, image_id in enumerate(self.image_ids):
            image_name = self.get_image_name(image_id)            
            out_image_file = os.path.join(pre_processing_dir, image_name)
            if os.path.exists(out_image_file) and not rewrite:
                continue
            
            input_image_file = os.path.join(self.dir_path, image_name)
            args = [input_image_file, out_image_file, image_processor]

            futures.append(process_pool.apply_async(processor_wrapper, args))

        for idx, future in enumerate(futures):
            sys.stdout.write('\rprocessed %d/%d' % (idx, len(futures)))
            future.get()

        process_pool.close()
        process_pool.join()

        sys.stdout.write('\nPre processing finished\n')
        self.pre_processed_with = processor_name

    def read_image_vector(self, image_id):
        """
        Takes image ID, reads correspoding image and returns it as 1D vector
        """
        if self.pre_processed_with is None:
            raise RuntimeError('Pre processing should be called first')


        image_name = os.path.join(self.pre_processor_dir(self.pre_processed_with), self.get_image_name(image_id))    
        if not os.path.exists(image_name):
            print("Training image %s does not exist..." % image_name)
            return np.zeros(150*150)     

        image = imread(image_name)
        # load only blue from RGB image
        if image.shape[-1] == 3:
            image = color.rgb2gray(image)
            #return np.resize(image[:,:,-1], image.shape[0]*image.shape[1])

        hist, _ = histogram(image)
        return np.asarray(hist)
        #return np.resize(image, image.shape[0]*image.shape[1])

    @staticmethod
    def get_image_name(image_id):
        return 'w_' + str(image_id) + '.jpg'

    @staticmethod
    def pre_processor_dir(processor_name):
        return os.path.join(os.path.abspath(PRE_PROCESSING_DIR), processor_name)

    @staticmethod
    def get_image_id(image_name):
        return int(image_name.split("_")[1].split(".")[0])
        