#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
import sys
import csv
import os
import numpy as np
import image_processors
from skimage.io import imread, imsave
from multiprocessing import Pool

TRAIN_FILENAME = 'train.csv'
SUBMISSION_FILENAME = 'submission.csv'
IMAGES_DIR = 'imgs'
PRE_PROCESSING_DIR = 'pre_processing'


def processor_wrapper(input_image_file, out_image_file, image_processor):
    image = imread(input_image_file)
    out_image = image_processor(image)
    imsave(out_image_file, out_image)


def image_name(image_id):
    return 'w_' + str(image_id) + '.jpg'


class ImagesReader(object):
    """
    ImagesReader reads images from specified directory
    """
    def __init__(self, images_dir_path):
        self.dir_path = os.path.abspath(images_dir_path)
        assert os.path.exists(self.dir_path) and os.path.isdir(self.dir_path)
        self.image_ids = [int(image_name[2:-4]) for image_name in os.listdir(self.dir_path)
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
            out_image_file = os.path.join(pre_processing_dir, image_name(image_id))
            if os.path.exists(out_image_file) and not rewrite:
                continue
            input_image_file = os.path.join(self.dir_path, image_name(image_id))
            args = [input_image_file, out_image_file, image_processor]
            futures.append(process_pool.apply_async(processor_wrapper, args))

        for idx, future in enumerate(futures):
            sys.stdout.write('\rprocessed %d/%d' % (idx, len(futures)))
            future.get()

        process_pool.close()
        process_pool.join()
        sys.stdout.write('\nPre processing finished')
        self.pre_processed_with = processor_name

    def read_image_vector(self, image_id):
        if self.pre_processed_with is None:
            raise RuntimeError('Pre processing should be called first')
        image = imread(os.path.join(self.pre_processor_dir(self.pre_processed_with), image_name(image_id)))
        return np.resize(image, image.shape[0] * image.shape[1])

    @staticmethod
    def pre_processor_dir(processor_name):
        return os.path.join(os.path.abspath(PRE_PROCESSING_DIR), processor_name)


def read_train(f):
    """
    Reads train data
    :param f: csv file containing rows of image number - whale type pairs
    :rtype : numpy array of image number - whale id pairs and whale types numpy array
    """
    reader = csv.reader(f)

    header = next(reader)
    assert (header[0], header[1]) == ('Image', 'whaleID')

    whale_types = set()
    image_ids_whale_types = []

    for row in reader:
        assert row[0].startswith('w_')
        assert row[0].endswith('.jpg')
        assert row[1].startswith('whale_')
        image_ids_whale_types.append((int(row[0][2:-4]), row[1]))
        whale_types.add(row[1])

    whale_types_list = list(whale_types)
    whale_type_to_id = dict(zip(whale_types_list, range(0, len(whale_types))))
    image_ids_whale_ids = [(image_id, whale_type_to_id[whale_type]) for image_id, whale_type in image_ids_whale_types]
    return np.array(image_ids_whale_ids), np.array(whale_types_list)


def write_submission(whale_types, image_ids_whale_probabilities, submission_file):
    """
    Writes image_ids_whale_probabilities to submission_file
    :param submission_file: file to write the submission to
    :param whale_types: array of whale types to be written to csv header
    :param image_ids_whale_probabilities: array of pairs image_id - whale probabilities array
    """
    assert len(image_ids_whale_probabilities) == 6925
    writer = csv.writer(submission_file)
    writer.writerow(['Image'] + list(whale_types))
    for image_id, whale_probabilities in image_ids_whale_probabilities:
        writer.writerow([image_name(image_id)] + list(whale_probabilities))


def main():
    with open(TRAIN_FILENAME) as train_data_file, open(SUBMISSION_FILENAME, 'wb') as submission_file:
        image_ids_whale_ids, whale_types = read_train(train_data_file)
        images_reader = ImagesReader(IMAGES_DIR)
        images_reader.pre_process(image_processors.region_crop_gray_downscale, rewrite=False, threads=3)

        # TODO replace random submission to a normal one
        train_image_ids = set(image_ids_whale_ids[:, 0])
        image_ids_whale_probabilities = [(image_id, np.random.random(len(whale_types)))
                                         for image_id in images_reader.image_ids if image_id not in train_image_ids]

        write_submission(whale_types, image_ids_whale_probabilities, submission_file)

    return 0

if __name__ == "__main__":
    sys.exit(main())
