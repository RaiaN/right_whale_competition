#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
import sys
import csv
import os
import numpy as np
import image_processors
import pipeline
from images_reader import ImagesReader
from skimage.io import imread, imsave
from sklearn.svm import SVC
from sklearn.metrics import log_loss

TRAIN_FILENAME = 'train.csv'
SUBMISSION_FILENAME = 'submission.csv'
IMAGES_DIR = 'imgs'


def processor_wrapper(input_image_file, out_image_file, image_processor):
    image = imread(input_image_file)
    out_image = image_processor(image)
    imsave(out_image_file, out_image)

def read_train(f):
    """
    Reads train data
    :param f: csv file containing rows of image number - whale type pairs
    :rtype : numpy array of image number - whale id pairs and whale types numpy array
    """
    reader = csv.reader(f)

    header = next(reader)
    assert (header[0], header[1]) == ('Image', 'whaleID')

    whale_ids = set()    
    image_ids_whale_ids = []

    for image_name, whale_name in reader:
        assert image_name.startswith('w_') and image_name.endswith('.jpg')
        assert whale_name.startswith('whale_')

        image_id = ImagesReader.get_image_id(image_name)   
        whale_id = int(whale_name.split("_")[1])        
        whale_ids.add(whale_id)

        image_ids_whale_ids.append((image_id, whale_id))       

    return np.array(image_ids_whale_ids), whale_ids


def write_submission(whale_ids, image_ids, whale_probs, submission_file):
    """
    Writes image_ids_whale_probabilities to submission_file
    :param submission_file: file to write the submission to
    :param whale_types: array of whale types to be written to csv header
    :param image_ids_whale_probabilities: array of pairs image_id - whale probabilities array
    """
    assert len(image_ids) == 6925 and len(whale_probs) == 6925
    writer = csv.writer(submission_file)
    writer.writerow(['Image'] + list(whale_ids))
    for image_id, whale_probs in zip(image_ids, whale_probs):
        writer.writerow([ImagesReader.get_image_name(image_id)] + list(whale_probs))


def main():
    with open(TRAIN_FILENAME) as train_data_file, open(SUBMISSION_FILENAME, 'wb') as submission_file:
        image_ids_whale_ids, whale_ids = read_train(train_data_file)

        images_reader = ImagesReader(IMAGES_DIR)
        images_reader.pre_process(image_processors.region_crop_gray_downscale, rewrite=False, threads=3)

        all_train_images_ids = image_ids_whale_ids[:, 0]
        unique_train_images_ids = set(all_train_images_ids)

        # TODO replace random submission to a normal one
        all_images_ids = set(images_reader.image_ids)
        result_images_ids = all_images_ids.difference(unique_train_images_ids)
        train_image_id_whale_id = dict(image_ids_whale_ids)        

        print('Reading train data\n')
        x_train = np.array([images_reader.read_image_vector(image_id) for image_id in all_train_images_ids])
        y_train = np.array([train_image_id_whale_id[image_id] for image_id in all_train_images_ids])
         
        clf = SVC(probability=True)
         
        print('Fitting\n')
        clf.fit(x_train, y_train)
 
        print('Reading data\n')
        x_test = np.array([images_reader.read_image_vector(image_id) for image_id in result_images_ids])
 
        print('Predicting\n')
        y_predicted = clf.predict_proba(x_test)

        print('Writing submission')
        write_submission(whale_ids, result_images_ids, y_predicted, submission_file)


    return 0

if __name__ == "__main__":
    sys.exit(main())