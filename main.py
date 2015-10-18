#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
import sys
import utility
import numpy as np
import image_processors
from images_reader import ImagesReader
from sklearn.svm import SVC
import simple_cnn

TRAIN_FILENAME = 'train.csv'
SUBMISSION_FILENAME = 'submission.csv'
IMAGES_DIR = 'imgs'


def main():
    with open(TRAIN_FILENAME) as train_data_file:
        image_ids_whale_ids, whale_ids = utility.read_train(train_data_file)

        images_reader = ImagesReader(IMAGES_DIR)
        images_reader.pre_process(image_processors.region_crop_gray_downscale, rewrite=False, threads=1)

        all_train_images_ids = image_ids_whale_ids[:, 0]
        unique_train_images_ids = set(all_train_images_ids)

        all_images_ids = set(images_reader.image_ids)
        result_images_ids = all_images_ids.difference(unique_train_images_ids)
        train_image_id_whale_id = dict(image_ids_whale_ids)        

        print('Reading train data\n')
        x_train = np.asarray([images_reader.read_image_vector(image_id)
                              for image_id in all_train_images_ids])
        y_train = np.asarray([train_image_id_whale_id[image_id] for image_id in all_train_images_ids])

        features_cnt = len(x_train[0])
        num_targets = len(set(y_train))
        clf = simple_cnn.CNN(features_cnt, num_targets, 
                             num_epochs=10,
                             fresh_start=False,
                             dump_dir="network_weights/",
                             filename_to_dump="net.w")
        # clf = SVC(probability=True)

        print('Fitting\n')
        clf.fit(x_train, y_train)

        print('Reading test data\n')
        x_test = np.array([images_reader.read_image_vector(image_id)
                           for image_id in result_images_ids])

        print('Predicting\n')
        y_predicted = clf.predict_proba(x_test)

        print('Writing submission')

    with open(SUBMISSION_FILENAME, 'w') as submission_file:
        utility.write_submission(whale_ids, result_images_ids, y_predicted, submission_file)

    return 0

if __name__ == "__main__":
    sys.exit(main())