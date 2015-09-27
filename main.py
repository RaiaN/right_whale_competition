#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
import sys
import csv
import numpy as np

TRAIN_FILENAME = 'train.csv'
SUBMISSION_FILENAME = 'submission.csv'


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


def main():
    with open(TRAIN_FILENAME) as train_data_file:
        image_ids_whale_ids, whale_types = read_train(train_data_file)

    return 0

if __name__ == "__main__":
    sys.exit(main())
