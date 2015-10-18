__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import numpy as np
import csv
from images_reader import ImagesReader

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
        if not (image_name.startswith('w_') and image_name.endswith('.jpg')):
            continue
        assert whale_name.startswith('whale_')

        image_id = ImagesReader.get_image_id(image_name)
        whale_id = whale_name.split("_")[1]
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

    submission_file.write(",".join(["Image"] + list('whale_'+str(whale_id) for whale_id in sorted(whale_ids))))
    submission_file.write("\n")
    for image_id, whale_probs in zip(image_ids, whale_probs):
        submission_file.write(",".join([ImagesReader.get_image_name(image_id)] + list(str(prob) for prob in whale_probs)))
        submission_file.write("\n")