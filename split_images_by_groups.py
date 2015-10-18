#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import os
import shutil
import utility

PREPROCESSING_DIR = "pre_processing/region_crop_gray_downscale/"
TRAIN_CSV = "train.csv"
GROUPS = "groups"

def main():
    """
        Split whale images by groups according to their IDs in train.CSV
    """
    with open(TRAIN_CSV) as inp:
        image_ids_whale_ids, _ = utility.read_train(inp)

    if not os.path.exists(GROUPS):
        try:
            os.makedirs(GROUPS)
        except:
            print("Something goes wrong. Please, check groups dir was created and you can copy a file to it!")

    for name_id, wid in image_ids_whale_ids:
        name = "w_%s.jpg" % name_id
        print(name, wid)
        target_dir = os.path.join(GROUPS, str(wid))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy(os.path.join(PREPROCESSING_DIR, name), os.path.join(target_dir, name))

if __name__ == "__main__":
    main()