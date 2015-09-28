#!/usr/bin/env python

import numpy as np
import os
from skimage.io import imread, imsave
from skimage import color
from skimage.filters.rank import median, mean, mean_bilateral
from skimage.morphology import disk


IMAGES_DIR = "imgs/"
OUTPUT_DIR = "filtered/"

def main():
    for _, _, img_filenames in os.walk(IMAGES_DIR):
        for ind, filename in enumerate(img_filenames):
            print("Reading the image...")
            image = color.rgb2gray(imread(IMAGES_DIR + filename))

            subdir = OUTPUT_DIR + str(ind) + "/"
            try:
            	os.makedirs(subdir)
            except:
            	print("%s already exists continuing" % subdir)


            print("Applying filters and storing...")
            image_filtered = mean(image, disk(5))
            imsave(subdir + filename + ".mean.jpg", image_filtered)
            image_filtered = mean_bilateral(image, disk(5))
            imsave(subdir + filename + ".mean_bilateral.jpg", image_filtered)
            image_filtered = median(image, disk(5))
            imsave(subdir + filename + ".median.jpg", image_filtered)



if __name__ == "__main__":
    main()