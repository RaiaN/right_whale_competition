#!/usr/bin/env python

import Image
import numpy as np
import os
from skimage.io import imread, imsave
from skimage import color, img_as_float
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from matplotlib import pyplot as pl


IMAGES_DIR = "imgs/"
OUTPUT_DIR = "filtered/"
CLUSTERS = 35

def calc_cluster_mean_color(img, clusters_mask):
    cluster_color = [0] * CLUSTERS
    cluster_size = [0] * CLUSTERS

    for row_ind, row in enumerate(clusters_mask):
        for col_ind, cluster_id in enumerate(row):
            cluster_color[cluster_id] += img[row_ind][col_ind]
            cluster_size[cluster_id] += 1

    for cluster_id in range(CLUSTERS):
        if cluster_size[cluster_id] > 0: 
            cluster_color[cluster_id] /= cluster_size[cluster_id]

    return cluster_color


def main():
    for _, _, img_filenames in os.walk(IMAGES_DIR):
        for filename in img_filenames:
            output_filename = OUTPUT_DIR + filename + ".kmeans40.jpg"
            if os.path.exists(output_filename):
                continue
            
            filesize = int(os.path.getsize(IMAGES_DIR + filename) / 1024)
            print("Image size: %s" % filesize)

            if filesize > 1000:
                with open("big_images.txt", "a") as outp:
                    outp.write(filename + "\n")
                continue
            print("\nReading the image...")

            image = color.rgb2gray(imread(IMAGES_DIR + filename))   
            print(image[0][0])         
            print("Kmeans...")
            mask = slic(image, n_segments=CLUSTERS)

            #boundaries = find_boundaries(mask)            

            print("Calculating clusters mean color...")
            cluster_color = calc_cluster_mean_color(image, mask)
            print(cluster_color)



if __name__ == "__main__":
    main()