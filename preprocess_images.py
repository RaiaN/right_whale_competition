#!/usr/bin/env python

import Image
import numpy as np
import os
from skimage.io import imread, imsave
from skimage import color, img_as_float
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from matplotlib import pyplot as pl
from collections import defaultdict


IMAGES_DIR = "imgs/"
OUTPUT_DIR = "filtered/"
CLUSTERS = 35

def calc_cluster_mean_color(img, clusters_mask):
    clusters_colors = [0] * CLUSTERS
    cluster_size = [0] * CLUSTERS

    for row_ind, row in enumerate(clusters_mask):
        for col_ind, cluster_id in enumerate(row):
            clusters_colors[cluster_id] += img[row_ind][col_ind]
            cluster_size[cluster_id] += 1

    for cluster_id in range(CLUSTERS):
        if cluster_size[cluster_id] > 0: 
            clusters_colors[cluster_id] /= cluster_size[cluster_id]

    color_diff = []
    for color1 in clusters_colors:
        for color2 in clusters_colors:
            if color1 != color2:
                color_diff.append(abs(color1 - color2))

    return clusters_colors, np.median(color_diff) + np.std(color_diff)


def build_clusters_adjacency_map(clusters_mask):
    amap = defaultdict(set)
    boundaries = find_boundaries(clusters_mask)  
    row_count, col_count = clusters_mask.shape

    for row_ind, row in enumerate(boundaries):
        for col_ind, val in enumerate(row):
            if val:
                curr_id = clusters_mask[row_ind][col_ind]
                if row_ind-1 >= 0:
                    up_id = clusters_mask[row_ind-1][col_ind]
                    if up_id != curr_id:
                        amap[curr_id].add(up_id)
                        amap[up_id].add(curr_id) 
                if col_ind-1 >= 0:
                    left_id = clusters_mask[row_ind][col_ind-1]
                    if left_id != curr_id:
                        amap[curr_id].add(left_id)
                        amap[left_id].add(curr_id)
                if row_ind+1 < row_count:
                    down_id = clusters_mask[row_ind+1][col_ind]
                    if down_id != curr_id:
                        amap[curr_id].add(down_id)
                        amap[down_id].add(curr_id) 
                if col_ind+1 < col_count:
                    right_id = clusters_mask[row_ind][col_ind+1]
                    if right_id != curr_id:
                        amap[curr_id].add(right_id)
                        amap[right_id].add(curr_id) 
                    
    return amap


def significant_diff(x, y, THRESHOLD):
    return abs(x - y) > THRESHOLD


def merge_clusters(amap, clusters_mask, clusters_colors, THRESHOLD):
    TL = clusters_mask[0][0]   
    TR = clusters_mask[0][-1]
    BL = clusters_mask[-1][0]
    BR = clusters_mask[-1][-1]  
    meta_clusters = []

    start_clusters = [TL, TR, BL, BR]
    while start_clusters:
        meta_cluster = set()
        start_cluster = start_clusters.pop(0)  
        processing = [start_cluster]
        visited = set()        

        while processing:
            curr_cluster = processing.pop(0)
            visited.add(curr_cluster)

            if not meta_cluster or not significant_diff( clusters_colors[start_cluster], clusters_colors[curr_cluster], THRESHOLD):
                meta_cluster.add(curr_cluster)                

            adj_clusters = amap[curr_cluster]
            for adj_cluster in adj_clusters:
                if adj_cluster in visited:
                    continue
                processing.append(adj_cluster)

        found = False
        for meta_cluster_prev in meta_clusters:
            if not meta_cluster_prev.isdisjoint(meta_cluster):
                found = True
                meta_cluster_prev.union(meta_cluster)
        if not found:
            meta_clusters.append(meta_cluster)

    return meta_clusters


def recolor_image(image, clusters_mask, meta_clusters):
    row_count, col_count = image.shape
    for meta_cluster in meta_clusters:
        for row_ind in range(row_count):
            for col_ind in range(col_count):
                if clusters_mask[row_ind][col_ind] in meta_cluster:
                    image[row_ind][col_ind] = 0.0


def main():
    for _, _, img_filenames in os.walk(IMAGES_DIR):
        for filename in img_filenames:
            output_filename = OUTPUT_DIR + filename + ".kmeans35.jpg"
            if os.path.exists(output_filename):
                continue
            
            filesize = int(os.path.getsize(IMAGES_DIR + filename) / 1024) 
            if filesize > 1000:
                print("Skipping large file %s" % filename)
                with open("big_images.txt", "a") as outp:
                    outp.write(filename + "\n")
                continue

            print("Image size: %s" % filesize)    
            
            print("\nReading the image...")
            image = color.rgb2gray(imread(IMAGES_DIR + filename))
            print("Kmeans...")
            clusters_mask = slic(image, n_segments=CLUSTERS)

            print("Calculating clusters mean color...")
            clusters_colors, THRESHOLD = calc_cluster_mean_color(image, clusters_mask)
            print("THRESHOLD for merging: %s" % THRESHOLD)

            print("Building clusters adjacency map...")        
            amap = build_clusters_adjacency_map(clusters_mask)  
            print("Merging clusters...") 
            meta_clusters = merge_clusters(amap, clusters_mask, clusters_colors, THRESHOLD)
            print("Recoloring the image...")
            recolor_image(image, clusters_mask, meta_clusters)
            print("Saving...")
            imsave(output_filename, image)



if __name__ == "__main__":
    main()