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
THRESHOLD = 0.05

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


def significant_diff(x, y, cluster_color):
    return cluster_color[x] - cluster_color[y] > THRESHOLD 


def merge_clusters(amap, clusters_mask, cluster_color):
    TL = clusters_mask[0][0]   
    TR = clusters_mask[0][-1]
    BL = clusters_mask[-1][0]
    BR = clusters_mask[-1][-1]  
    meta_clusters = []

    start_clusters = [TL, TR, BL, BR]
    while len(start_clusters) > 0:
        meta_cluster = [] 
        start_cluster = start_clusters.pop(0)  
        processing = [start_cluster]

        while len(processing) > 0:
            curr_cluster = processing.pop(0)
            if not large_cluster or not significant_diff(start_cluster, curr_cluster, cluster_color):
                meta_cluster.append(curr_cluster)

            adj_clusters = amap[start]
            for adj_cluster in adj_clusters:
                processing.append(adj_cluster)

        meta_clusters.append(meta_cluster)

    return meta_clusters


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
            print("Kmeans...")
            mask = slic(image, n_segments=CLUSTERS)

            print("Calculating clusters mean color...")
            cluster_color = calc_cluster_mean_color(image, mask)
            #print(cluster_color)

            print("Building clustes adjacency map...")        
            amap, TL, TR, BL, BR = build_clusters_adjacency_map(mask)   
            large_clusters = merge_clusters(amap, mask, cluster_color)



if __name__ == "__main__":
    main()