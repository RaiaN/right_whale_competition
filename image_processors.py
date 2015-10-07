#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
from skimage.measure import regionprops, label
from skimage.morphology import erosion, rectangle
from skimage import color, filters
from skimage import transform
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
from skimage.feature import canny
import numpy as np


def yen_mask(rgbImage):
    gray_image = color.rgb2gray(rgbImage)
    val = filters.threshold_yen(gray_image)
    return gray_image <= val


def region_filter_crop(rgbImage):    
    rgbImage = rgbImage[100:-100, 100:-100]    
    rgbImage = transform.resize(rgbImage, (1000, 1000))    
    eroded_mask = erosion(yen_mask(rgbImage), rectangle(20, 20))
    
    regions = list(regionprops(label(eroded_mask)))
    if len(regions) == 0:
        return rgbImage
    biggest_region = max(regions, key=lambda x: x.area)

    minr, minc, maxr, maxc = biggest_region.bbox
    rgbImage = rgbImage[minr:maxr, minc:maxc, :]

    minr, minc, maxr, maxc = biggest_region.bbox
    rgbImage = rgbImage[minr:maxr, minc:maxc, :]
    return rgbImage


def gray_and_downscale(rgbImage):
    downscaled = transform.resize(color.rgb2gray(rgbImage), (150, 150))
    return downscaled


def region_crop_gray_downscale(rgmImage):
    return gray_and_downscale(region_filter_crop(rgmImage))

def CANNY(rgbImage):
    image = color.rgb2gray(rgbImage)

    image = image[100:-100, 100:-100]    
    image = transform.resize(image, (1000, 1000))

    edges_mask = canny(image)
    image[edges_mask] = 0.0    

    val = filters.threshold_yen(image)
    mask = image <= val

    regions = list(regionprops(label(mask)))
    if len(regions) == 0:
        return image
    biggest_region = max(regions, key=lambda x: x.area)

    minr, minc, maxr, maxc = biggest_region.bbox
    image = image[minr:maxr, minc:maxc]       

    return image
