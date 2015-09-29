#!/usr/bin/env python
from skimage.transform import downscale_local_mean

__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
from skimage.measure import regionprops, label
from skimage.morphology import rectangle
from skimage import color, filter, segmentation
from skimage.morphology import erosion
import numpy as np


def yen_mask(rgbImage):
    gray_image = color.rgb2gray(rgbImage)
    val = filter.threshold_yen(gray_image)
    return gray_image < val


def rag_mean_colour(img):
    labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    return color.label2rgb(labels1, img)


def region_filter_crop(rgbImage):
    if rgbImage.shape[0] > 1000 and rgbImage.shape[1] > 1000:
        rgbImage = downscale_local_mean(rgbImage, )
    eroded_mask = erosion(yen_mask(rgbImage), rectangle(20, 20))

    biggest_region = max(regionprops(label(eroded_mask)), key=lambda x: x.area)
    minr, minc, maxr, maxc = biggest_region.bbox
    rgbImage = rgbImage[minr:maxr, minc:maxc, :]
    return rgbImage


def gray_and_downscale(rgbImage):
    return downscale_local_mean(color.rgb2gray(rgbImage), (256, 256))


def region_crop_gray_downscale(rgmImage):
    return gray_and_downscale(region_filter_crop(rgmImage))