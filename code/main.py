'''
Project 5: Panorama - main.py
CSCI 1290 Computational Photography, Brown U.
Converted to Python by Megan Gessner.

Usage-

To run on all data:
    
    python main.py

To run a single source data dir:

    python main.py -s <source_file_dir>

    e.g. python main.py -s /data/source001

'''

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from correspondences import find_correspondences
from exifextract import (extractEXIFData, getFocalLengthInPixels,
                         getSensorWidthMM)
from student import calculate_transform, composite, warp_images

SOURCE_PATH = '../data'
OUTPUT_PATH = '../results'

#
# Roy Shilkrot (Stony Brook) - numpy / cv2 fast cylindrical warp - thank you
# https://www.morethantechnical.com/blog/2018/10/30/cylindrical-image-warping-for-panorama-stitching/
#


def cylindricalWarp(img, focalLengthPixels):
    """This function returns the cylindrical warp for a given image and focal length in pixels"""
    h_, w_ = img.shape[:2]

    # Build intrinsic matrix K given focal length in pixels
    # Assume principal point is in the middle of the frame (not necessarily true)
    K = np.array([[focalLengthPixels, 0, w_/2],
                 [0, focalLengthPixels, h_/2], [0, 0, 1]])

    # create a set of pixel coordinates to warp
    y_i, x_i = np.indices((h_, w_))

    # Convert to homogeneous coordinates
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_*w_, 3)
    # Turn into set of rays
    Kinv = np.linalg.inv(K)
    # Normalized coords
    X = Kinv.dot(X.T).T

    # Calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])],
                 axis=-1).reshape(w_*h_, 3)
    # Project back to image-pixels plane
    B = K.dot(A.T).T
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]

    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    # for transparent borders...
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT, dst=np.zeros(img_rgba.shape, dtype=np.uint8))


def homographyPanorama(source_files):

    # Open the first image
    A = cv2.imread(source_files[0])
    A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

    # iterate through the rest of the images
    for source_file in source_files[1:]:

        B = cv2.imread(source_file)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

        pointsA, pointsB = find_correspondences(A, B)
        #visualize_correspondences(A, B, pointsA, pointsB) # step 1.2
        M = calculate_transform(pointsA, pointsB, 'affine', False)

        warpedA, warpedB = warp_images(A, B, M)
        stitched = composite(warpedA, warpedB)
        A = stitched

    return stitched


def visualize_correspondences(imageA, imageB, pointsA, pointsB):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(imageA)
    ax[0].scatter([p[0] for p in pointsA], [p[1] for p in pointsA], color='r')

    ax[1].imshow(imageB)
    ax[1].scatter([p[0] for p in pointsB], [p[1] for p in pointsB], color='r')

    for (xA, yA), (xB, yB) in zip(pointsA, pointsB):
        plt.plot([xA, xB + imageA.shape[1]], [yA, yB], color='yellow')

    plt.show()


def cylindricalPanorama(source_files):

    # Open the first image
    A = cv2.imread(source_files[0])
    A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

    _, cameramodel, focalLengthMM, _ = extractEXIFData(source_files[0])
    sensorWidthMM = getSensorWidthMM(cameramodel)
    focalLengthPixels = getFocalLengthInPixels(focalLengthMM, A.shape[1], sensorWidthMM)

    if focalLengthPixels is None:
        return None 

    A = cylindricalWarp(A, focalLengthPixels)
    plt.imshow(A / 255.0), plt.show()

    # iterate through the rest of the images
    for source_file in source_files[1:]:

        B = cv2.imread(source_file)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
        B = cylindricalWarp(B, focalLengthPixels)

        pointsA, pointsB = find_correspondences(A, B)
        M = calculate_transform(pointsA, pointsB, 'homography', True)

        warpedA, warpedB = warp_images(A, B, M)
        stitched = composite(warpedA, warpedB)
        A = stitched

    return stitched


if __name__ == '__main__':
    # list source directories
    source_dirs = [os.path.join(
        SOURCE_PATH, f'source00{i+1}') for i in range(len(os.listdir(SOURCE_PATH)))]

    for sd, source_dir in enumerate(source_dirs):
        source_files = [os.path.join(
            source_dir, f'{source_file}') for source_file in sorted(os.listdir(source_dir))]

        mode = 'homography'  # mode cylindrical

        if mode == 'homography':
            stitched = homographyPanorama(source_files)
        elif mode == 'cylindrical':
            stitched = cylindricalPanorama(source_files)

        if stitched is not None:
            plt.imshow(stitched / 255.0), plt.show()
            stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'panorama_' +
                        str(sd)+'_'+mode+'.png'), stitched)
