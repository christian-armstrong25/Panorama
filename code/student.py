import cv2
import numpy as np


def calculate_transform(pointsA, pointsB, mode, useRANSAC):
    '''
    This function computes a 3 x 3 transformation matrix between two sets of points.

    Input
    - pointsA: 2 x 1 matrix of points in one image (float)
    - pointsB: 2 x 1 matrix of points in two image (float)
    - mode: Either 'translation', 'rigid', 'affine', or 'homography'.
    - useRANSAC: Whether to use a robust random sample consensus transform estimator, or not. If not, a least squares solution is used instead.

    Output
    - M: A 3 x 3 matrix representing the transformation.
    '''

    # useRANSAC == False
    # Implement a least squares solution by yourself for each transform type.

    # useRANSAC == True
    # Use OpenCV functions to robustly estimate a transform even in the presence of outliers.
    #
    # For robust fitting, use a random sample consensus method (RANSAC).
    # - RANSAC estimates a transformation matrix from a _random_ subset of the correspondences pointsA and pointsB. 
    # - Under the estimated transform, RANSAC determines which pairs of corresponding points would be inliers to some threshold, 
    #   e.g., that reproject under the transform to be close to the target points. 
    # - RANSAC iteratively estimates hundreds (or thousands!) of transforms from random subsets. 
    #   The subset with the highest number of inliers is the best transform under noise, and is kept.

    # Modern OpenCV has a robust and fast RANSAC method called MAGSAC++. 
    # Rather than passing a cv2.RANSAC flag into a transform estimation functions, use flag cv2.USAC_MAGSAC.
    # 

    # TODO

    return np.array([[1,0,0],[0,1,0],[0,0,1]])


def warp_images(A, B, transform_M):
    '''
    input:
        imgA, imgB, and transform_M - the 3x3 matrix homography transforming A into B 
                        calculated from their correspondences in previous step 

    returns:        
        warped_A:      image A warped into coordinate space of image B by transform
        warped_B:      image B warped by translation if necessary to keep A in bounds

        These two images will be the same size and they should include the entirety of both images after transformation. 
        This is basically aligning them as necessary to be composited.
    '''
    
    # TODO: Step 1 - Find the bounding box of transformed/warped A in the coordinate frame of B
    #   so that we can determine the dimensions of our composited image.
    A_rect = ??? # Coordinates of the rectangle defining A
    warped_A_rect = cv2.perspectiveTransform(A_rect, transform_M) 

    # TODO: Step 2 - Calculate the translation, if any, that is needed to bring A into fully nonnegative coordinates. 
    #   If we transform A without regard to the bounds, it may get cropped. 
    translation_xy = ??? 

    # TODO: Step 3 - Calculate the width and height of the output image.
    W = ???
    H = ???

    # TODO: Create a translation transform T that translates B to account for any shift of A. This is a 2x3 affine matrix representing the translation.
    transform_T = ???

    # Update transform M with the translation needed to keep A in frame.
    transform_M = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ transform_M


    # Create the warped images
    warped_A = cv2.warpPerspective(A, transform_M, (int(W),int(H)))
    warped_B = cv2.warpAffine(B, transform_T, (int(W),int(H)))

    return warped_A, warped_B


def composite(imgA, imgB):
    '''
    Composite imgA and imgB, both of which have already been warped by warp_images
    '''
    assert(imgA.shape == imgB.shape)

    out = np.zeros(imgA.shape, dtype=np.float32)  # TODO

    return out
