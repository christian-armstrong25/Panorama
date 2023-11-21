import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


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

    if useRANSAC:
        # RANSAC using OpenCV's functions
        if mode == 'translation':
            dx = np.mean(pointsB[:,0] - pointsA[:,0])
            dy = np.mean(pointsB[:,1] - pointsA[:,1])
            M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        elif mode == 'rigid':
            M, _ = cv2.estimateAffinePartial2D(pointsA, pointsB, method=cv2.USAC_MAGSAC)
            M = np.vstack([M, [0, 0, 1]]) 
        elif mode == 'affine':
            M, _ = cv2.estimateAffine2D(pointsA, pointsB, method=cv2.USAC_MAGSAC)
            M = np.vstack([M, [0, 0, 1]]) 
        elif mode == 'homography':
            M, _ = cv2.findHomography(pointsA, pointsB, method=cv2.RANSAC)
        else:
            raise ValueError("Invalid mode")
    else:
        # Least squares solutions
        if mode == 'translation':
            dx = np.mean(pointsB[:,0] - pointsA[:,0])
            dy = np.mean(pointsB[:,1] - pointsA[:,1])
            M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        elif mode == 'rigid':
            centroid_A = np.mean(pointsA, axis=0)
            centroid_B = np.mean(pointsB, axis=0)

            centered_A = pointsA - centroid_A
            centered_B = pointsB - centroid_B

            H = np.dot(centered_A.T, centered_B)

            U, S, Vt = np.linalg.svd(H)

            R = np.dot(Vt.T, U.T)

            if np.linalg.det(R) < 0:
                Vt[-1,:] *= -1
                R = np.dot(Vt.T, U.T)

            t = centroid_B.T - np.dot(R, centroid_A.T)

            M = np.identity(3)
            M[:2, :2] = R
            M[:2, 2] = t
        elif mode == 'affine':
            assert pointsA.shape[1] == 2 and pointsB.shape[1] == 2

            A = np.hstack([pointsA, np.ones((pointsA.shape[0], 1))])  

            x = np.linalg.lstsq(A, pointsB[:, 0], rcond=None)[0]  
            y = np.linalg.lstsq(A, pointsB[:, 1], rcond=None)[0]  

            M = np.array([[x[0], x[1], x[2]], 
                        [y[0], y[1], y[2]], 
                        [0, 0, 1]])
        elif mode == 'homography':
            if len(pointsA) < 4 or len(pointsB) < 4:
                raise ValueError("At least 4 points are required for homography estimation")
            A = []
            for i in range(len(pointsA)):
                x, y = pointsA[i][0], pointsA[i][1]
                u, v = pointsB[i][0], pointsB[i][1]
                A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
                A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

            A = np.array(A)
            U, S, Vh = np.linalg.svd(A)
            L = Vh[-1,:] / Vh[-1,-1]
            M = L.reshape(3, 3)
        else:
            raise ValueError("Invalid mode")
    
    return M


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
    
    A_rect = np.array([[0, 0], [A.shape[1], 0], [A.shape[1], A.shape[0]], [0, A.shape[0]]], dtype="float32").reshape(-1, 1, 2)
    warped_A_rect = cv2.perspectiveTransform(A_rect, transform_M)

    x_min = min(warped_A_rect[:, 0, 0])
    y_min = min(warped_A_rect[:, 0, 1])
    translation_xy = (-x_min if x_min < 0 else 0, -y_min if y_min < 0 else 0)

    x_max = max(max(warped_A_rect[:, 0, 0]), B.shape[1])
    y_max = max(max(warped_A_rect[:, 0, 1]), B.shape[0])
    W, H = x_max + translation_xy[0], y_max + translation_xy[1]

    transform_T = np.array([[1, 0, translation_xy[0]], [0, 1, translation_xy[1]]], dtype="float32")

    transform_M = np.dot(np.concatenate((transform_T, [[0, 0, 1]]), axis=0), transform_M)

    warped_A = cv2.warpPerspective(A, transform_M, (int(W), int(H)))
    warped_B = cv2.warpAffine(B, transform_T, (int(W), int(H)))

    return warped_A, warped_B


def composite(imgA, imgB):
    '''
    Composite imgA and imgB, both of which have already been warped by warp_images
    '''
    if imgA.shape[2] == 4:
        imgA = imgA[:, :, 0:3]
    imgA_uint8 = imgA.astype(np.uint8)

    if imgB.shape[2] == 4:
        imgB = imgB[:, :, 0:3]
    imgB_uint8 = imgB.astype(np.uint8)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    maskA = grayA > 0
    maskA_uint8 = maskA.astype(np.uint8)
    maskA_expanded = np.expand_dims(maskA_uint8, axis=-1)
    maskA_expanded = np.repeat(maskA_expanded, 3, axis=-1)

    imgA_uint8 = imgA.astype(np.uint8)
    imgB_uint8 = imgB.astype(np.uint8)

    return (imgA_uint8 * maskA_expanded) + (imgB_uint8 * (1 - maskA_expanded))