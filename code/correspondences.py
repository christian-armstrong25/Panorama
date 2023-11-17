import numpy as np
import cv2
import scipy

def find_keypoints_and_features(img):
    '''
    input:
        img:            input image
        block_size:     parameter to define block_size x block_size neighborhood around 
                        each pixel used in deciding whether it's a corner

    returns:        
        keypoints:      an array of xy coordinates of interest points
        features:       an array of features corresponding to each of the keypoints
    '''

    image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') # this is an alternate 8-bit conversion from the grayscale conversion that also normalizes the image. seems to work better with sift. found on stack overflow https://stackoverflow.com/questions/50298329/error-5-image-is-empty-or-has-incorrect-depth-cv-8u-in-function-cvsift

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image8bit,None)

    kp = [[point.pt[0], point.pt[1]] for point in kp]

    return kp, des


def find_correspondences(imgA, imgB):
    ''' 
    Automatically computes a set of correspondence points between two images.

    imgA:         input image A
    imgB:         input image B
    block_size:   size of the area around an interest point that we will 
                  use to create a feature vector. Default to 

    pointsA:      xy locations of the correspondence points in image A
    pointsB:      xy locations of the correspondence points in image B   
    '''

    # Step 1:   Use Harris Corner Detector to find a list of interest points in both images, 
    #   and compute features descriptors for each of those keypoints. Here, we are calculating
    #   the robust SIFT (i.e. Scale Invariant Feature Transform) descriptors, which detects corners
    #   at multiple scales.

    kp1, des1 = find_keypoints_and_features(imgA)
    kp2, des2 = find_keypoints_and_features(imgB)

    # Step 2: Find correspondences between the interest points in both images using the feature
    #   descriptors we've calculated for each of the points. 
    #
    # - Step 2a: Calculate and store the distance between feature vectors of all pairs (one from A and one from B) 
    #   of interest points. 
    #   As you may recall, there are many possible distance/similarity metrics. You're welcome to experiment 
    #   but we recommend the L2 norm, tried and true. (hint: scipy.spatial.distance.cdist)

    distances = scipy.spatial.distance.cdist(des1, des2)

    # - Step 2b: Find the best matches (pairs of points with the most similarity) that are below some error threshold. 
    #   You're aiming for some number of matches greater than MIN_NUMBER_MATCHES, otherwise you may not have enough information
    #   for later steps. Each point should only have one match, and we want to throw out any points that have no matches.

    MIN_NUMBER_MATCHES = 20
    FEATURE_THRESHOLD = 0.2

    # sort the distances. smaller is better
    sorted_distances_indices = np.argsort(distances) #returns the indices that would sort distances
    sorted_distances = np.take_along_axis(distances, sorted_distances_indices, -1)

    # the top two best matches in a row are compared
    # we want this for the ratio test https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    top_two_match_ratios = sorted_distances[:,0] / sorted_distances[:, 1]

    # sort the comparisons
    sorted_ratio_indices = np.argsort(top_two_match_ratios)
    sorted_ratios = top_two_match_ratios[sorted_ratio_indices]

    # find the indices of A's keypoints that produce the smallest match distance ratios
    bestA = sorted_ratio_indices[sorted_ratios < FEATURE_THRESHOLD]
    if len(bestA) < MIN_NUMBER_MATCHES:
        #if we don't have enough matches below the threshold, take the top MIN_NUMBER_MATCHES
        bestA = sorted_ratio_indices[:MIN_NUMBER_MATCHES]
    
    # get best B from best A
    bestB = sorted_distances_indices[bestA, 0]

    pointsA = np.array(kp1)[bestA]
    pointsB = np.array(kp2)[bestB]

    return pointsA, pointsB