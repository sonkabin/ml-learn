import numpy as np
import cv2 as cv

'''Brute-Force Matching with ORB Descriptors
'''
def bf_orb(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # because we use ORB, the distance measurement is cv.NORM_HAMMING
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,\
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow('img', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''Brute-Force Matching with SIFT Descriptors and Ratio Test
I don't use it unless I can add the third libraries.
'''
def bf_sift(img1, img2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

'''Fast Library for Approximate Nearest Neighbors
 It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. 
 It works faster than BFMatcher for large datasets.
'''
def flann_match(img1, img2):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50) # Higher values gives better precision, but also takes more time. 

    flann = cv.FlannBasedMatcher(index_params, search_params)
    # If k=2, it will draw two match-lines for each keypoint. 
    # So we have to pass a mask if we want to selectively draw it.
    # We will take k=2 so that we can apply ratio test explained by D.Lowe in his paper.
    matches = flann.knnMatch(des1, des2, k=2) 
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            print(m.distance, n.distance)
            matchesMask[i]=[1,0]
            
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv.imshow('img', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    base_path = 'opencv/data/'
    img1 = cv.imread(base_path + 'box.png', 0) # queryImage
    img2 = cv.imread(base_path + 'box_in_scene.png', 0) # trainImage

    # bf_orb(img1, img2)
    # bf_sift(img1, img2)
    flann_match(img1, img2)