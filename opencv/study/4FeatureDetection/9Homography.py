import numpy as np
import cv2 as cv

base_path = 'opencv/data/'
img1 = cv.imread(base_path + 'box.png', 0)
img2 = cv.imread(base_path + 'box_in_scene.png', 0)
MIN_MATCH_COUNT = 10

# Init
orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50) 

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # returns a mask which specifies the inlier and outlier points
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # print(M, mask.shape)

    h, w = img1.shape
    # 至少需要4个点才能完成透视变换
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # print(pts)
    dst = cv.perspectiveTransform(pts,M)

    # 灰度图像上，画彩色的线都会变成白色线
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    # cv.imshow('img2', img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # 只画正常值
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv.imshow('img3', img3)
cv.waitKey(0)
cv.destroyAllWindows()