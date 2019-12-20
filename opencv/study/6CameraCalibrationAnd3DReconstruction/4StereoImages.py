import numpy as np
import cv2 as cv

def stereo_match(imgL, imgR):
    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey(0)


if __name__ == '__main__':
    base_path = 'opencv/data/'
    imgL = cv.imread(base_path + 'aloeL.jpg')
    imgR = cv.imread(base_path + 'aloeR.jpg')
    stereo_match(imgL, imgR)
    cv.destroyAllWindows()