import numpy as np
import cv2 as cv
import argparse
import time

'''There are two steps.
1. Taking the first frame, detecting some Shi-Tomasi corner points
2. Track those points iteratively using Lucas-Kanada optical flow
'''
def sparse_optical_flow():
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    base_path = 'opencv/data/'
    # python opencv/study/5VideoAnalysis/3OpticalFlow.py --image=test.mp4
    parser.add_argument('--image', type=str, help='path to image file',default='slow_traffic_small.mp4')
    args = parser.parse_args()
    # print(args.image)
    cap = cv.VideoCapture(base_path + args.image)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()

        if ret == True:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate optical flow, feed previous frame, previous points and next frame
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)
            cv.imshow('frame',img)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        else:
            break

'''稠密光流
'''
def dense_optical_flow():
    base_path = 'opencv/data/'
    cap = cv.VideoCapture(base_path + 'vtest.avi')
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        if ret == True:
            next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # Computes a dense optical flow using the Gunnar Farneback's algorithm.
            flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 'flow' is a 2-channels array with optical flow vectors (u,v)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1]) # 计算2D向量的大小和角度
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            cv.imshow('frame2',bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png',frame2)
                cv.imwrite('opticalhsv.png',bgr)
            prvs = next_frame
        else:
            break

if __name__ == "__main__":
    sparse_optical_flow()
    # dense_optical_flow()