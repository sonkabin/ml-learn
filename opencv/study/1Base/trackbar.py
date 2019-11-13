import cv2 as cv
import numpy as np

def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)

# adjust radius
cv.createTrackbar('radius', 'image', 10, 200, nothing)
# bind mouse event
def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        radius = cv.getTrackbarPos('radius', 'image')
        r = cv.getTrackbarPos('R','image')
        g = cv.getTrackbarPos('G','image')
        b = cv.getTrackbarPos('B','image')
        cv.circle(img, (x, y), radius, (b, g, r), -1)
cv.setMouseCallback('image', draw_circle)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    # r = cv.getTrackbarPos('R','image')
    # g = cv.getTrackbarPos('G','image')
    # b = cv.getTrackbarPos('B','image')
    # s = cv.getTrackbarPos(switch,'image')
    # if s == 0:
    #     img[:] = 0
    # else:
    #     img[:] = [b,g,r]
    # img[:] = [b,g,r] 

cv.destroyAllWindows()