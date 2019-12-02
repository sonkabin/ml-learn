import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def base(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50,50,450,450)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img), plt.colorbar(), plt.show()

def nothing(x):
    pass

def draw_rect(event, x, y, flags, param):
    global ix, iy, img, rect, mode
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        if not mode:
            cv.rectangle(img, (ix, iy), (x, y), 255, 1)
            rect = (ix, iy, x, y)
            mode = not mode
        else:
            # I don't know how to find the region after I drawed the shape.
            typebar = cv.getTrackbarPos('line?', 'img')
            if typebar == 0:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
            else:
                cv.line(img, (ix, iy), (x, y), (0, 0, 255), 1)

def with_trackbar():
    global rect, img
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    rect = -1

def mouse_event():
    events = [i for i in dir(cv) if 'EVENT' in i]
    print(events)


if __name__ == "__main__":
    ix, iy = -1, -1
    rect = -1
    mode = False
    base_path = 'opencv/data/'
    img = cv.imread(base_path + '2.png')
    # base(img)
    # mouse_event()
    cv.namedWindow('img')
    cv.setMouseCallback('img', draw_rect)
    switch = 'bgd?'
    cv.createTrackbar(switch, 'img', 0, 1, nothing)
    typebar = 'line?'
    cv.createTrackbar(typebar, 'img', 0, 1, nothing)


    while(1):
        cv.imshow('img',img)
        if rect != -1:
            with_trackbar()
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()