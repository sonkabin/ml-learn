import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = 'opencv/data/'

# 匹配单个对象
def math_one_object():
    img = cv.imread(base_path + '1.jpg')[:,:,::-1]
    img2 = img.copy()
    template = cv.imread(base_path + '1face.jpg')[:,:,::-1]
    # print(template.shape)
    w, h, n = template.shape

    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0]+w, top_left[1]+h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121),plt.imshow(res)
        # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img)
        # plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

# 匹配多个对象，使用阈值
def match_objects():
    img_bgr = cv.imread(base_path + '2.png')
    template = cv.imread(base_path + '2face.jpg')
    w, h, n = template.shape

    res = cv.matchTemplate(img_bgr, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.79
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), 255, 2)
    cv.imwrite(base_path + 'res.png', img_bgr)

if __name__ == "__main__":
    math_one_object()
    # match_objects()