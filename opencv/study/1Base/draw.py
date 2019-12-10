import numpy as np
import cv2 as cv

'''draw a line, a circle, a rectangle, a ellipse, a polygen, a text. There are some common arguments
    - img: 画在哪张image上
    - color: 要画形状的颜色
    - tickness: 线或圆的厚度，默认为1。填-1表示填充圆
    - lineType: 线的类型
'''

img = np.zeros((512, 512, 3), np.uint8)
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv.circle(img, (447,63), 63, (0, 0, 255), -1) # 这里-1
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 270, 255, -1) # angle：椭圆按逆时针方向旋转的角度
cv.ellipse(img, (256, 70), (50, 30), 100, 0, 180, (0, 0, 255), -1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img,[pts],True,(0,255,255)) # 是否闭合

pts2 = np.array([[70, 5], [90, 30], [130, 20], [110, 10]], np.int32)
cv.polylines(img, [pts2], False, 255, 3) 

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 1, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()