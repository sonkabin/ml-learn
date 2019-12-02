import numpy as np
import cv2 as cv

base_path = './opencv/data/'
img = cv.imread(base_path + '1.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print('hierarchy', hierarchy)
cv.drawContours(img, contours, -1, (0, 255, 0), 3)
# cnt = contours[3]
# cv.drawContours(img, [cnt], 0, (0, 255, 0), 3) # draw 3rd contour
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Contour Feature
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[3]
M = cv.moments(cnt)
print(M)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv.contourArea(cnt)
print(area, M['m00'])
perimeter = cv.arcLength(cnt, True)
print(perimeter)
hull = cv.convexHull(cnt, returnPoints=False) # 返回索引
hull2 = cv.convexHull(cnt) # 返回坐标
print(cnt[25], hull2[0]) # hull的第一个是25
print(cv.isContourConvex(cnt)) # check if a curve is convex or not
# Bounding Rectangle, Enclosing Circle, Ellipse, Line
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0, 0, 255), 2)
(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img, center, radius, (255, 255, 0), 2)
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img, ellipse, (0, 255, 255), 2)
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Properties
img = cv.imread(base_path + '1.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 2, 1)
cnt = contours[3]
x, y, w, h = cv.boundingRect(cnt)
aspect_ratio = float(w) / h # 长宽比
print(aspect_ratio)
hull = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s, e, d, f = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    cv.line(img, start, end, (0, 255, 255), 2)
dist = cv.pointPolygonTest(cnt,(50,50),True)
print('dist:', dist)
where = cv.pointPolygonTest(cnt,(50,50),False) # the point is inside or outside or on the contour, the speed is quicker than flag=True
print(where)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# test matchShapes()
a1 = cv.imread(base_path + 'A1.png')
a2 = cv.imread(base_path + 'A2.png')
img_gray1 = cv.cvtColor(a1, cv.COLOR_BGR2GRAY)
img_gray2 = cv.cvtColor(a2, cv.COLOR_BGR2GRAY)
diff = cv.matchShapes(img_gray1, img_gray2, cv.CONTOURS_MATCH_I1, 1.0)
print(diff)