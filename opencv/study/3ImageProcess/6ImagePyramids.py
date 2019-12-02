import cv2 as cv

base_path = './opencv/data/'
img = cv.imread(base_path + '1.jpg')
lower_reso = cv.pyrDown(img)
lower_reso2 = cv.pyrDown(lower_reso)
higer_reso = cv.pyrUp(lower_reso2)
higer_reso2 = cv.pyrUp(higer_reso)
cv.imshow('origin', img)
cv.imshow('img', higer_reso2)
cv.imshow('lower_reso2', lower_reso2)
cv.waitKey(0)
cv.destroyAllWindows()