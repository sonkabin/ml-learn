# -*- coding: utf-8 -*-  
import cv2
import matplotlib.pyplot as plt

base_path = './opencv/data/'

img = cv2.imread(base_path + '1.jpg')
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow("image", img)
# k = cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows()
# elif k == ord('s'):
#     cv2.imwrite(base_path + 'm4a1.jpg', img)
#     cv2.destroyAllWindows()

# 使用pyplot
# img2 = img[:,:,::-1]
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.xticks([]); plt.yticks([])
plt.show()