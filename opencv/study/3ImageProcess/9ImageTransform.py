import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Fourier Transform

''' In numpy
'''
def dft_in_numpy(img):
    # img = cv.imread(base_path + '2.png', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) # 将左上角的结果移动到中间
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Specturm'), plt.xticks([]), plt.yticks([])
    plt.show()
    return fshift

def idft_in_numpy(img, fshift):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0 # 60*60的窗口来移除低频
    f_ishift = np.fft.ifftshift(fshift) # 反转操作，从中心移到左上角
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back) # 高通滤波之后的图片，作用是边缘检测

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()

'''In OpenCV
'''
def dft_in_opencv(img):
    # img = cv.imread(base_path + '2.png', 0)
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return dft_shift

def idft_in_opencv(img, dft_shift):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1 # 移除高频
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

'''DFT性能优化
用padding来加速算法
'''
def dft_optimization(img):
    rows, cols = img.shape
    print(rows, cols)
    nrows = cv.getOptimalDFTSize(rows)
    ncols = cv.getOptimalDFTSize(cols)
    print(nrows, ncols)
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img


if __name__ == "__main__":
    base_path = 'opencv/data/'
    img = cv.imread(base_path + '2.png', 0)
    # fshift = dft_in_numpy(img)
    # idft_in_numpy(img, fshift)

    # dft_shift = dft_in_opencv(img)
    # idft_in_opencv(img, dft_shift)
    dft_optimization(img)
