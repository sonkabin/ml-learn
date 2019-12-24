import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def denoising_colored(base_path):
    cap = cv.VideoCapture(base_path + 'vtest.avi')
    # img = cv.imread(base_path + 'butterfly.jpg')
    img = cap.read()[1]
    tmp = np.float64(img) # 暂存double格式
    noise = np.random.randn(*img.shape)*20
    print(noise)
    noisy = tmp + noise
    
    noisy = np.uint8(np.clip(noisy, 0, 255))
    dst = cv.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
    # print(img)
    cv.imshow('origin', img)
    cv.imshow('noisy', noisy)
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def denosing_multi(base_path):
    cap = cv.VideoCapture(base_path + 'vtest.avi')

    # create a list of first 5 frames
    img = [cap.read()[1] for i in range(5)]
    gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
    gray = [np.float64(i) for i in gray] # convert all to float64

    # create a noise of variance 25
    noise = np.random.randn(*gray[1].shape)*10 # randn(*(3,4)) 等价于 randn(3, 4)

    # Add this noise to images
    noisy = [i+noise for i in gray]

    # Convert back to uint8
    noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

    # Denoise 3rd frame considering all the 5 frames
    '''
    For example, imgToDenoiseIndex=2, temporalWindowSize=3, then frame-1, frame-2, frame-3 are used
    to be denoised frame-2
    h : parameter deciding filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
    templateWindowSize : should be odd. (recommended 7) 
    searchWindowSize : should be odd. (recommended 21)
    '''
    dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 10, 7, 21)

    plt.subplot(131),plt.imshow(gray[2],'gray')
    plt.subplot(132),plt.imshow(noisy[2],'gray')
    plt.subplot(133),plt.imshow(dst,'gray')
    plt.show()

if __name__ == '__main__':
    base_path = 'opencv/data/'
    denoising_colored(base_path)
    # denosing_multi(base_path)
    