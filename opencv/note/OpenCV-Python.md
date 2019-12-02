# OpenCV-Python

1. 文件路径中不要出现中文

2. OpenCV是BGR模式，而pyplot是RGB模式，故用OpenCV读取图片要转换一下

   ```python
   img = cv2.imread('1.jpg')
   # method 1
   img2 = img[:,:,::-1]
   # method 2
   img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```

   接下来介绍下`[:,:,::-1]`的用法

   ```python
   a = np.array([[1,3], [3,4]]) # 2*2维矩阵
   b = a[:,::-1] # 这里逗号前的冒号是表示所有行；逗号后的冒号分别表示从第一列开始，到最后一列结束，步长为-1
   
   # 图像数据打印出来的结果为
   '''
   [[[42 45 43]
     ...
     [31 34 32]]
     
    [[31 34 32]
     ...
     [22 25 23]]
     
    [[42 42 36]
     ...
     [27 27 27]]]
   '''
   # 因此呢，[:,:,::-1]第一个冒号表示第一维，第二个冒号表示第二维，第三个和上面的解释相同
   ```

3. **img.dtype** is very important while debugging because a large number of errors in OpenCV-Python code is caused by invalid datatype.

4. cv.destroyAllWindows() 和cv.destroyWindow() 

   **cv.destroyAllWindows()**见名知意，**cv.destroyWindow() **传入window的name

5. cv.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])

   src：输入图片；M：偏移矩阵；dsize：输出图片的尺寸

6. Erosion和Dilation

   一般处理二进制图像（指的是灰度图像吗？）。

   Erosion：消除白噪点

   Dilation：消除黑噪点

   Opening is just another name of **erosion followed by dilation**

7. 等高线（contours）

   前提：1）为了更高的精度，在寻找等高线之前，应用阈值或canny边缘检测处理图像；2）对象必须是白色，背景是黑色，才能寻找等高线

8. histogram(直方图)

   [Opencv histogram](https://docs.opencv.org/master/de/db2/tutorial_py_table_of_contents_histograms.html)

9. 