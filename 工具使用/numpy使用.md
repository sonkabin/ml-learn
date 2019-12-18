# Numpy使用

1. 矩阵\*数组相当于矩阵\*矩阵

   ```python
   b = matrix([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
   a = array([[1, 1],
          [2, 2],
          [3, 3]])
   b*a=matrix([[14, 14],
           [32, 32],
           [50, 50]])
   ```

2. ```python
   matrix = matrix([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
   matrix[0:2,1:2] // 从第一行开始取2行，每行取第1个元素，作为子矩阵  out：matrix([[2],[5]])
   matrix[1,:] // 取第二行作为子矩阵  out：matrix([[4, 5, 6]])
   matrix[1,1:3] // 从第二行第二个元素开始，取2个元素，作为子矩阵	out：matrix([[5, 6]])
   
   matrix[:2,:] // 从第一行开始取2行，作为子矩阵
   ```




3. `np.mean(a, axis=None, ...)`

   ```python
   a = np.mat([[1,0,3],[0,5,6],[0,0,0]]) # 由于是二维数组，故axis最多为2个
   print(np.mean(a)) # 所有数的均值，以数值表示	2.5
   print(np.mean(a,axis=0)) # 每列的均值		
   print(np.mean(a,axis=1)) # 每行的均值
   print(np.mean(a,axis=(0,1))) # 所有数的均值，放在matrix中 	[[2.5]]
   ```

4. `np.var(a, axis=None,...)`

   求**方差**，默认是所有数据的方差。axis=0按列，axis=1按行

5. ```python
   a = np.mat([1,2,3])
   a / 2 # [0.5, 1, 1.5]
   ```

6. `np.tile(A, reps)`：构造数组，重复A reps次

   ```python
   a = np.tile([1,2,3],2) # [1 2 3 1 2 3]
   a = np.tile([1,2,3],(2,1)) # [[1 2 3] [1 2 3]]
   a = np.tile([1,2,3],(2,2)) # [[1 2 3 1 2 3] [1 2 3 1 2 3]]
   ```

7. ```python
   a = np.mat([1,2,3])
   len(a) 等价于 np.shape(a)[0]
   ```

8. `np.vstack()`：将n\*1 reshape为1\*n，type为numpy.ndarray

   ```python
   a1 = np.linspace(-1, 1, 3) # [-1, 0, 1]
   a2 = np.vstack(a1) # [[-1], [0], [1]]
   # 对于list
   b1 = [np.linspace(-1, 1, 3) for _ in range(5)] # 5个array
   b2 = np.vstack(b1) # 3*5
   ```

9. `np.random.uniform()`：返回ndarry or scalar

   ```python
   a = np.random.uniform(1, 2, size=3) # [1.14688523 1.63880608 1.75369664].  区间[1,2)
   ```

10. `np.newaxis`：NoneType

    ```python
    x = np.linspace(-1 ,1, 3)[:, np.newaxis] # 3×1矩阵
    z = np.linspace(-1 ,1, 3)[np.newaxis, :] # 1×3矩阵
    ```

11. `np.random.normal(loc=0.0, scale=1.0, size=None)`：loc均值，scale标准差，size返回的shape。高斯分布

12. `np.flatten()`与`np.ravel()`

    **相同**：都用于将多维数组降为一维数组，类型为`numpy.ndarray`，转为list类型可用`np.flatten().tolist()`

    **区别**：`np.flatten()`复制多维数组的拷贝，而`np.ravel()`是原来数组的视图。也就是说对于`np.flatten()`得到的数组进行修改不影响原数组，而对`np.ravel()`得到的数组进行修改影响原数组。

13. `...`与`:`

    ```python
    # 在2维中，两个用法效果相同
    
    # 在3维中，有点区别
    t1, t2, t3 = np.ones((1, 3, 4)), np.ones((1, 3, 4)), np.ones((1, 3, 4))
    t1[..., 1] = 255 # [[[  1. 255.   1.   1.] [  1. 255.   1.   1.] [  1. 255.   1.   1.]]]
    t2[:, 1] = 255 # [[[  1.   1.   1.   1.] [255. 255. 255. 255.] [  1.   1.   1.   1.]]]
    t3[:, :, 1] = 255 # t1 == t3
    ```

14. `np.save()`、`np.savez()`、`np.load()`

    [scipy numpy doc](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html)

15. 

