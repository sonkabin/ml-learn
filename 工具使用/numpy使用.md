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




3. np.mean(a, axis=None, ...)

   ```python
   a = np.mat([[1,0,3],[0,5,6],[0,0,0]]) # 由于是二维数组，故axis最多为2个
   print(np.mean(a)) # 所有数的均值，以数值表示	2.5
   print(np.mean(a,axis=0)) # 每列的均值		
   print(np.mean(a,axis=1)) # 每行的均值
   print(np.mean(a,axis=(0,1))) # 所有数的均值，放在matrix中 	[[2.5]]
   ```

4. np.var(a, axis=None,...)

   求**方差**，默认是所有数据的方差。axis=0按列，axis=1按行

5. ```python
   a = np.mat([1,2,3])
   a / 2 # [0.5, 1, 1.5]
   ```

6. np.tile(A, reps)：构造数组，重复A reps次

   ```python
   a = np.tile([1,2,3],2) # [1 2 3 1 2 3]
   a = np.tile([1,2,3],(2,1)) # [[1 2 3] [1 2 3]]
   a = np.tile([1,2,3],(2,2)) # [[1 2 3 1 2 3] [1 2 3 1 2 3]]
   ```

7. ```python
   a = np.mat([1,2,3])
   len(a) 等价于 np.shape(a)[0]
   ```

8. 

