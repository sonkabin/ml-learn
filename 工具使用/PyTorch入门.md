# PyTorch入门

> Any operation that mutates a tensor in-place is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

```python
x = torch.ones(5,3)
y = torch.rand(5, 3)
y.add(x) # y = y
y.add_(x) # y = x + y
```



> the size -1 is inferred from other dimensions

```python
x = torch.randn(4, 4)
y = x.view(16) # torch.Size([16])
z = x.view(8, -1) # torch.Size([8, 2])
```



> The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.
>
> All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

**Converting a Torch Tensor to a NumPy Array**

```python
a = torch.ones(5)
b = a.numpy()
a.add_(1) # then b will also be changed
```

**Converting NumPy Array to Torch Tensor**

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```





> `torch.nn` only supports mini-batches. The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample.
>
> For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.
>
> If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.



**在Windows上，**

> If running on Windows and you get a BrokenPipeError, try setting the num_worker of torch.utils.data.DataLoader() to 0.



## 技巧

参看neuralnetworkanddeeplearning的手写体识别-神经网络



