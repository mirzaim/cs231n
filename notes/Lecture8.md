# Lecture 8

# CPUs vs. GPUs

We are going to compare programs that are executed on GPUs and CPUs and finally specify what type of program suits to run on GPUs.

CPU is optimized for latency. CPUs try to complete an instruction fast as much as possible. On the other hand, GPUs are optimized for throughput. Try to complete more instructions at a constant interval. GPUs have thousands of simple cores and are well suited for massively parallel applications, applications that could be divided into many simple independent problems, like matrix multiplication.

![https://upload.wikimedia.org/wikipedia/commons/c/c6/Cpu-gpu.svg](https://upload.wikimedia.org/wikipedia/commons/c/c6/Cpu-gpu.svg)

## GPU programming languages

The two most commonly used programming languages for GPU programming are CUDA and OpenCL. We give a brief comparison between the two below.

### CUDA

just for Nvidia GPUs

has extensive libraries like cuBLAS, cuDNN, etc.

### OpenCL

runs on CPU, GPU, and even FPGAs

has good libraries but not as CUDA

## CPU / GPU communication

The roofline model states that the performance of the processing units can be bound by memory or computational power. To reach maximum performance, we shouldn't allow programs to become memory-bound. This problem is more serious for programs run on GPUs. In deep learning applications, loading data to GPU could become the bottleneck of performance. There is some advice to mitigate this kind of problem.

- Use SSD
- Prefetch data with multiple threads on the CPU
- Read all data to RAM if possible

![https://upload.wikimedia.org/wikipedia/commons/5/5a/Example_of_a_naive_Roofline_model.svg](https://upload.wikimedia.org/wikipedia/commons/5/5a/Example_of_a_naive_Roofline_model.svg)

# Deep learning frameworks

PyTorch and Tensorflow are widely used frameworks for deep learning. 

### deep learning framework advantages

- easily make computational graphs
- easily get gradient from computational graphs
- implemented on top of GPUs

## PyTorch

PyTorch is a deep learning framework made by Facebook. It has Dynamic nature and the code written with it is much cleaner than TensorFlow.

### PyTorch abstraction levels

- **Tensor:** like NumPy arrays but run on GPU

```python
>>> torch.zeros([2, 4], dtype=torch.int32)
tensor([[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]], dtype=torch.int32)
```

- **Variable:** is a node in computational graph and stores forward and backward pass data in itself

(deprecated I think)?

- **Module:** could be an independent NN.

```python
import torch
from torch import nn

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(in_features, out_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias
```

## Static vs Dynamic graphs

### Static graphs

Define graph ones and execute it many times

The framework could optimize the graph before executing it.

The graph could be serialized and executed without the code to build the graph.

Adding dynamic control flow to the graph is much more complicated.

### Dynamic graphs

Define new in each forward pass

The graph couldn't be optimized because of its dynamic nature.

The graph always needs the code to be executed.

Conditional and dynamic control flow could add to the graph easily.