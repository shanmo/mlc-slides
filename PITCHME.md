---
marp: true
title: Machine Learning Compiler
description: Machine Learning Compiler slides
theme: uncover
paginate: true
_paginate: false
---

![bg](./assets/gradient.jpg)

# <!--fit--> Machine Learning Compiler

---

# Motivation 

- Increasing need to deploy AI to end devices 
    - Privacy protection 
    - Distributed learning 

- Deploy trained NN models can be challenging 
    - variations in hardware, e.g. ARM, x86, GPU, FPGA  
    - diversities in the OS, e.g. Windows, Ubuntu, Mac 
    - different accelerators, e.g. `TVM`, Tensor-RT, XLA OpenVino

---

# ML Compilation

- transform, optimize ML execution from `development` to `deployment`
    - development form: PyTorch, TensorFlow, etc 
    - Deployment form: elements needed to execute the ML applications

- goals: 
    - Integration and dependency minimization
    - Leveraging hardware native acceleration
    - Optimization to minimize memory usage or improve execution efficiency

---

# ML Compilation

- MLC is the `programmatic transformations`
![img](https://user-images.githubusercontent.com/8708551/176990173-97f98532-6a95-48fd-906f-5e6f35808e2b.png)

![img](https://user-images.githubusercontent.com/8708551/176990178-fa9505b5-9930-44e7-8a28-622edd5880d6.png)

---

# Abstractions

- ways to represent the same system interface, e.g. `vector add`
- implementations in python and tvm 
```python 
def vector_add(a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]
```
```python
def vector_add(n):
    """TVM expression for vector add"""
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C
```

---

# <!--fit--> Tensor Program Abstraction

- `loop` and layout transformation 
- key components:
    - buffers that holds the input, output, and intermediate results
    - loop nests that drive compute iterations
    - computations statement
```python 
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], 
             B: T.Buffer[128, "float32"], 
             C: T.Buffer[128, "float32"]):
        # extra annotations for the function
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(128):
            with T.block("C"):
                # declare a data parallel iterator on spatial domain
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]
```
---

# <!--fit--> Tensor Program Transformation

- loop splitting 
```python 
# Get block by its name
block_c = sch.get_block("C")
# Get loops surronding the block
(i,) = sch.get_loops(block_c)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[None, 4, 4])
```
- loop reordering 
```
sch.reorder(i_0, i_2, i_1)
```

---

# <!--fit--> Tensor Program Transformation

- vectorize the inner loop `sch.parallel(i_0)`
```python 
for i_0 in tir.parallel(8):
    for i_2, i_1 in tir.grid(4, 4):
        with tir.block("C"):
            vi = tir.axis.spatial(128, i_0 * 16 + i_1 * 4 + i_2)
            tir.reads(A_1[vi], B_1[vi])
            tir.writes(C_1[vi])
            C_1[vi] = A_1[vi] + B_1[vi]
```

---

# <!--fit--> Tensor Program Transformation

- reorder to access rows for matrix multiplication ([image source](https://tvm.d2l.ai/chapter_cpu_schedules/matmul.html))
![img](https://user-images.githubusercontent.com/8708551/175759096-f8e260ac-f88d-4bc8-8a4a-dcdff527b560.png)
![img](https://user-images.githubusercontent.com/8708551/175759019-89d42afe-6493-4950-903d-26cdc39d215b.png)
```python
def reorder(n):
    s, (A, B, C) = default(n)
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    return s, (A, B, C)
```

---

# <!--fit--> Tensor Program Transformation

- leveraging the advantage brought by `data locality` is one of the most important performance optimization principles
![img](https://user-images.githubusercontent.com/8708551/176990088-bc85928d-8ce7-4a58-a56f-ebb339f7c1fd.png)

---

# A simple MLP 

- we can classify the image in `fashion MNIST` using a simple MLP
![image](https://user-images.githubusercontent.com/8708551/178094462-6fa8c440-d8a2-4b0d-8d19-091e02ae0ef5.png)
- computational graph 
![image](https://user-images.githubusercontent.com/8708551/178094404-7150aa5f-2de8-4e5c-8b92-d770998ecbbc.png)

---

# A simple MLP 

- in numpy 
![image](https://user-images.githubusercontent.com/8708551/178094481-40a4411b-d072-416e-9452-bf4b449d2c2a.png)

---

# A simple MLP 

- in low level numpy
![img](https://user-images.githubusercontent.com/8708551/178094522-83b1bdcf-bc2d-42f6-a17a-e5ee55b05fb7.png)

---

# A simple MLP 

- IRModule in TVMScript
![img](https://user-images.githubusercontent.com/8708551/178094563-ee02688e-1277-49e1-8451-902e9c6dd837.png)

---



---

# Probablistic programming

- we can create a `search space` of possible programs depending on the specific decisions made at each sampling step
- useful to experiment with parameters we are not certain, eg batch size of each loop 
![image](https://user-images.githubusercontent.com/8708551/179339899-52e17d1e-f390-40af-b6d5-6dd7ddf5cf95.png)

---

# Probablistic programming

- we can transform the MLP by using stochastic transforms, and replace the original model with the optimized one 
![image](https://user-images.githubusercontent.com/8708551/179341943-74402839-b408-4b5f-833f-e87211fededb.png)

---

# References 

- [Machine Learning Compiler](https://mlc.ai/summer22/)
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)



