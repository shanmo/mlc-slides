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

# References 

- [Machine Learning Compiler](https://mlc.ai/summer22/)
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)



