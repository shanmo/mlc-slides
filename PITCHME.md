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

# References 

- [Machine Learning Compiler](https://mlc.ai/summer22/)
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)



