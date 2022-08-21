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

# Integration with pytorch

- ML framework contains computation graph, which can be traversed topologically  
- we can create a mapping of `computation graph` between tvm and the framework 
```python
MLPModuleHighLevel = from_fx(
    fx.symbolic_trace(mlp_model), 
    input_shapes = [(1, 784)], 
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear_op,
        torch.nn.ReLU: map_nn_relu_op,
    },
)
```

---

# GPU acceleration

- A typical GPU contains two-level hierarchy
    - Each thread is indexed by `threadIdx.x` and `blockIdx.x`
- Shared memory helps cache data commonly used across the threads within the same block
    - Encourage memory reuse during GPU optimization
![image](https://user-images.githubusercontent.com/8708551/183281854-f7bc61fa-fd60-403d-a7ce-537601db017e.png)

---

# GPU acceleration

- We can leverage the automatic program optimization framework to tune the program

```python
from tvm import meta_schedule as ms

sch_tuned = ms.tune_tir(
    mod=MyModuleMatmul,
    target="nvidia/tesla-p100",
    config=ms.TuneConfig(
      max_trials_global=64,
      num_trials_per_iter=64,
    ),
    work_dir="./tune_tmp",
    task_name="main"
)

rt_mod = tvm.build(sch_tuned.mod, target="nvidia/tesla-p100")
dev = tvm.cuda(0)
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

---

# Hardware Acceleration

- one emerging theme recently is `specialization`
- we build our solutions on generic `scalar processors`, where we can perform operations on one floating point at a time
- The `vector instructions` set such as AVX and ARM/Neon provide effective ways to speed up our programs but also bring some complexities to how we write the programs
- latest accelerators for machine learning introduced specialized units for `tensor computing`, with instructions for multi-dimensional data copy and matrix/tensor computations

---

# Hardware Acceleration

```python
with T.block("tmm-16x16"):
    T.reads(A[vi0 * 16 : vi0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16], B[vj0 * 16 : vj0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16])
    T.writes(C[vi0 * 16 : vi0 * 16 + 16, vj0 * 16 : vj0 * 16 + 16])
    ...
```

- This block reads from a 16x16 region from A and B, and writes to a 16x16 region of C. In this case the content of the block contains further details about a specific implementation of the subregion computations. 
- We call this block a tensorized block as they contain computations that span over sub-regions of tensors.

---

# Hardware Acceleration

```python
@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer[(1024, 1024), "float32"],
        B: T.Buffer[(1024, 1024), "float32"],
        C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]
```

- TensorIR provides a primitive call `blockization` to group subregions of a loop together to form a tensorized computation block

---

# Hardware Acceleration

- The remaining step is to map some of the tensorized blocks to use a specific implementation that maps to the hardware accelerated instructions. 
    - This mapping process is called `tensorization`

- To prepare for tensorization, we first register a tensor intrinsic (TensorIntrin) that contains a description of the computation and implementation

- The system will use the description to find relevant regions that match the computation, while implementation maps the computation to accelerated hardware instructions

---

# Computational Graph Optimization

- high-level transformations among computational graphs can be automated 

![image](https://user-images.githubusercontent.com/8708551/185782275-cbee15bb-4460-4b99-b08a-5aee94bff494.png)

---

- rewrite the program: traverse MyModule's AST recursively and generate a transformed AST, using `visitor pattern`
    - A Visitor class exposing methods that operate on each element
    - A dispatch function that recursively walks over the tree and calls the relevant method on Visitor

```python 
@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")

        if call.op != add_op:
            return call

        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call
        
        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call


updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
updated_fn.show()
```

---

- result rewrites gv0 to the fused operator but leaves lv0 in the code

```python 
@R.function
def main(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")) -> Tensor(None, "float32", ndim = 2):
    # block 0
    with R.dataflow():
        gv0: Tensor((3, 4), "float32") = relax.ewise_fma(x, y, y)
        R.output(gv0)
    return gv0
```

---

- The fused IRModule only contains calls into high-level operations

- To further low-level optimization and code generation, we need to translate those high-level primitive operators into corresponding TensorIR functions
    - we leverage the internal block builder in each Mutator and return the transformed value using call_te

```python
def map_dense(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.dense, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])


op_map = {
  "relax.nn.dense": map_dense,
  "relax.add": map_add,
  "relax.nn.relu": map_relu
}
```

---

# References 

- [Machine Learning Compiler](https://mlc.ai/summer22/)
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)



