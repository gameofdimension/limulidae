# limulidae

- 用于测试 NVIDIA GPU 和 Ascend NPU 的实际算力；
- 用于测试 NVIDIA GPU 和 Ascend NPU 的实际节点内通信带宽。

## 实测数据
 
### 算力

| 数据类型 | 加速器 | 实测算力 TFlops |
|----------|--------|------------------|
| BF16     | A800   | 286              |
| BF16     | 910B   | 328              |
| FP32     | A800   | 19               |
| FP32     | 910B   | 87               |

### 节点内带宽

| 卡数 | 加速器 | all_gather 带宽GB/s | all_reduce 带宽GB/s |
|------|--------|---------------------|---------------------|
| 2    | A800   | 230                 | 143                 |
| 2    | 910B   | 38                  | 18                  |
| 4    | A800   | 190                 | 104                 |
| 4    | 910B   | 64                  | 30                  |
| 8    | A800   | 173                 | 89                  |
| 8    | 910B   | 149                 | 72                  |


### 显存带宽

| 算子 | 加速器 | 显存带宽GB/s |
|------|------|------|
| `torch.exp` |A800|884|
| `torch.exp` |910B|642|
|`torch.nn.Sigmoid`|A800|887|
|`torch.nn.Sigmoid`|910B|640|
|$\frac{1}{1+e^{-x}}$（手写 sigmoid）|A800|176|
|$\frac{1}{1+e^{-x}}$（手写 sigmoid）|910B|128|


## 复现步骤

### 准备工作
1. 910B 安装相关 CANN(8.0.0.beta1), torch(cpu+2.4.0) 和 torch_npu(2.4.0.post2) 等。[详细参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/configandinstg/instg/insg_0001.html)；
2. 安装本项目依赖。

### 算力测试
`python bench_flops.py ${dtype}`, dtype 可取 fp32/fp16/bf16。

### 测试带宽
`torchrun --nproc-per-node=${卡数} bench_collective.py ${通信算子}` 通信算子目前支持 `all_reduce` 和 `all_gather`。

## 参考

- 算力测试：https://github.com/mag-/gpu_benchmark
- 带宽测试：https://github.com/IBM/pytorch-communication-benchmarks
