# Thread Block Cluster、Wave、Processing Block 三个问题

---

# 问题一：属于同一个 thread block cluster 的 thread block 一定是 blockIdx 相邻的吗

**在 blockIdx 的多维坐标里是相邻的，但按线性化后的编号不一定相邻。**

## 一个 cluster 就是 grid 里的一块规则的小长方体

设 cluster 的维度是 (cx, cy, cz)，grid 的每一维必须能被对应的 cluster 维度整除，否则 launch 会失败。

blockIdx 为 (bx, by, bz) 的 block，属于编号 (bx/cx, by/cy, bz/cz) 的那个 cluster。所以同一个 cluster 里的 block，每一维上的 blockIdx 都是连续的，合起来就是 grid 里一块 cx×cy×cz 的矩形区域。

## 什么时候线性编号不相邻

blockIdx 线性化的方式是 `x + y * gridDim.x + ...`，x 变化最快。

- cluster 沿 x 方向，比如 (2,1,1)：一个 cluster 里是 blockIdx.x 为 0 和 1，线性编号 0 和 1，相邻。
- cluster 沿 y 方向，比如 (1,2,1)，grid 为 (4,2)：一个 cluster 里是 (0,0) 和 (0,1)，线性编号是 0 和 4，**不相邻**。
- cluster 为 (2,2,1)：包含 (0,0)、(1,0)、(0,1)、(1,1)，线性编号是 0、1、4、5。

## 几点补充

- **cluster 内部的序号**：用 `cluster.block_rank()` 取 block 在 cluster 内的序号（x 最快），用 `cluster.block_index()` 取它在 cluster 内的三维坐标，不要自己用 blockIdx 去推。
- **物理位置**：硬件保证同一个 cluster 的 block 同时驻留在同一个 GPC 内，这样才能互相访问对方的 shared memory（distributed shared memory）。但具体落在哪几个 SM，没有保证。
- **cluster 之间没有顺序保证**：不同 cluster 的调度顺序不确定，不要依赖 cluster 编号的先后。
- **硬件要求**：cluster 是 Hopper（sm_90）才有的特性，你的 RTX 3050（sm_86）上用不了。

以上是我按 CUDA Programming Guide 的描述凭记忆整理的，没有重新核对。如果你要在代码里依赖这个映射，建议直接查 Guide 的 "Thread Block Clusters" 一节确认。

---

# 问题二：number of waves 是 wave 的大小还是 wave 的数量

> It is desirable to have a large number of waves when the load across blocks is imbalanced to provide enough opportunities for the hardware to balance the load across SMs. On the other hand, in cases where the load across blocks is balanced, a small number of waves is acceptable and sometimes desirable as we will see in Chapter 6 when we discuss thread coarsening.
>  number of waves 是wave的大小还是wave的数量

**是 wave 的数量。**

一个 wave 指 GPU 上所有 SM 同时装满 block 的一轮执行。

**number of waves ≈ 总 block 数 ÷ GPU 一次能同时驻留的 block 数**

举例：RTX 3050 有 20 个 SM，假设每个 SM 能同时驻留 4 个 block，那么一个 wave 就是 80 个 block（这里的 4 只是示例，实际取决于 kernel 的寄存器和 shared memory 用量）。launch 800 个 block，就是 10 个 wave。

## 这句话为什么成立

**block 的负载不均衡时，wave 多更好。**
SM 一旦空出位置，硬件就会派下一个 block 过来。block 多、wave 多，快的 SM 做完就能接着领新活，慢 block 的影响会被摊平。如果只有一个 wave，所有 block 一次性派完，某个 SM 拿到特别重的 block，其他 SM 就只能干等它。

**block 的负载均衡时，wave 少也行，有时更好。**
这时硬件不需要靠调度来平衡负载，多 wave 带来的好处没有了，反而有额外开销：每个 block 都要启动、加载数据、做重复的索引计算。第 6 章的线程粗化就是让每个线程多干活，**总 block 数减少，wave 数随之减少**，省掉这些重复开销。

## 还要注意尾部效应

总 block 数不是一个 wave 容量的整数倍时，最后一个 wave 是不满的，一部分 SM 会空闲。wave 总数少的时候，这个浪费占比大。比如 1.1 个 wave 要跑 2 轮，利用率只有 55% 左右；100.1 个 wave 要跑 101 轮，利用率几乎是满的。这也是"wave 多更容易摊平浪费"的另一个原因。

**wave 的大小**（一轮能同时跑多少个 block）是硬件和 kernel 资源用量共同决定的，不是书里这句话在说的内容。

## 追问：一个 wave 也不一定一起执行完成，有可能第一个 wave 里的一些 block 执行完了，空出的资源去执行下一个 wave 里的 block

你说得对，我上一条把 wave 讲得太理想化了。

**wave 只是一个计数用的概念**，硬件并不按 wave 分批调度。实际调度是这样的：

- 硬件看到任何一个 SM 上有足够的资源（寄存器、shared memory、线程槽、block 槽），就立刻把队列里的下一个 block 派过去。
- 所以第一批 block 里有的先结束，空出的位置马上被后面的 block 接上，不会等整批做完。第 1 个 wave 和第 2 个 wave 在时间上是重叠的。
- "wave" 只是用 `总 block 数 ÷ 一次能同时驻留的 block 数` 算出来的一个估算值，用来帮助理解利用率。

我上一条里"所有 block 一次性派完""要跑 2 轮"这类说法，会让人以为每个 wave 之间有同步屏障，这是不对的。

### 这对前面几个结论的影响

**1. 负载不均衡时 wave 多更好，这个结论仍然成立，原因要换个说法。**
block 多，意味着有更多可以随时补位的 block，快的 SM 做完一个就能立刻领下一个，快慢自然被摊平。如果 block 总数只有一个 wave 那么多，就没有"后备"的 block 可以补位。

**2. 尾部效应也要更正。**
最后几个 block 派完之后，没有新的 block 可领，SM 逐个变空，这才是尾部浪费。我之前说的"1.1 个 wave 利用率约 55%"是按严格分批算的，实际会好一些，因为不存在 wave 之间的屏障，但最后少量 block 单独占着少数 SM 的问题依然存在。block 总数越多，这段尾巴占总时间的比例越小。

**3. 一个 wave 内的 block 完成时间也不一样。**
同一时刻驻留的 block，各自的执行时间会因为数据、分支、内存延迟而不同，所以也没有"同一批同时结束"的说法。

### 一个更准确的表述

**wave 数是总 block 数相对于 GPU 同时驻留能力的倍数**，倍数越大，动态调度越有余地平衡负载，尾部占比也越小；倍数很小（接近 1 甚至不足 1）时，容易出现部分 SM 空闲或被个别重 block 拖住。

如果想亲手验证，可以写一个 kernel，让不同 block 做不同长度的循环，用 `clock64()` 记录每个 block 的起止时间和所在的 SM（读 `%smid`），把结果画出来，就能直接看到 block 是怎么滚动补位的。需要的话我可以写这个小实验。

---

# 问题三：一个 processing block 只能放一个 warp 吗？由于一个 warp 有 32 个 threads 而且 warp 内的 thread 同一时刻执行同一条指令，那是不是要求 processing block 至少拥有 32 个 sm Processer

**不是，一个 processing block 里可以同时驻留多个 warp，执行单元也不需要凑够 32 个。**

## 1. 一个 processing block 放多少个 warp

processing block（SM 子分区）里有一个 warp scheduler，它管理的是一**批**驻留的 warp，不是一个。

以你的 sm_86 为例：一个 SM 最多驻留 1536 个线程，也就是 48 个 warp，分到 4 个子分区，每个子分区最多管 12 个 warp。每个时钟周期，scheduler 从这些 warp 里挑一个就绪的，发射它的一条指令。

驻留 warp 多是为了**隐藏延迟**：一个 warp 在等内存时，scheduler 马上切到另一个就绪的 warp，不浪费周期。这也是 occupancy 重要的原因。

## 2. 执行单元不需要 32 个

"一个 warp 内 32 个线程执行同一条指令"，指的是**一次发射一条指令，对 32 个线程生效**，不等于必须在同一个时钟周期里用 32 个执行通道同时算完。

如果执行单元的通道数少于 32，就把这个 warp 的指令分几个周期流水线地做完。例子（数字是我凭记忆写的，不同架构不同，不要当精确值）：

- **Volta/Turing 以及 A100**：每个子分区有 16 个 FP32 通道，一条 FP32 指令的 warp 要 2 个周期才能全部算完。
- **GA10x（RTX 30 系列，包括你的 3050）**：每个子分区的 FP32 吞吐翻倍，有两路数据通路，其中一路可以做 FP32 或 INT32，所以 FP32 密集时一个子分区每周期约能完成 32 个 FP32 运算。这就是 RTX 30 系列纸面 FP32 算力翻倍的来源。
- **SFU**：每个子分区只有 4 个左右，一条超越函数指令的 warp 要 8 个周期左右。
- **FP64**：消费级卡上很少，所以速度只有 FP32 的几十分之一。
- **LD/ST 单元**：同样是少于 32 个通道，分多个周期处理。

所以不同类型的指令，一个 warp 占用执行单元的时间长度不同，scheduler 靠不断切换不同的 warp 来填满各类单元。

## 3. 一个常见的误解

"同一时刻执行同一条指令"是**逻辑模型**（SIMT）。物理上 warp 是按"发射一条指令、分若干周期执行"来做的。从 Volta 开始还引入了独立线程调度，同一个 warp 内分叉的线程各自有程序计数器，不再严格同步，但仍然是同一时刻只发射一个路径的指令，所以性能上分支分叉仍然有代价。

## 4. 和 Tensor Core 的类比

Tensor Core 的 `mma` 指令是 **warp 级**的：32 个线程共同提供矩阵片段，由一个 Tensor Core 完成一块矩阵乘加，这时候"32 个线程一起干一件事"才是真正的物理含义。
