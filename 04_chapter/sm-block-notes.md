# SM 与 Block：资源分配、Warp 驻留与 Block 过大的处理

## 一、问题一：保证 block 能跑完，为什么又要给 SM 更多 warp？

**结论：两者不矛盾，讲的是不同层面的事。**

### 1. 整块分配：保证一个 block 能跑完

- block 以**整块**为单位分配到 SM 上：寄存器、shared memory、warp 槽位在 block 上 SM 时一次性分配好，直到整个 block 结束才释放。
- 不会出现"一半线程在 SM 上、另一半在等资源"的情况。
- 原因：`__syncthreads()` 要求 block 内所有线程都到达屏障。如果只有部分线程驻留，已到达屏障的线程会一直等没驻留的线程，而没驻留的线程又在等资源释放，就会死锁。
- 硬件规则：资源不够放下整个 block，这个 block 就不上 SM。

### 2. 多放 warp：提高驻留数量，而不是让一个 block 只拿一部分

- "给 SM 更多 warp"指的是让 SM 上同时驻留的 **block 更多**（占用率更高），每个 block 仍然整块分配。
- 例：一个 SM 最多驻留 64 个 warp，block 为 256 线程（8 个 warp），资源够的话可同时放 8 个 block，共 64 个 warp。

### 3. 为什么要驻留这么多 warp

- 驻留和执行是两回事。每个周期，warp 调度器只从驻留的 warp 中挑出就绪的几个发射指令。
- 其余 warp 可能在等全局内存、屏障或依赖，此时调度器切换到别的就绪 warp，用这种方式掩盖延迟。
- 驻留 warp 越多，越容易总能找到就绪的 warp，延迟隐藏越好。

### 4. 整块预分配让切换更便宜

- warp 的寄存器和 shared memory 在驻留期间一直留在片上，不会被换出。
- 所以 warp 切换几乎没有开销，不需要像 CPU 线程切换那样保存、恢复上下文。

### 5. 真正的取舍

- 每个 block 占的资源越多（每线程寄存器多、shared memory 大），SM 能同时驻留的 block 越少，占用率越低，延迟隐藏能力下降。
- 调优时是在"单线程资源多"和"驻留 warp 多"之间权衡，而不是资源保证与多 warp 之间有矛盾。

### 6. 补充：保证只在 block 内部

- 硬件只保证**同一个 block** 的线程同时驻留，不保证不同 block 同时驻留。
- 因此不同 block 之间不能靠互相等待来同步。
- 需要全局同步时：拆成多个 kernel，或使用 cooperative groups 的协作启动（要求整个 grid 都能同时驻留）。

---

## 二、问题二：如果一个 block 过大，SM 放不下怎么办？

**结论：这种 block 不会被启动，kernel 启动时直接报错，不会排队等待，也不会拆开跑。**

### 1. 具体表现

- 启动失败，常见错误：`cudaErrorLaunchOutOfResources`（"too many resources requested for launch"）。
- 如果不检查错误码，kernel 会静默不运行，结果可能是旧数据或 0。启动后要检查 `cudaGetLastError()`。

### 2. 两类限制

- **硬性上限**：与 block 本身有关，例如一个 block 最多 1024 个线程、shared memory 超过单 block 上限等，具体数值因架构而异。
- **由资源推出的上限**（较隐蔽）：例如大多数架构上一个 SM 有 64K 个 32 位寄存器。若每线程用 80 个寄存器，则 block 最多约 65536 / 80 ≈ 819 个线程，设成 1024 会启动失败，设成 512 才能跑。

### 3. 为什么直接拒绝而不是拆开跑

- `__syncthreads()` 要求整个 block 同时驻留；拆开跑就无法保证同步语义，所以硬件宁可拒绝。

### 4. 解决办法

| 办法 | 说明 |
| --- | --- |
| 减小 block 大小 | 最直接；可用 `cudaOccupancyMaxPotentialBlockSize` 让运行时计算合适的值 |
| 降低寄存器用量 | 用 `__launch_bounds__(maxThreadsPerBlock)` 或编译选项 `-maxrregcount`；代价是寄存器不够时会溢出到 local memory（实际在显存），可能变慢 |
| 减小 shared memory | 缩小 tile 尺寸，或用动态 shared memory；部分架构上动态 shared memory 超过默认 48KB 时需用 `cudaFuncSetAttribute` 明确申请 |
| 改算法 | 用 grid-stride loop，让每个线程处理更多数据，不需要超大 block 也能覆盖全部数据 |

### 5. 实用建议

- block 不是越大越好。通常 **128～256 线程（4～8 个 warp）** 是不错的起点：容易放下，也能让 SM 同时驻留多个 block，为延迟隐藏留出空间。

---

## 三、两个问题的关系（一句话总结）

整块分配保证"一个 block 要么完整驻留、要么不上 SM"，从而避免同步死锁；多放 block/warp 是在此前提下提高占用率以隐藏延迟；而单个 block 太大放不下时，硬件直接拒绝启动，需要你减小 block 或降低它的资源占用。
