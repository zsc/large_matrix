# 第16章：多模态推荐的张量分解

现代推荐系统面临着日益复杂的多维交互建模挑战。用户不仅与物品产生交互，还涉及时间、地点、设备类型、社交关系等多个维度。传统的矩阵分解方法在处理这些高阶交互时显得力不从心。本章探讨如何利用张量分解技术优雅地建模多模态推荐场景，重点关注可扩展性、稀疏性处理以及跨域知识迁移。我们将深入剖析CP分解和Tucker分解在十亿级数据上的工程实现，以及如何通过耦合矩阵分解实现跨域推荐。

## 16.1 高阶交互的张量建模

### 16.1.1 从矩阵到张量：维度的诅咒与祝福

在传统的协同过滤中，我们用二维矩阵 $\mathbf{R} \in \mathbb{R}^{m \times n}$ 表示用户-物品交互。然而，现实中的推荐场景往往涉及更多维度：

- **时间维度**：用户偏好随时间变化
- **上下文维度**：地点、设备、天气等环境因素
- **社交维度**：好友关系、群组归属
- **内容维度**：物品的多个属性标签

这些多维交互自然地形成了张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$。例如，一个三阶张量 $\mathcal{X} \in \mathbb{R}^{m \times n \times t}$ 可以表示"用户-物品-时间"的交互模式。

### 16.1.2 张量的基本运算与性质

**纤维（Fiber）与切片（Slice）**：
- Mode-n 纤维：固定除第n个索引外的所有索引得到的向量
- Mode-n 切片：固定第n个索引得到的子张量

**Mode-n 展开（Matricization）**：
将张量 $\mathcal{X}$ 沿第n个模式展开成矩阵 $\mathbf{X}_{(n)}$，这是许多张量算法的基础操作。展开规则：
$$\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times \prod_{k \neq n} I_k}$$

**张量与矩阵的乘积**：
Mode-n 乘积定义为：
$$\mathcal{Y} = \mathcal{X} \times_n \mathbf{U} \Leftrightarrow \mathbf{Y}_{(n)} = \mathbf{U} \mathbf{X}_{(n)}$$

### 16.1.3 推荐系统中的张量建模实例

**案例1：时序推荐**
构建三阶张量 $\mathcal{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}| \times |\mathcal{T}|}$，其中：
- $\mathcal{U}$：用户集合
- $\mathcal{I}$：物品集合  
- $\mathcal{T}$：时间片段（如小时、天、周）

张量元素 $r_{uit}$ 表示用户 $u$ 在时间段 $t$ 对物品 $i$ 的评分或隐式反馈。

**案例2：多模态内容推荐**
对于包含文本、图像、视频的多模态物品，构建四阶张量：
$$\mathcal{X} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}| \times |\mathcal{M}| \times |\mathcal{F}|}$$
其中 $\mathcal{M}$ 是模态类型，$\mathcal{F}$ 是特征维度。

### 16.1.4 稀疏性挑战与机遇

推荐系统中的张量极度稀疏，观测率通常低于 0.01%。这带来了计算和统计两方面的挑战：

**计算挑战**：
- 存储开销：稠密存储不现实
- 运算效率：大量零元素参与计算
- 内存访问：随机访问模式导致cache miss

**统计挑战**：
- 过拟合风险：参数远多于观测
- 冷启动问题：新用户/物品缺乏数据
- 负采样偏差：未观测不等于负样本

然而，稀疏性也带来机遇：
- 低秩假设更可能成立
- 可利用隐式正则化
- 计算可大幅优化

## 16.2 CP分解与Tucker分解的可扩展实现

### 16.2.1 CP分解（CANDECOMP/PARAFAC）

CP分解将张量表示为秩一张量之和：
$$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)} \circ \cdots \circ \mathbf{a}_r^{(N)}$$

其中 $\circ$ 表示外积，$\lambda_r$ 是权重，$\mathbf{a}_r^{(n)}$ 是第 $n$ 个模式的因子向量。

**矩阵形式**：
定义因子矩阵 $\mathbf{A}^{(n)} = [\mathbf{a}_1^{(n)}, \ldots, \mathbf{a}_R^{(n)}]$，则：
$$\mathbf{X}_{(n)} \approx \mathbf{A}^{(n)} \boldsymbol{\Lambda} (\mathbf{A}^{(N)} \odot \cdots \odot \mathbf{A}^{(n+1)} \odot \mathbf{A}^{(n-1)} \odot \cdots \odot \mathbf{A}^{(1)})^T$$

### 16.2.2 可扩展的ALS-CP算法

交替最小二乘（ALS）是求解CP分解的主流方法。对于第 $n$ 个模式的更新：

$$\mathbf{A}^{(n)} = \mathbf{X}_{(n)} \mathbf{Z}^{(n)} (\mathbf{V}^{(n)})^{-1}$$

其中：
- $\mathbf{Z}^{(n)} = \mathbf{A}^{(N)} \odot \cdots \odot \mathbf{A}^{(n+1)} \odot \mathbf{A}^{(n-1)} \odot \cdots \odot \mathbf{A}^{(1)}$
- $\mathbf{V}^{(n)} = \prod_{k \neq n} (\mathbf{A}^{(k)})^T \mathbf{A}^{(k)}$（逐元素乘积）

**稀疏优化技巧**：
1. **MTTKRP优化**（Matricized Tensor Times Khatri-Rao Product）：
   - 避免显式形成 Khatri-Rao 积
   - 利用稀疏张量的坐标格式（COO）
   - 复杂度从 $O(R \prod_k I_k)$ 降至 $O(R \cdot nnz(\mathcal{X}))$

2. **并行化策略**：
   - Mode-wise 并行：不同进程更新不同模式
   - Element-wise 并行：划分张量元素
   - Hybrid 并行：结合两种策略

### 16.2.3 Tucker分解

Tucker分解是CP分解的推广：
$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_N \mathbf{U}^{(N)}$$

其中 $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}$ 是核心张量，$\mathbf{U}^{(n)} \in \mathbb{R}^{I_n \times R_n}$ 是因子矩阵。

**与CP分解的关系**：
当核心张量 $\mathcal{G}$ 是超对角时，Tucker分解退化为CP分解。

### 16.2.4 高效的Tucker-ALS实现

**HOOI算法**（Higher-Order Orthogonal Iteration）：
1. 初始化因子矩阵 $\mathbf{U}^{(n)}$
2. 重复直至收敛：
   - 对每个模式 $n$：
     $$\mathbf{Y} = \mathcal{X} \times_1 (\mathbf{U}^{(1)})^T \times_2 \cdots \times_{n-1} (\mathbf{U}^{(n-1)})^T \times_{n+1} (\mathbf{U}^{(n+1)})^T \times_N \cdots \times_N (\mathbf{U}^{(N)})^T$$
     $$\mathbf{U}^{(n)} = \text{leading\_eigenvectors}(\mathbf{Y}_{(n)} \mathbf{Y}_{(n)}^T, R_n)$$
3. 计算核心张量：$\mathcal{G} = \mathcal{X} \times_1 (\mathbf{U}^{(1)})^T \times_2 \cdots \times_N (\mathbf{U}^{(N)})^T$

**内存优化**：
- 避免存储中间张量 $\mathbf{Y}$
- 使用 tensor-matrix chain multiplication
- 实现 memory-efficient HOOI (ME-HOOI)

### 16.2.5 随机化加速

**随机采样Tucker分解**：
1. 对每个模式随机采样纤维
2. 在采样的子张量上执行Tucker分解
3. 使用采样分解初始化完整分解

理论保证：在一定条件下，采样误差以高概率被控制在：
$$\|\mathcal{X} - \hat{\mathcal{X}}\|_F \leq (1 + \epsilon) \|\mathcal{X} - \mathcal{X}_k\|_F$$

其中 $\mathcal{X}_k$ 是最优的秩-$(R_1, \ldots, R_N)$ 近似。

## 16.3 稀疏张量的高效存储与计算

### 16.3.1 稀疏张量存储格式

**COO格式**（Coordinate）：
存储非零元素的坐标和值：
```
indices: [[i1, j1, k1], [i2, j2, k2], ...]
values: [v1, v2, ...]
```
优点：简单，支持高效的元素级操作
缺点：不支持高效的切片操作

**CSF格式**（Compressed Sparse Fiber）：
多层级的压缩存储，类似于矩阵的CSR格式在高维的推广。

**HiCOO格式**（Hierarchical COO）：
将张量分块，每块内使用COO格式：
- 块级别：压缩存储
- 块内部：COO格式
优点：更好的局部性，支持并行计算

### 16.3.2 稀疏MTTKRP的优化实现

MTTKRP（Matricized Tensor Times Khatri-Rao Product）是张量分解的核心操作：
$$\mathbf{Y} = \mathbf{X}_{(n)} (\mathbf{A}^{(N)} \odot \cdots \odot \mathbf{A}^{(n+1)} \odot \mathbf{A}^{(n-1)} \odot \cdots \odot \mathbf{A}^{(1)})$$

**稀疏优化算法**：
```
for each nonzero (i1, ..., iN, v) in X:
    for r = 1 to R:
        Y[in, r] += v * ∏(k≠n) A[ik, r]
```

**性能优化技巧**：
1. **向量化**：使用SIMD指令
2. **缓存优化**：
   - 重排序非零元素提高局部性
   - 使用blocking技术
3. **并行化**：
   - 原子操作处理写冲突
   - 使用私有累加器+归约

### 16.3.3 分布式稀疏张量计算

**张量分区策略**：
1. **Coarse-grained**：按模式分区
   - 每个节点负责某些fiber
   - 通信发生在模式切换时
   
2. **Fine-grained**：按非零元素分区
   - 更好的负载均衡
   - 更复杂的通信模式

3. **Hybrid**：多级分区
   - 顶层按模式分区
   - 底层按元素分区

**通信优化**：
- All-to-all 通信的优化调度
- 使用 overlap 技术隐藏通信延迟
- 压缩传输的因子矩阵

### 16.3.4 GPU加速的稀疏张量运算

**挑战**：
- 不规则的内存访问模式
- 负载不均衡
- 有限的GPU内存

**优化策略**：
1. **COO-based GPU kernel**：
   - Warp-level 并行
   - Shared memory 缓存因子矩阵
   
2. **混合精度计算**：
   - FP16 计算，FP32 累加
   - 利用Tensor Core加速

3. **多GPU并行**：
   - 模型并行：分解因子矩阵
   - 数据并行：分区张量

## 16.4 跨域推荐的耦合矩阵分解

### 16.4.1 跨域推荐的动机与挑战

现代用户在多个域（domain）中产生行为数据：
- 电商平台：购买历史
- 视频平台：观看记录
- 音乐平台：收听历史
- 社交平台：互动行为

**跨域推荐的价值**：
1. **缓解冷启动**：利用源域的丰富数据帮助目标域
2. **提升推荐质量**：挖掘跨域的互补信息
3. **统一用户画像**：构建全面的用户兴趣模型

**核心挑战**：
- 域间的语义鸿沟
- 数据分布的异质性
- 隐私和数据孤岛
- 负迁移风险

### 16.4.2 集体矩阵分解（CMF）框架

考虑 $K$ 个域的推荐问题，每个域有交互矩阵 $\mathbf{R}^{(k)} \in \mathbb{R}^{m_k \times n_k}$。CMF通过共享因子实现知识迁移：

$$\min \sum_{k=1}^{K} \alpha_k \|\mathbf{R}^{(k)} - \mathbf{U}^{(k)} (\mathbf{V}^{(k)})^T\|_F^2 + \text{regularization}$$

**共享策略**：
1. **用户侧共享**：当用户在多个域中重叠时
   $$\mathbf{U}^{(k)} = \mathbf{U}_{\text{shared}} \mathbf{B}^{(k)}$$
   其中 $\mathbf{B}^{(k)}$ 是域特定的变换矩阵

2. **物品侧共享**：当物品具有跨域属性时
   $$\mathbf{V}^{(k)} = \mathbf{V}_{\text{shared}} \mathbf{C}^{(k)}$$

3. **深度共享**：通过深度网络学习域间映射

### 16.4.3 张量耦合分解

当跨域数据具有高阶结构时，使用耦合张量分解：

**CMTF（Coupled Matrix-Tensor Factorization）**：
同时分解矩阵和张量，通过共享模式实现耦合：

$$\min \|\mathcal{X} - \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r\|_F^2 + \beta \|\mathbf{Y} - \mathbf{A}\mathbf{D}^T\|_F^2$$

其中张量 $\mathcal{X}$ 和矩阵 $\mathbf{Y}$ 共享因子矩阵 $\mathbf{A}$。

**应用实例**：
- $\mathcal{X}$：用户-物品-时间 交互张量
- $\mathbf{Y}$：用户-属性 特征矩阵
- 共享用户因子 $\mathbf{A}$ 实现特征增强

### 16.4.4 非负耦合分解

推荐系统中的数据通常非负（评分、点击次数等），非负约束带来更好的可解释性：

**非负CMTF**：
$$\min_{\mathbf{A}, \mathbf{B}, \mathbf{C} \geq 0} \|\mathcal{X} - \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r\|_F^2$$

**乘性更新规则**：
保证非负性的同时优化目标函数：
$$\mathbf{A} \leftarrow \mathbf{A} \odot \frac{[\nabla_{\mathbf{A}} f]^-}{[\nabla_{\mathbf{A}} f]^+ + \epsilon}$$

其中 $[x]^+ = \max(x, 0)$，$[x]^- = \max(-x, 0)$。

### 16.4.5 隐私保护的联邦耦合分解

跨域推荐面临严重的隐私挑战。联邦学习提供了一种解决方案：

**垂直联邦CMF**：
- 各域保留原始数据
- 仅交换加密的中间结果
- 使用安全多方计算或同态加密

**算法框架**：
1. 各域本地更新私有因子
2. 安全聚合共享因子的梯度
3. 使用差分隐私添加噪声
4. 更新全局共享因子

**隐私-效用权衡**：
$$\mathcal{L}_{\text{private}} = \mathcal{L}_{\text{original}} + \lambda \cdot \text{DP-noise}$$

选择合适的 $\lambda$ 平衡隐私保护和推荐性能。

### 16.4.6 在线耦合分解

实时跨域推荐需要增量更新算法：

**增量CMTF算法**：
当新数据 $\Delta \mathcal{X}$ 到达时：
1. 固定其他域的因子
2. 使用 Woodbury 恒等式更新共享因子
3. 局部微调域特定因子

**时间复杂度**：
从 $O(KNR^2)$ 降至 $O(|\Delta|R^2)$，其中 $|\Delta|$ 是新数据量。

## 本章小结

本章深入探讨了多模态推荐中的张量分解技术：

1. **张量建模**为多维交互提供了自然的数学框架，但稀疏性带来了计算和统计挑战

2. **CP分解和Tucker分解**各有优势：
   - CP分解：参数少，易解释
   - Tucker分解：表达能力强，灵活性高

3. **稀疏张量计算**的关键在于：
   - 选择合适的存储格式
   - 优化核心操作（如MTTKRP）
   - 利用并行和分布式计算

4. **跨域推荐**通过耦合分解实现知识迁移，但需要carefully设计共享机制并考虑隐私保护

**关键公式汇总**：
- CP分解：$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)} \circ \cdots \circ \mathbf{a}_r^{(N)}$
- Tucker分解：$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_N \mathbf{U}^{(N)}$
- MTTKRP：$\mathbf{Y} = \mathbf{X}_{(n)} (\mathbf{A}^{(N)} \odot \cdots \odot \mathbf{A}^{(1)})$
- CMF目标：$\min \sum_{k} \alpha_k \|\mathbf{R}^{(k)} - \mathbf{U}^{(k)} (\mathbf{V}^{(k)})^T\|_F^2$

## 常见陷阱与错误（Gotchas）

### 1. 秩选择的陷阱
- **错误**：选择过大的秩 $R$，期望获得更好的拟合
- **问题**：过拟合、计算开销剧增、数值不稳定
- **解决**：使用交叉验证、核心一致性诊断（Core Consistency Diagnostic）

### 2. 初始化敏感性
- **错误**：使用全零或统一随机初始化
- **问题**：陷入局部最优、收敛缓慢
- **解决**：使用SVD-based初始化、多次随机重启

### 3. 稀疏性处理不当
- **错误**：将缺失值填充为0进行稠密计算
- **问题**：引入大量偏差、内存爆炸
- **解决**：只在观测位置计算损失、使用稀疏张量格式

### 4. 负采样偏差
- **错误**：将所有未观测项视为负样本
- **问题**：严重的假阴性问题
- **解决**：加权采样、置信度建模、使用辅助信息

### 5. 数值稳定性问题
- **错误**：直接计算 Khatri-Rao 积的逆
- **问题**：条件数爆炸、数值误差累积
- **解决**：使用伪逆、添加正则化项、QR分解

### 6. 跨域负迁移
- **错误**：盲目共享所有因子
- **问题**：源域噪声污染目标域
- **解决**：选择性迁移、域自适应机制、注意力机制

### 7. 分布式计算的通信瓶颈
- **错误**：频繁的全局同步
- **问题**：通信成本超过计算成本
- **解决**：异步更新、本地计算优先、梯度压缩

### 8. 内存溢出问题
- **错误**：存储完整的中间张量
- **问题**：即使稀疏张量也可能产生稠密中间结果
- **解决**：流式计算、分块处理、及时释放内存

## 最佳实践检查清单

### 张量分解设计审查

1. **数据建模**
   - [ ] 是否正确识别了所有相关维度？
   - [ ] 张量的阶数是否合理（通常3-5阶）？
   - [ ] 是否考虑了时间动态性？
   - [ ] 稀疏性模式是否已分析？

2. **算法选择**
   - [ ] CP vs Tucker：是否基于可解释性需求选择？
   - [ ] 秩的选择是否经过验证？
   - [ ] 是否需要非负约束？
   - [ ] 正则化策略是否合理？

3. **实现优化**
   - [ ] 存储格式是否匹配访问模式？
   - [ ] MTTKRP是否已优化？
   - [ ] 是否利用了并行计算？
   - [ ] 内存使用是否在控制范围内？

4. **分布式计算**
   - [ ] 分区策略是否均衡负载？
   - [ ] 通信模式是否已优化？
   - [ ] 是否处理了节点故障？
   - [ ] 同步频率是否合理？

5. **跨域推荐**
   - [ ] 共享机制是否合理？
   - [ ] 是否评估了负迁移风险？
   - [ ] 隐私保护措施是否到位？
   - [ ] 域自适应机制是否必要？

6. **工程实践**
   - [ ] 是否有增量更新能力？
   - [ ] 模型版本管理策略？
   - [ ] A/B测试框架是否就绪？
   - [ ] 监控指标是否完善？

### 性能优化检查

1. **计算性能**
   - [ ] 热点函数是否已识别并优化？
   - [ ] 向量化机会是否充分利用？
   - [ ] 缓存友好的数据布局？
   - [ ] GPU利用率是否充分？

2. **可扩展性**
   - [ ] 是否在不同数据规模下测试？
   - [ ] 强扩展性和弱扩展性如何？
   - [ ] 瓶颈在计算还是通信？
   - [ ] 是否有规模上限？

3. **数值稳定性**
   - [ ] 条件数是否监控？
   - [ ] 是否使用了数值稳定的算法？
   - [ ] 精度损失是否可接受？
   - [ ] 是否有异常值处理？

## 进一步研究方向

1. **自适应张量分解**：根据数据特性自动选择分解方法和参数
2. **神经张量网络**：结合深度学习和张量分解的优势
3. **量子张量算法**：利用量子计算加速大规模张量运算
4. **因果张量分析**：从关联到因果的推荐系统
5. **持续学习张量**：处理概念漂移和遗忘机制
6. **图张量网络**：结合图结构和张量方法

## 练习题

### 练习16.1：张量建模设计（基础）
设计一个四阶张量来建模"用户-物品-时间-地点"的交互。讨论这个张量的稀疏性特征，并估计在1M用户、100K物品、365天、1000个地点的场景下，需要多少内存来存储（假设1%的观测率）。

**Hint**: 考虑COO格式的存储开销，每个非零元素需要存储坐标和值。

<details>
<summary>答案</summary>

四阶张量 $\mathcal{X} \in \mathbb{R}^{10^6 \times 10^5 \times 365 \times 10^3}$

总元素数：$10^6 \times 10^5 \times 365 \times 10^3 = 3.65 \times 10^{16}$

观测元素数（1%）：$3.65 \times 10^{14}$

COO格式存储：
- 每个元素需要：4个整数索引（4×4=16字节）+ 1个浮点值（8字节）= 24字节
- 总内存：$3.65 \times 10^{14} \times 24 \text{ bytes} ≈ 8.76 \text{ PB}$

这说明即使是稀疏存储，大规模张量仍然面临严重的存储挑战。实践中需要：
1. 进一步压缩（如时间分桶）
2. 分布式存储
3. 采样或sketch技术
</details>

### 练习16.2：CP分解收敛性分析（基础）
证明当张量元素独立同分布于 $\mathcal{N}(0, \sigma^2)$ 时，CP-ALS算法的条件数随秩 $R$ 的增长关系。这对算法收敛有什么影响？

**Hint**: 考虑 Khatri-Rao 积的 Gram 矩阵 $\mathbf{V}^{(n)}$ 的特征值分布。

<details>
<summary>答案</summary>

对于随机初始化的因子矩阵，Gram 矩阵 $\mathbf{V}^{(n)} = \prod_{k \neq n} (\mathbf{A}^{(k)})^T \mathbf{A}^{(k)}$ 的条件数近似为：

$$\kappa(\mathbf{V}^{(n)}) \approx O(R^{N-1})$$

其中 $N$ 是张量的阶数。

证明要点：
1. 每个 $(\mathbf{A}^{(k)})^T \mathbf{A}^{(k)}$ 的条件数约为 $O(R)$
2. Hadamard 积保持正定性但会放大条件数
3. 最终条件数是各项的乘积

影响：
- 高秩分解数值不稳定
- 收敛速度显著降低
- 需要正则化或预条件技术
</details>

### 练习16.3：稀疏MTTKRP优化（挑战）
设计一个缓存友好的稀疏MTTKRP算法，使得内存访问模式最大化利用 L1/L2 缓存。分析你的算法在不同稀疏模式下的性能。

**Hint**: 考虑非零元素的重排序和分块策略。

<details>
<summary>答案</summary>

缓存友好的稀疏MTTKRP算法：

1. **非零元素重排序**：
   - 使用 Z-order (Morton order) 对坐标排序
   - 保持空间局部性

2. **分块策略**：
   ```
   将张量分成大小为 B×B×B 的块
   for each 块 b:
       加载相关的因子矩阵片段到缓存
       for each 非零元素 (i,j,k,v) in 块 b:
           for r = 1 to R:
               Y[i,r] += v * A[j,r] * B[k,r]
   ```

3. **性能分析**：
   - 随机稀疏：缓存命中率 ≈ B³/总块数
   - 结构化稀疏（如对角带）：可达到 90%+ 缓存命中率
   - 块大小选择：B = ∛(L2_cache_size / (3×R×sizeof(float)))

4. **进一步优化**：
   - 使用 tiling 处理因子矩阵
   - SIMD 向量化内层循环
   - 预取下一个块的数据
</details>

### 练习16.4：Tucker vs CP选择（挑战）
给定一个三阶张量，设计一个自动化方法来决定使用 Tucker 分解还是 CP 分解。你的方法应该考虑准确性、可解释性和计算效率。

**Hint**: 可以从核心一致性诊断（CORCONDIA）开始。

<details>
<summary>答案</summary>

自动选择算法：

1. **快速预分析**：
   ```python
   # 计算张量的模态相关性
   for n in [1, 2, 3]:
       S[n] = svd(unfold(X, n), k=10)
       decay_rate[n] = S[n][5] / S[n][0]
   
   # 如果所有模态的奇异值都快速衰减，倾向于CP
   if all(decay_rate < 0.1):
       initial_choice = "CP"
   else:
       initial_choice = "Tucker"
   ```

2. **CORCONDIA 测试**（针对CP）：
   ```
   对于秩 R = 1, 2, ..., R_max:
       执行 CP 分解得到 [[A, B, C]]
       计算核心一致性：
       G = X ×₁ A† ×₂ B† ×₃ C†
       corcondia[R] = 100 * (1 - ∑ᵢ≠ⱼ g²ᵢⱼₖ / ∑ᵢⱼₖ g²ᵢⱼₖ)
   
   如果 max(corcondia) < 90:
       选择 Tucker
   ```

3. **复杂度-精度权衡**：
   ```
   CP_cost = O(nnz(X) × R × iterations)
   Tucker_cost = O(nnz(X) × (R₁R₂ + R₂R₃ + R₁R₃))
   
   if Tucker_cost / CP_cost > threshold AND accuracy_gap < ε:
       选择 CP
   ```

4. **最终决策树**：
   - 需要可解释性 → CP
   - 秩很低(<10) → CP  
   - 模态间相关性强 → Tucker
   - 内存受限 → CP
   - 否则 → 交叉验证选择
</details>

### 练习16.5：跨域负迁移检测（开放性）
设计一个在线算法来检测跨域推荐中的负迁移现象。算法应该能够自动识别哪些共享的因子对目标域有害，并提出缓解策略。

**Hint**: 考虑使用验证集上的性能变化和梯度分析。

<details>
<summary>答案</summary>

负迁移检测与缓解算法：

1. **基线建立**：
   - 仅使用目标域数据训练模型M₀
   - 记录验证集性能 P₀

2. **增量共享测试**：
   ```python
   for each 共享层级 l in [1, ..., L]:
       # 逐步增加共享程度
       M_l = CMF with sharing up to layer l
       P_l = evaluate(M_l, validation_set)
       
       # 检测性能下降
       if P_l < P_{l-1} - ε:
           negative_transfer_detected(layer=l)
   ```

3. **因子级别分析**：
   ```python
   # 计算各因子的贡献
   for each factor f:
       # 梯度相似性分析
       g_source = gradient(L_source, f)
       g_target = gradient(L_target, f)
       similarity[f] = cosine(g_source, g_target)
       
       # 负相似性表示可能的负迁移
       if similarity[f] < -threshold:
           suspicious_factors.add(f)
   ```

4. **自适应权重机制**：
   ```python
   # 为每个共享因子学习权重
   w[f] = sigmoid(MLP([g_source[f], g_target[f], similarity[f]]))
   shared_factor[f] = w[f] * source_factor[f] + (1-w[f]) * target_factor[f]
   ```

5. **缓解策略**：
   - **选择性共享**：只共享正迁移的因子
   - **梯度反转**：对负迁移因子使用梯度反转层
   - **域自适应正则化**：L = L_target + λ·MMD(source, target)
   - **元学习**：学习如何快速适应新域

6. **在线监控**：
   ```python
   # 滑动窗口监控
   window_performance = deque(maxlen=100)
   if trend(window_performance) < 0:
       trigger_retraining()
   ```
</details>

### 练习16.6：分布式Tucker分解通信优化（挑战）
设计一个通信高效的分布式Tucker-ALS算法，最小化节点间的数据传输。考虑 P 个节点，每个节点存储张量的一个分区。

**Hint**: 利用 Tucker 分解的结构，设计部分更新和延迟同步策略。

<details>
<summary>答案</summary>

通信优化的分布式Tucker-ALS：

1. **张量分区策略**：
   ```
   # Mode-1 分区（用户维度）
   每个节点 p 存储 X[i_p:i_{p+1}, :, :]
   因子矩阵分区：U^(1)[i_p:i_{p+1}, :] 本地存储
   ```

2. **计算-通信重叠**：
   ```python
   for iteration in range(max_iter):
       # 阶段1：本地计算
       for n in [1, 2, 3]:
           if n == 1:  # 本地模态
               U_local[n] = local_tucker_update(X_local, U[2], U[3])
           else:
               # 准备通信的数据
               prepare_scatter_data(U[n])
       
       # 阶段2：异步通信
       futures = []
       for n in [2, 3]:
           futures.append(async_allgather(U_local[n]))
       
       # 阶段3：本地计算其他工作
       compute_core_tensor_contribution()
       
       # 阶段4：等待通信完成
       U[2], U[3] = wait_all(futures)
   ```

3. **压缩通信**：
   ```python
   # 低秩压缩
   def compress_factor(U, compression_rate=0.1):
       rank = int(U.shape[1] * compression_rate)
       U_compressed, S, V = randomized_svd(U, rank)
       return U_compressed, S, V
   
   # 只传输压缩的因子
   U_c, S, V = compress_factor(U_local)
   send(U_c, S, V)  # 通信量减少 90%
   ```

4. **延迟同步**：
   ```python
   # 不是每次迭代都同步
   if iteration % sync_frequency == 0:
       synchronize_all_factors()
   else:
       # 使用过期的因子继续计算
       use_stale_factors()
   ```

5. **通信模式优化**：
   - Ring-allreduce for factor matrices
   - Butterfly mixing for partial updates
   - Hierarchical reduction for multi-level clusters

6. **性能分析**：
   - 通信复杂度：O(P × R × (I/P)) per iteration
   - 优化后：O(√P × R × (I/P)) with compression
   - 进一步优化：O(log P × R × (I/P)) with hierarchical
</details>

### 练习16.7：实时张量补全（开放性）
设计一个在线算法，当新的观测 $(i,j,k,v)$ 到达时，能够实时更新张量补全结果。要求更新时间复杂度为 $O(R^2)$ 或更好。

**Hint**: 考虑使用 Sherman-Morrison-Woodbury 公式和缓存中间结果。

<details>
<summary>答案</summary>

实时张量补全算法：

1. **预计算结构**：
   ```python
   # 维护因子矩阵和辅助矩阵
   A, B, C  # 因子矩阵
   G = (B.T @ B) * (C.T @ C)  # Gram矩阵缓存
   invG = inv(G + λI)  # 预计算的逆
   ```

2. **单点更新算法**：
   ```python
   def update_single_observation(i, j, k, v_new):
       # 提取相关向量
       a_i = A[i, :]
       b_j = B[j, :]
       c_k = C[k, :]
       
       # 计算预测误差
       v_pred = np.sum(a_i * b_j * c_k)
       error = v_new - v_pred
       
       # Sherman-Morrison 更新
       z = b_j * c_k  # R维向量
       denominator = 1 + z @ invG @ z
       
       # 更新 A[i,:]
       A[i, :] += (error / denominator) * invG @ z
       
       # 更新辅助矩阵（可选，用于periodic refresh）
       if update_count % refresh_period == 0:
           update_auxiliary_matrices()
   ```

3. **批量更新优化**：
   ```python
   def batch_update(observations):
       # 按第一个模式分组
       grouped = group_by_mode1(observations)
       
       for i, group in grouped.items():
           # 构建批量更新矩阵
           Z = stack([B[j] * C[k] for (_, j, k, _) in group])
           V = [v for (_, _, _, v) in group]
           
           # 批量 Woodbury 更新
           M = I + Z @ invG @ Z.T
           A[i, :] += invG @ Z.T @ solve(M, V - predictions)
   ```

4. **动态秩调整**：
   ```python
   # 监控重构误差
   if reconstruction_error > threshold:
       # 增加秩
       add_new_factor_column()
   elif rank_usage < 0.8:
       # 减少秩
       remove_least_important_factor()
   ```

5. **缓存策略**：
   - LRU 缓存最近的预测值
   - 增量更新 Gram 矩阵
   - 周期性重新计算以控制数值误差

时间复杂度：O(R²) 每次更新（由矩阵-向量乘法主导）
空间复杂度：O(R²) 用于存储逆矩阵
</details>

### 练习16.8：张量神经网络混合模型（开放性）
设计一个结合张量分解和深度学习的推荐模型。模型应该能够利用张量分解的可解释性和神经网络的表达能力。

**Hint**: 考虑将张量分解的因子作为神经网络的输入或正则化项。

<details>
<summary>答案</summary>

张量-神经网络混合架构：

1. **基础架构**：
   ```python
   class TensorNeuralHybrid(nn.Module):
       def __init__(self, dims, rank, hidden_dims):
           # 张量分解组件
           self.cp_factors = nn.ParameterList([
               nn.Parameter(torch.randn(dim, rank)) 
               for dim in dims
           ])
           
           # 神经网络组件
           self.mlp = MLP(rank * len(dims), hidden_dims)
           self.fusion_gate = nn.Linear(hidden_dims[-1], rank)
   ```

2. **前向传播**：
   ```python
   def forward(self, indices):
       # 张量分解预测
       cp_embeds = []
       for n, idx in enumerate(indices):
           cp_embeds.append(self.cp_factors[n][idx])
       
       # CP 预测
       cp_pred = torch.prod(torch.stack(cp_embeds), dim=0).sum()
       
       # 神经网络增强
       concat_embeds = torch.cat(cp_embeds, dim=-1)
       nn_features = self.mlp(concat_embeds)
       
       # 自适应融合
       gate = torch.sigmoid(self.fusion_gate(nn_features))
       final_pred = gate * cp_pred + (1-gate) * nn_features.sum()
       
       return final_pred, cp_embeds  # 返回嵌入用于可解释性
   ```

3. **多任务学习**：
   ```python
   def multi_task_loss(pred, target, cp_embeds, aux_targets):
       # 主任务：预测损失
       main_loss = F.mse_loss(pred, target)
       
       # 辅助任务：因子可解释性
       interp_loss = factor_interpretability_loss(cp_embeds)
       
       # 正交性正则化
       ortho_loss = orthogonality_regularization(self.cp_factors)
       
       # 稀疏性正则化
       sparse_loss = sparsity_regularization(self.mlp)
       
       return main_loss + λ₁*interp_loss + λ₂*ortho_loss + λ₃*sparse_loss
   ```

4. **因子可解释性约束**：
   ```python
   def factor_interpretability_loss(factors):
       # 促进因子的聚类结构
       cluster_loss = 0
       for factor in factors:
           # 计算因子间的相似度矩阵
           sim_matrix = F.cosine_similarity(factor.unsqueeze(1), 
                                           factor.unsqueeze(0), dim=2)
           # 促进块对角结构
           cluster_loss += block_diagonal_loss(sim_matrix)
       return cluster_loss
   ```

5. **增量学习能力**：
   ```python
   def incremental_update(self, new_data):
       # 冻结神经网络，只更新张量因子
       for param in self.mlp.parameters():
           param.requires_grad = False
       
       # 快速适应新数据
       optimizer = torch.optim.Adam(self.cp_factors.parameters(), lr=0.01)
       for epoch in range(5):  # 少量epoch
           loss = self.forward_on_new_data(new_data)
           loss.backward()
           optimizer.step()
       
       # 解冻进行联合微调
       for param in self.mlp.parameters():
           param.requires_grad = True
   ```

6. **可解释性分析**：
   ```python
   def explain_prediction(self, indices):
       with torch.no_grad():
           pred, cp_embeds = self.forward(indices)
           
           # 因子贡献分析
           contributions = []
           for n, embed in enumerate(cp_embeds):
               contrib = embed / embed.norm()
               contributions.append(contrib)
           
           # 注意力可视化
           attention = self.get_attention_weights(indices)
           
           return {
               'prediction': pred,
               'factor_contributions': contributions,
               'attention_weights': attention,
               'top_features': self.get_top_features(indices)
           }
   ```

优势：
- 结合了张量分解的可解释性
- 利用神经网络捕获非线性模式
- 支持增量学习和多任务学习
- 提供预测解释机制
</details>