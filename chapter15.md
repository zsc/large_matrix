# 第15章：实时推荐的增量矩阵方法

在现代推荐系统中，用户行为数据以流的形式不断产生，系统需要实时响应并更新推荐结果。传统的批处理矩阵分解方法已无法满足毫秒级延迟的需求。本章深入探讨如何设计高效的增量算法，在保证推荐质量的同时实现快速更新。我们将分析在线学习的理论保证、数值稳定性挑战，以及工程实现的最佳实践。

## 15.1 在线矩阵分解的遗忘机制

### 15.1.1 时间衰减的数学建模

在实时推荐场景中，用户兴趣随时间演化，旧数据的重要性逐渐降低。我们需要设计合理的遗忘机制来平衡历史信息与新鲜度。遗忘机制的设计直接影响模型对趋势变化的敏感度和稳定性。

**指数遗忘模型**

考虑矩阵分解的目标函数：
$$\mathcal{L}_t = \sum_{(i,j) \in \Omega_t} w_{ij}^{(t)} (r_{ij} - \mathbf{u}_i^T \mathbf{v}_j)^2 + \lambda (\|\mathbf{U}\|_F^2 + \|\mathbf{V}\|_F^2)$$

其中时间权重 $w_{ij}^{(t)} = \exp(-\beta(t - t_{ij}))$，$t_{ij}$ 是观测时间，$\beta$ 控制遗忘速率。

指数遗忘的优势在于：
- 平滑的权重衰减，避免突变
- 参数 $\beta$ 具有明确的物理意义（半衰期 $t_{1/2} = \ln(2)/\beta$）
- 与在线学习理论中的指数加权平均（EWA）相联系

**滑动窗口方法**

另一种方法是只考虑最近 $W$ 个时间单位的数据：
$$w_{ij}^{(t)} = \begin{cases} 
1 & \text{if } t - t_{ij} \leq W \\
0 & \text{otherwise}
\end{cases}$$

这种硬截断虽然简单，但可能导致信息突变。实践中常用软窗口变体：
$$w_{ij}^{(t)} = \sigma\left(\frac{W - (t - t_{ij})}{\tau}\right)$$
其中 $\sigma$ 是sigmoid函数，$\tau$ 控制过渡的平滑度。

**自适应遗忘模型**

更高级的方法是让遗忘率随数据特性动态调整：
$$\beta_{ij}(t) = \beta_0 \cdot f(\text{volatility}_i, \text{popularity}_j, \text{sparsity}_{ij})$$

其中：
- $\text{volatility}_i$ 衡量用户兴趣的变化速度
- $\text{popularity}_j$ 反映物品的流行度趋势
- $\text{sparsity}_{ij}$ 考虑数据稀疏性

### 15.1.2 增量SGD的收敛性分析

对于流式数据 $(i_t, j_t, r_t)$，标准SGD更新为：
$$\mathbf{u}_{i_t} \leftarrow \mathbf{u}_{i_t} - \eta_t \nabla_{\mathbf{u}_{i_t}} \ell_t$$
$$\mathbf{v}_{j_t} \leftarrow \mathbf{v}_{j_t} - \eta_t \nabla_{\mathbf{v}_{j_t}} \ell_t$$

其中 $\ell_t = (r_t - \mathbf{u}_{i_t}^T \mathbf{v}_{j_t})^2 + \lambda(\|\mathbf{u}_{i_t}\|^2 + \|\mathbf{v}_{j_t}\|^2)$。

**收敛性定理**（非凸情况）

在以下条件下：
1. 损失函数 $\ell_t$ 是 $L$-光滑的
2. 梯度有界：$\|\nabla \ell_t\| \leq G$
3. 学习率满足：$\eta_t = \eta_0/\sqrt{t}$

则增量SGD收敛到稳定点的速率为：
$$\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T \|\nabla \mathcal{L}_t\|^2\right] = O\left(\frac{1}{\sqrt{T}}\right)$$

**带遗忘的SGD分析**

考虑指数遗忘权重，修正的更新规则变为：
$$\mathbf{u}_{i_t} \leftarrow (1-\gamma)\mathbf{u}_{i_t} - \eta_t e^{-\beta(t-t_{ij})} \nabla_{\mathbf{u}_{i_t}} \ell_t$$

其中 $\gamma$ 是衰减因子，防止历史信息完全主导。

**关键研究方向**：
- 非凸优化的在线regret界（特别是矩阵分解的低秩约束）
- 自适应学习率（AdaGrad、Adam）在矩阵分解中的理论保证
- 稀疏数据流下的样本复杂度
- 时变环境下的追踪误差（tracking error）分析

### 15.1.3 动态正则化策略

随着数据累积，不同用户/物品的样本数差异巨大。动态正则化可以解决这个问题：

$$\lambda_i^{(t)} = \lambda_0 / \sqrt{n_i^{(t)} + 1}$$

其中 $n_i^{(t)}$ 是用户 $i$ 到时刻 $t$ 的交互次数。

**理论基础**

这种设计基于贝叶斯观点：
- 先验：$\mathbf{u}_i \sim \mathcal{N}(0, \sigma^2\mathbf{I})$
- 后验精度随观测数增加：$\text{precision} \propto n_i^{(t)}$
- 等价于自适应正则化

**多尺度正则化**

考虑不同时间尺度的行为：
$$\lambda_i^{(t)} = \lambda_{\text{long}} / \sqrt{n_i^{\text{all}}} + \lambda_{\text{short}} / \sqrt{n_i^{\text{recent}}}$$

其中：
- $n_i^{\text{all}}$ 是历史总交互数
- $n_i^{\text{recent}}$ 是近期（如最近7天）交互数

**数值稳定性考虑**：
- 防止除零错误：使用 $n_i^{(t)} + \epsilon$，$\epsilon \sim 0.1$
- 正则化参数的上下界限制：$\lambda_{\min} \leq \lambda_i^{(t)} \leq \lambda_{\max}$
- 增量统计量的精确维护：使用Welford算法计算在线均值和方差
- 数值下溢处理：对极小的正则化值使用对数空间计算

### 15.1.4 混合遗忘策略

实践中，单一遗忘机制往往不够灵活。混合策略结合多种方法的优势：

**分层遗忘模型**
$$w_{ij}^{(t)} = \alpha \cdot w_{\text{exp}}^{(t)} + (1-\alpha) \cdot w_{\text{window}}^{(t)}$$

其中 $\alpha$ 可以自适应调整：
$$\alpha(t) = \sigma\left(\frac{\text{MSE}_{\text{exp}} - \text{MSE}_{\text{window}}}{\tau}\right)$$

**事件驱动遗忘**

某些事件（如节假日、促销）会导致用户行为突变：
$$\beta(t) = \begin{cases}
\beta_{\text{normal}} & \text{常规时期} \\
\beta_{\text{fast}} & \text{事件期间} \\
\beta_{\text{recovery}} & \text{事件后恢复期}
\end{cases}$$

**研究挑战**：
- 自动检测行为模式变化点
- 多用户群体的差异化遗忘策略
- 遗忘机制与推荐多样性的关系

## 15.2 用户/物品嵌入的快速更新

### 15.2.1 秩一更新的Sherman-Morrison公式

当新增一个评分时，可以利用秩一更新避免完全重新计算。这在实时系统中至关重要，将更新复杂度从 $O(r^3)$ 降至 $O(r^2)$。

给定 $\mathbf{A} = \mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$，新增样本 $(\mathbf{x}, y)$ 后：
$$\mathbf{A}' = \mathbf{A} + \mathbf{x}\mathbf{x}^T$$

利用Sherman-Morrison公式：
$$(\mathbf{A}')^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{x}\mathbf{x}^T\mathbf{A}^{-1}}{1 + \mathbf{x}^T\mathbf{A}^{-1}\mathbf{x}}$$

**数值稳定性增强**

标准Sherman-Morrison公式在 $1 + \mathbf{x}^T\mathbf{A}^{-1}\mathbf{x} \approx 0$ 时不稳定。稳定版本：

$$(\mathbf{A}')^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{u}\mathbf{u}^T}{1 + \|\mathbf{u}\|^2}$$
其中 $\mathbf{u} = \mathbf{A}^{-1/2}\mathbf{x}$

**删除操作的处理**

当需要删除旧数据（遗忘）时，使用减法版本：
$$(\mathbf{A} - \mathbf{x}\mathbf{x}^T)^{-1} = \mathbf{A}^{-1} + \frac{\mathbf{A}^{-1}\mathbf{x}\mathbf{x}^T\mathbf{A}^{-1}}{1 - \mathbf{x}^T\mathbf{A}^{-1}\mathbf{x}}$$

注意：需要检查 $\mathbf{x}^T\mathbf{A}^{-1}\mathbf{x} < 1$ 以保证正定性。

**应用于矩阵分解**

对于用户嵌入更新：
$$\mathbf{u}_i = (\mathbf{V}_{\Omega_i}^T\mathbf{V}_{\Omega_i} + \lambda\mathbf{I})^{-1}\mathbf{V}_{\Omega_i}^T\mathbf{r}_i$$

新增评分 $(i,j,r_{ij})$ 后，只需更新：
1. $\mathbf{A}_i' = \mathbf{A}_i + \mathbf{v}_j\mathbf{v}_j^T$
2. $\mathbf{b}_i' = \mathbf{b}_i + r_{ij}\mathbf{v}_j$
3. $\mathbf{u}_i' = (\mathbf{A}_i')^{-1}\mathbf{b}_i'$

### 15.2.2 块更新与并行化

对于批量更新，Woodbury矩阵恒等式提供了高效方案：
$$(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}^T\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{A}^{-1}$$

**批量评分更新**

当同时到达 $k$ 个新评分时：
- $\mathbf{U} = [\mathbf{v}_{j_1}, ..., \mathbf{v}_{j_k}]$
- $\mathbf{C} = \text{diag}(1, ..., 1)$
- $\mathbf{V} = \mathbf{U}$

复杂度：$O(r^2k + k^3)$，当 $k \ll r$ 时高效。

**并行化策略**

1. **用户级并行**：不同用户的嵌入可独立更新
   ```
   parallel for each user i:
       update A_i and u_i
   ```

2. **批内并行**：利用矩阵乘法的并行性
   - BLAS Level 3操作：`gemm`, `syrk`
   - GPU加速：适合大批量更新

3. **流水线并行**：
   - Stage 1: 收集更新，形成批
   - Stage 2: 计算矩阵更新
   - Stage 3: 更新嵌入向量

**实现要点**：
- 缓存 $\mathbf{A}^{-1}$ 的分解形式（如Cholesky）
- 异步更新的一致性保证：使用版本控制
- 数值误差累积的定期修正：每 $N$ 次更新后重新计算
- 内存局部性优化：按用户分组存储相关矩阵

### 15.2.3 懒惰求值与缓存策略

不是所有嵌入都需要实时更新。懒惰求值策略大幅减少计算量：

**三级更新策略**

1. **立即更新**（热用户）：
   - 活跃用户（最近1小时有交互）
   - 高价值用户（VIP、付费用户）
   - 更新延迟 < 100ms

2. **延迟更新**（温用户）：
   - 周期性活跃用户
   - 批量聚合后更新
   - 更新延迟 < 1分钟

3. **懒惰更新**（冷用户）：
   - 低活跃用户
   - 仅在查询时更新
   - 可接受陈旧结果

**智能缓存管理**

```
CacheEntry {
    embedding: Vector
    last_update: Timestamp
    pending_updates: Queue<Update>
    access_count: int
    update_cost: float
}
```

**更新决策函数**：
$$\text{should\_update} = \frac{\text{staleness} \times \text{importance}}{\text{update\_cost}} > \theta$$

其中：
- $\text{staleness} = t_{\text{now}} - t_{\text{last\_update}}$
- $\text{importance} = f(\text{access\_frequency}, \text{user\_value})$
- $\text{update\_cost}$ 考虑计算复杂度和当前负载

**版本化缓存**

支持多版本读取，避免更新阻塞查询：
```
VersionedEmbedding {
    versions: RingBuffer<(Version, Embedding)>
    current_version: AtomicInt
}
```

**研究线索**：
- 缓存命中率与推荐质量的权衡
- 分布式环境下的缓存一致性（使用Raft或Paxos）
- 近似更新的误差界：$\|\mathbf{u}_{\text{approx}} - \mathbf{u}_{\text{exact}}\| \leq \epsilon$
- 自适应缓存大小：基于内存压力和访问模式

### 15.2.4 增量矩阵分解的并行算法

**Lock-Free更新算法**

避免锁竞争的无锁更新：
```
AtomicUpdate(user_id, item_id, rating):
    loop:
        old_A = load(A[user_id])
        new_A = old_A + v[item_id] * v[item_id]'
        if CAS(A[user_id], old_A, new_A):
            break
```

**异步SGD的理论保证**

在延迟 $\tau$ 有界的情况下：
$$\mathbb{E}[\|\mathbf{u}_T - \mathbf{u}^*\|^2] \leq O\left(\frac{1}{\sqrt{T}} + \frac{\tau}{T}\right)$$

这表明适度的异步不会显著影响收敛性。

**分布式快照**

使用Chandy-Lamport算法实现一致性快照：
1. 主节点发起快照
2. 各节点记录本地状态
3. 记录传输中的消息
4. 组合形成全局一致状态

这允许在不停止系统的情况下进行checkpoint和恢复。

## 15.3 冷启动问题的矩阵补全视角

### 15.3.1 迁移学习框架

将冷启动视为少样本矩阵补全问题，利用已有用户/物品的知识构建有效的先验。这种方法将冷启动从"无中生有"转变为"知识迁移"。

**共享隐空间模型**

基本假设：新用户的隐向量可以从其特征预测：
$$\mathbf{u}_{\text{new}} = \mathbf{W}_u \mathbf{f}_u + \boldsymbol{\epsilon}_u$$

其中：
- $\mathbf{f}_u \in \mathbb{R}^d$ 是用户特征（人口统计学、注册信息等）
- $\mathbf{W}_u \in \mathbb{R}^{r \times d}$ 是学习的映射矩阵
- $\boldsymbol{\epsilon}_u$ 是个性化偏差

**分层贝叶斯模型**

更精细的建模考虑不确定性：
$$\begin{aligned}
\mathbf{u}_i &\sim \mathcal{N}(\mathbf{W}_u \mathbf{f}_i, \boldsymbol{\Sigma}_u) \\
\mathbf{W}_u &\sim \mathcal{MN}(\mathbf{M}_0, \boldsymbol{\Sigma}_u, \boldsymbol{\Omega}) \\
r_{ij} &\sim \mathcal{N}(\mathbf{u}_i^T \mathbf{v}_j, \sigma^2)
\end{aligned}$$

这提供了预测的不确定性估计，对主动学习至关重要。

**元学习视角**

将冷启动视为few-shot learning问题：
1. **元训练**：在历史用户上学习"如何快速学习"
2. **元测试**：对新用户快速适应

使用MAML（Model-Agnostic Meta-Learning）框架：
$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \sum_{i \in \mathcal{T}_{\text{train}}} \mathcal{L}(\mathbf{W} - \alpha\nabla_{\mathbf{W}}\mathcal{L}_i(\mathbf{W}))$$

**多任务学习**

不同用户群体共享部分结构：
$$\mathbf{U} = \mathbf{U}_{\text{shared}} + \sum_{k=1}^K \mathbf{U}_k \odot \mathbf{Z}_k$$

其中 $\mathbf{Z}_k$ 是群体指示矩阵。

### 15.3.2 主动学习策略

选择最优的初始交互来快速学习新用户偏好。关键是平衡探索（减少不确定性）与利用（提供好的推荐）。

**不确定性采样**

基于后验方差选择物品：
$$j^* = \arg\max_j \text{Var}[\hat{r}_{ij} | \mathcal{D}]$$

对于线性模型，方差可解析计算：
$$\text{Var}[\hat{r}_{ij}] = \mathbf{v}_j^T (\mathbf{V}_{\mathcal{D}}^T\mathbf{V}_{\mathcal{D}} + \lambda\mathbf{I})^{-1} \mathbf{v}_j$$

**信息增益最大化**

选择最大化互信息的物品：
$$j^* = \arg\max_j I(\mathbf{u}_i; r_{ij} | \mathcal{D})$$

对于高斯分布：
$$I(\mathbf{u}_i; r_{ij}) = \frac{1}{2}\log\left(1 + \frac{\mathbf{v}_j^T\boldsymbol{\Sigma}_{\mathbf{u}|i}\mathbf{v}_j}{\sigma^2}\right)$$

**Thompson采样**

平衡探索与利用的贝叶斯方法：
1. 从后验采样：$\tilde{\mathbf{u}}_i \sim p(\mathbf{u}_i | \mathcal{D})$
2. 选择期望回报最高的物品：$j^* = \arg\max_j \tilde{\mathbf{u}}_i^T \mathbf{v}_j$

**批量主动学习**

同时选择 $k$ 个物品的次模优化问题：
$$\mathcal{S}^* = \arg\max_{|\mathcal{S}|=k} f(\mathcal{S})$$

其中 $f(\mathcal{S})$ 是次模函数（如信息增益）。贪心算法提供 $(1-1/e)$ 近似保证。

**上下文相关的主动学习**

考虑时间、位置等上下文：
$$j^* = \arg\max_j g(\text{uncertainty}_j, \text{context}_t, \text{diversity}(\mathcal{S} \cup \{j\}))$$

### 15.3.3 理论保证

**样本复杂度界**

给定秩-$r$ 矩阵，达到 $\epsilon$-精度需要的样本数：

**均匀采样情况**：
$$m = O(r(n_1 + n_2)\log(1/\epsilon))$$

**非均匀采样的改进界**：
$$m = O(\mu r \log(n_1 + n_2) \log(1/\epsilon))$$
其中 $\mu$ 是相干性参数。

**在线学习的regret界**

对于冷启动用户的累积regret：
$$R_T = \sum_{t=1}^T (r^* - r_t) = O(\sqrt{rT\log T})$$

其中 $r^*$ 是最优推荐的回报。

**主动学习的加速**

相比随机选择，主动学习可以指数级减少样本需求：
- 随机：$m_{\text{random}} = O(r^2/\epsilon^2)$
- 主动：$m_{\text{active}} = O(r\log(1/\epsilon))$

**矩阵补全的信息论下界**

任何算法至少需要：
$$m \geq c \cdot r(n_1 + n_2 - r)$$
个观测才能精确恢复秩-$r$ 矩阵。

**关键挑战**：
- 非均匀缺失模式下的理论分析
- 在线设置下的自适应采样
- 隐私保护约束下的冷启动（差分隐私）
- 对抗性环境下的鲁棒性保证

### 15.3.4 实用技术与优化

**混合方法**

结合多种信息源：
1. **内容特征**：使用深度学习提取
2. **社交信号**：利用用户关系网络
3. **隐式反馈**：浏览、搜索等弱信号
4. **跨域信息**：从其他产品迁移知识

**渐进式个性化**

随着数据积累逐步过渡：
$$\mathbf{u}_i(t) = \alpha(t) \cdot \mathbf{u}_{\text{prior}} + (1-\alpha(t)) \cdot \mathbf{u}_{\text{learned}}$$

其中 $\alpha(t) = \exp(-\gamma \cdot n_i(t))$。

**群体先验**

利用相似用户群体：
$$\mathbf{u}_{\text{prior}} = \frac{1}{|\mathcal{N}_i|}\sum_{k \in \mathcal{N}_i} \mathbf{u}_k$$

其中 $\mathcal{N}_i$ 是基于特征的最近邻。

**在线更新的数值技巧**

1. **增量SVD**：避免重新分解
2. **稀疏更新**：只更新相关维度
3. **量化嵌入**：减少存储和计算
4. **分级精度**：新用户用低精度，逐步提升

## 15.4 时序动态的矩阵建模

### 15.4.1 时变隐因子模型

用户和物品的隐因子随时间演化：
$$\mathbf{u}_i(t) = \mathbf{u}_i(t-1) + \mathbf{w}_i(t)$$
$$\mathbf{v}_j(t) = \mathbf{v}_j(t-1) + \mathbf{z}_j(t)$$

其中 $\mathbf{w}_i(t), \mathbf{z}_j(t)$ 是演化噪声。

### 15.4.2 Kalman滤波的应用

将隐因子演化建模为状态空间模型：

**状态方程**：
$$\mathbf{x}_t = \mathbf{F}\mathbf{x}_t + \mathbf{w}_t$$

**观测方程**：
$$r_t = \mathbf{h}_t^T \mathbf{x}_t + v_t$$

Kalman滤波提供了最优的在线估计。

### 15.4.3 周期性模式捕获

许多推荐场景存在周期性（日、周、季节）：

**傅里叶基展开**：
$$\mathbf{u}_i(t) = \mathbf{u}_i^{(0)} + \sum_{k=1}^K (\mathbf{a}_{ik} \cos(\omega_k t) + \mathbf{b}_{ik} \sin(\omega_k t))$$

**研究方向**：
- 自适应频率检测
- 非平稳周期性建模
- 多尺度时间动态的联合建模

### 15.4.4 变点检测

用户兴趣的突变需要及时检测：

**CUSUM算法**的矩阵版本：
$$S_t = \max(0, S_{t-1} + \|\mathbf{r}_t - \mathbf{U}_t\mathbf{V}_t^T\|_F - \delta)$$

当 $S_t > h$ 时触发变点警报。

**高级主题**：
- 在线贝叶斯变点检测
- 多用户协同变点检测
- 假阳性率与检测延迟的权衡

## 15.5 常见陷阱与错误（Gotchas）

### 15.5.1 数值稳定性陷阱

1. **梯度爆炸/消失**
   - 错误：直接使用固定学习率
   - 正确：自适应学习率 + 梯度裁剪
   - 诊断：监控梯度范数 $\|\nabla\mathcal{L}\|$

2. **矩阵奇异性**
   - 错误：直接求逆 $(\mathbf{X}^T\mathbf{X})^{-1}$
   - 正确：添加正则化 $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$
   - 诊断：检查条件数 $\kappa(\mathbf{A})$

3. **浮点误差累积**
   - 错误：无限累积增量更新
   - 正确：定期重新计算基准值
   - 诊断：比较增量结果与批处理结果

### 15.5.2 算法设计陷阱

1. **过度遗忘**
   - 症状：模型性能剧烈波动
   - 原因：遗忘率 $\beta$ 设置过大
   - 解决：自适应遗忘率，考虑数据稀疏性

2. **冷启动过拟合**
   - 症状：新用户推荐极端化
   - 原因：样本过少时过度更新
   - 解决：设置最小样本阈值，使用先验

3. **并发更新冲突**
   - 症状：结果不确定性
   - 原因：多线程同时更新同一嵌入
   - 解决：细粒度锁或无锁算法

### 15.5.3 系统实现陷阱

1. **内存泄漏**
   - 原因：历史数据无限增长
   - 解决：实现严格的生命周期管理
   - 工具：内存profiler定期检查

2. **缓存失效风暴**
   - 原因：大规模更新触发全局失效
   - 解决：渐进式失效策略
   - 监控：缓存命中率实时监控

## 15.6 最佳实践检查清单

### 15.6.1 算法设计审查

- [ ] **遗忘机制设计**
  - 是否考虑了数据分布的非平稳性？
  - 遗忘率是否自适应调整？
  - 是否保留了关键历史信息？

- [ ] **更新策略优化**
  - 是否利用了矩阵结构（低秩、稀疏）？
  - 增量更新的计算复杂度是否优于重新训练？
  - 是否实现了懒惰求值？

- [ ] **数值稳定性保证**
  - 是否添加了适当的正则化？
  - 是否实现了梯度裁剪？
  - 是否定期检查数值健康度？

### 15.6.2 系统实现审查

- [ ] **并发控制**
  - 是否正确处理了并发更新？
  - 锁粒度是否合理？
  - 是否考虑了无锁算法？

- [ ] **资源管理**
  - 内存使用是否有上界？
  - 是否实现了优雅降级？
  - 是否有资源泄漏检测？

- [ ] **监控与调试**
  - 是否记录了关键性能指标？
  - 是否可以回放历史更新？
  - 是否有异常检测机制？

### 15.6.3 性能优化审查

- [ ] **计算优化**
  - 是否使用了SIMD指令？
  - 矩阵运算是否cache-friendly？
  - 是否考虑了GPU加速？

- [ ] **存储优化**
  - 是否使用了紧凑的数据结构？
  - 热数据是否在内存中？
  - 是否实现了分层存储？

## 15.7 本章小结

本章深入探讨了实时推荐系统中的增量矩阵方法，主要贡献包括：

1. **理论框架**：建立了在线矩阵分解的统一分析框架，包括遗忘机制、收敛性保证和regret界。

2. **算法创新**：
   - 自适应遗忘率的在线SGD
   - 基于Sherman-Morrison的快速嵌入更新
   - 矩阵补全视角的冷启动解决方案
   - Kalman滤波的时序建模

3. **实践指导**：总结了常见陷阱、最佳实践和性能优化技巧。

**关键公式汇总**：

- 指数遗忘权重：$w_{ij}^{(t)} = \exp(-\beta(t - t_{ij}))$
- Sherman-Morrison更新：$(\mathbf{A} + \mathbf{x}\mathbf{x}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{x}\mathbf{x}^T\mathbf{A}^{-1}}{1 + \mathbf{x}^T\mathbf{A}^{-1}\mathbf{x}}$
- 动态正则化：$\lambda_i^{(t)} = \lambda_0 / \sqrt{n_i^{(t)} + 1}$
- 时变隐因子：$\mathbf{u}_i(t) = \mathbf{u}_i(t-1) + \mathbf{w}_i(t)$

**未来研究方向**：
- 深度学习与增量矩阵方法的结合
- 联邦学习场景下的分布式增量更新
- 因果推断在动态推荐中的应用
- 量子算法加速矩阵更新

## 15.8 练习题

### 基础题

**练习15.1** 证明指数遗忘权重下的SGD更新等价于求解一个加权最小二乘问题。

*提示*：考虑累积损失函数 $\sum_{s=1}^t \beta^{t-s} \ell_s$。

<details>
<summary>答案</summary>

定义累积损失：
$$\mathcal{L}_t = \sum_{s=1}^t \beta^{t-s} (r_s - \mathbf{u}_{i_s}^T \mathbf{v}_{j_s})^2$$

对 $\mathbf{u}_i$ 求导：
$$\nabla_{\mathbf{u}_i} \mathcal{L}_t = -2 \sum_{s: i_s=i} \beta^{t-s} (r_s - \mathbf{u}_i^T \mathbf{v}_{j_s}) \mathbf{v}_{j_s}$$

这正是加权最小二乘的梯度，权重为 $w_s = \beta^{t-s}$。

</details>

**练习15.2** 给定秩为 $r$ 的 $m \times n$ 矩阵，使用Sherman-Morrison公式计算添加一个新行后的SVD更新复杂度。

*提示*：考虑增量SVD的计算步骤。

<details>
<summary>答案</summary>

原始SVD：$\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$

添加新行 $\mathbf{a}^T$ 后：
$$\mathbf{M}' = \begin{bmatrix} \mathbf{M} \\ \mathbf{a}^T \end{bmatrix}$$

计算步骤：
1. 投影：$\mathbf{p} = \mathbf{V}^T\mathbf{a}$，复杂度 $O(nr)$
2. 正交分量：$\mathbf{q} = \mathbf{a} - \mathbf{V}\mathbf{p}$，复杂度 $O(nr)$
3. 更新小矩阵SVD：$O(r^2)$

总复杂度：$O(nr + r^2)$，远小于重新计算 $O(mn\min(m,n))$。

</details>

**练习15.3** 设计一个自适应遗忘率算法，使得稀疏用户的数据保留更久。

*提示*：遗忘率应该与用户活跃度成反比。

<details>
<summary>答案</summary>

自适应遗忘率：
$$\beta_i(t) = \beta_0 \cdot \frac{\bar{n}}{n_i(t) + \epsilon}$$

其中：
- $n_i(t)$ 是用户 $i$ 在时间窗口内的交互次数
- $\bar{n}$ 是所有用户的平均交互次数
- $\epsilon$ 防止除零

这样稀疏用户（$n_i(t)$ 小）的 $\beta_i(t)$ 更小，数据保留更久。

</details>

### 挑战题

**练习15.4** 推导在线矩阵分解的regret界，考虑数据流是对抗性选择的情况。

*提示*：使用在线凸优化的分析框架，注意矩阵分解的非凸性。

<details>
<summary>答案</summary>

定义regret：
$$R_T = \sum_{t=1}^T \ell_t(\mathbf{U}_t, \mathbf{V}_t) - \min_{\mathbf{U},\mathbf{V}} \sum_{t=1}^T \ell_t(\mathbf{U}, \mathbf{V})$$

关键步骤：
1. 限制在有界集合：$\|\mathbf{U}\|_F \leq B_U$，$\|\mathbf{V}\|_F \leq B_V$
2. 使用Online Gradient Descent的regret界
3. 处理非凸性：考虑局部线性化

在适当条件下，可得：
$$R_T = O(\sqrt{T}(B_U + B_V))$$

注意：这是局部regret，全局最优需要额外假设。

</details>

**练习15.5** 设计一个分布式增量矩阵分解算法，支持异步更新且保证最终一致性。

*提示*：考虑参数服务器架构和延迟补偿。

<details>
<summary>答案</summary>

算法框架：
1. **参数服务器**存储全局嵌入 $\mathbf{U}, \mathbf{V}$
2. **工作节点**处理本地数据流

异步更新协议：
```
Worker k:
1. Pull: 获取相关嵌入 u_i, v_j
2. Compute: 计算梯度 g_u, g_v
3. Push: 发送带时间戳的更新 (g_u, g_v, τ)

Server:
1. 接收更新 (g, τ)
2. 延迟补偿：g' = g * decay(t - τ)
3. 应用更新：param += -η * g'
```

一致性保证：
- 有界延迟：$t - τ \leq \Delta$
- 衰减函数：$\text{decay}(\delta) = \exp(-\alpha\delta)$
- 收敛条件：$\eta < \frac{1}{L(1 + \alpha\Delta)}$

</details>

**练习15.6** 分析Kalman滤波在隐因子演化中的计算瓶颈，提出一个近似算法将复杂度从 $O(r^3)$ 降到 $O(r)$。

*提示*：利用协方差矩阵的特殊结构。

<details>
<summary>答案</summary>

标准Kalman更新的瓶颈在协方差更新：
$$\mathbf{P}_t = \mathbf{P}_{t|t-1} - \mathbf{K}_t\mathbf{H}_t\mathbf{P}_{t|t-1}$$

近似方案：
1. **对角近似**：假设 $\mathbf{P}_t \approx \text{diag}(p_1, ..., p_r)$
2. **更新公式**：
   $$p_i^{(t)} = \frac{p_i^{(t-1)} + q}{1 + h_i^2(p_i^{(t-1)} + q)/r}$$
   
3. **误差补偿**：定期（每 $k$ 步）运行完整更新

复杂度分析：
- 对角更新：$O(r)$ per step
- 完整更新：$O(r^3)$ every $k$ steps
- 均摊复杂度：$O(r + r^3/k)$

</details>

**练习15.7** 推导多任务学习框架下的增量矩阵分解，其中不同任务共享部分隐空间。

*提示*：使用分块矩阵表示共享和特定结构。

<details>
<summary>答案</summary>

多任务隐因子分解：
$$\mathbf{U}_i^{(k)} = [\mathbf{U}_i^{(\text{shared})}, \mathbf{U}_i^{(k,\text{specific})}]$$

目标函数：
$$\mathcal{L} = \sum_{k=1}^K \sum_{(i,j) \in \Omega_k} (r_{ij}^{(k)} - (\mathbf{U}_i^{(k)})^T \mathbf{V}_j^{(k)})^2 + \lambda \|\mathbf{U}^{(\text{shared})}\|_F^2$$

增量更新策略：
1. 共享部分：聚合所有任务的梯度
   $$\mathbf{u}_i^{(\text{shared})} \leftarrow \mathbf{u}_i^{(\text{shared})} - \eta \sum_k \alpha_k \nabla_{\text{shared}} \ell^{(k)}$$

2. 特定部分：独立更新
   $$\mathbf{u}_i^{(k,\text{specific})} \leftarrow \mathbf{u}_i^{(k,\text{specific})} - \eta \nabla_{\text{specific}} \ell^{(k)}$$

关键挑战：
- 平衡共享与特定的贡献
- 任务权重 $\alpha_k$ 的自适应调整
- 异构数据流的处理

</details>

**练习15.8** 设计一个在线算法同时处理显式评分和隐式反馈，考虑两种数据的不同噪声特性和到达频率。

*提示*：使用多目标优化框架。

<details>
<summary>答案</summary>

统一建模框架：
$$\mathcal{L} = \underbrace{\sum_{(i,j) \in \Omega_e} (r_{ij} - \mathbf{u}_i^T\mathbf{v}_j)^2}_{\text{显式评分}} + \underbrace{\sum_{(i,j) \in \Omega_i} c_{ij}(p_{ij} - \mathbf{u}_i^T\mathbf{v}_j)^2}_{\text{隐式反馈}}$$

其中：
- $c_{ij} = 1 + \alpha \log(1 + \text{count}_{ij})$ 是置信度权重
- $p_{ij} = 1$ 表示有交互，$p_{ij} = 0$ 表示采样的负例

自适应更新策略：
```
if explicit_rating arrives:
    η_e = η_0 / sqrt(n_explicit)
    update with weight w_e
    
if implicit_feedback arrives:
    η_i = η_0 / sqrt(n_implicit)
    update with weight w_i * c_ij
```

动态权重调整：
$$w_e : w_i = \frac{\text{var}[\text{implicit}]}{\text{var}[\text{explicit}]} \cdot \frac{|\Omega_i|}{|\Omega_e|}$$

这平衡了两种数据源的贡献，考虑了噪声水平和数据量。

</details>