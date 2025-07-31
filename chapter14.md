# 第14章：大规模协同过滤的矩阵技术

推荐系统是现代互联网服务的核心组件，而矩阵分解技术则是其数学基石。本章深入探讨在处理数十亿用户和数百万物品时，如何设计和优化矩阵计算方法。我们将重点关注隐式反馈场景——这是实际系统中最常见但也最具挑战性的设置。通过学习本章，读者将掌握工业级推荐系统背后的核心矩阵技术，理解从理论到实践的关键权衡，并能够设计适合特定场景的高效算法。

## 14.1 隐式反馈矩阵分解的加权策略

### 14.1.1 隐式反馈与显式评分的本质差异

在显式评分系统中，用户主动为物品打分（如1-5星），我们可以直接将评分矩阵 $\mathbf{R} \in \mathbb{R}^{m \times n}$ 分解为：
$$\mathbf{R} \approx \mathbf{P}\mathbf{Q}^T$$
其中 $\mathbf{P} \in \mathbb{R}^{m \times k}$ 是用户因子矩阵，$\mathbf{Q} \in \mathbb{R}^{n \times k}$ 是物品因子矩阵。

然而，隐式反馈（点击、浏览、购买等）带来了根本性的挑战：

1. **正样本偏差**：观察到的交互不等同于偏好强度
2. **负样本缺失**：未交互不等于不感兴趣
3. **置信度差异**：不同类型交互的可靠性不同

这导致我们需要重新设计目标函数：
$$\min_{\mathbf{P},\mathbf{Q}} \sum_{(u,i) \in \mathcal{D}} c_{ui}(p_{ui} - \mathbf{p}_u^T\mathbf{q}_i)^2 + \lambda(\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2)$$

其中 $c_{ui}$ 是置信度权重，$p_{ui}$ 是偏好指示器（通常二值化）。

### 14.1.2 置信度矩阵的构建原理

置信度矩阵 $\mathbf{C}$ 的设计直接影响模型质量。经典的Hu等人提出的线性置信度函数：
$$c_{ui} = 1 + \alpha \cdot r_{ui}$$
其中 $r_{ui}$ 是原始交互强度（如播放次数）。

**高级置信度设计考虑因素**：

1. **时间衰减**：
   $$c_{ui}(t) = c_{ui} \cdot \exp(-\beta(t - t_{ui}))$$
   
2. **用户活跃度归一化**：
   $$\tilde{c}_{ui} = \frac{c_{ui}}{\sqrt{\sum_j c_{uj}}}$$
   
3. **物品流行度惩罚**：
   $$\hat{c}_{ui} = c_{ui} \cdot \left(\frac{N}{\text{count}(i)}\right)^\gamma$$

### 14.1.3 加权正则化的数学推导

标准L2正则化假设所有参数同等重要，但在隐式反馈中，我们需要考虑：

1. **用户侧加权正则化**：
   $$R_u(\mathbf{p}_u) = \lambda_u \|\mathbf{p}_u\|^2, \quad \lambda_u = \lambda \cdot n_u^\beta$$
   其中 $n_u$ 是用户交互数。

2. **物品侧自适应正则化**：
   $$R_i(\mathbf{q}_i) = \lambda_i \|\mathbf{q}_i - \mathbf{q}_i^{(0)}\|^2$$
   其中 $\mathbf{q}_i^{(0)}$ 是基于内容的先验嵌入。

**理论依据**：从贝叶斯角度，这相当于对参数施加不同方差的高斯先验：
$$p(\mathbf{p}_u) \propto \exp\left(-\frac{\lambda_u}{2}\|\mathbf{p}_u\|^2\right)$$

### 14.1.4 研究前沿：因果推断视角

最新研究将隐式反馈建模为带有选择偏差的因果推断问题：

1. **倾向分数加权**：
   $$\mathcal{L} = \sum_{(u,i) \in \mathcal{D}} \frac{1}{p(o_{ui}=1|u,i)} \ell(r_{ui}, \hat{r}_{ui})$$

2. **反事实风险最小化**：
   $$\mathcal{L}_{IPS} = \frac{1}{|\mathcal{D}|}\sum_{(u,i) \in \mathcal{D}} \frac{\ell(r_{ui}, \hat{r}_{ui})}{p(o_{ui}=1)} + \lambda R(\theta)$$

这为处理推荐系统中的偏差问题开辟了新方向。

## 14.2 ALS-WR算法的并行化与数值优化

### 14.2.1 交替最小二乘的计算复杂度分析

ALS-WR (Alternating Least Squares with Weighted Regularization) 的核心是交替固定一组参数，优化另一组。对于用户 $u$，更新方程为：

$$\mathbf{p}_u = (\mathbf{Q}^T\mathbf{C}^u\mathbf{Q} + \lambda\mathbf{I})^{-1}\mathbf{Q}^T\mathbf{C}^u\mathbf{p}(u)$$

其中 $\mathbf{C}^u$ 是用户 $u$ 的对角置信度矩阵。

**计算复杂度分解**：
- 朴素实现：$O(n^2k + nk^2 + k^3)$ per user
- 利用稀疏性：$O(n_u k^2 + k^3)$，其中 $n_u \ll n$

**关键优化**：预计算 $\mathbf{Q}^T\mathbf{Q}$，复杂度降至 $O(n_u k^2)$。

### 14.2.2 分布式ALS的通信模式

大规模部署需要精心设计的分布式策略：

1. **数据并行模式**：
   - 用户分片：每个节点负责一部分用户
   - 物品因子广播：$O(nk)$ 通信量
   - 适合用户数 $\gg$ 物品数的场景

2. **模型并行模式**：
   - 因子矩阵分块：$\mathbf{P} = [\mathbf{P}_1, \mathbf{P}_2, ..., \mathbf{P}_p]$
   - 通信量：$O(mk/p)$ per iteration
   - 适合超大规模因子维度

3. **混合并行策略**：
   ```
   将用户分成 G 组，每组内执行：
   1. 本地更新用户因子
   2. 收集局部物品梯度
   3. All-reduce 聚合梯度
   4. 更新物品因子
   ```

### 14.2.3 数值稳定性与收敛加速技巧

**数值稳定性保障**：

1. **条件数控制**：
   $$\kappa(\mathbf{A}) = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})} < \tau$$
   通过自适应正则化保证：$\lambda = \max(\lambda_0, \epsilon \cdot \|\mathbf{Q}^T\mathbf{C}^u\mathbf{Q}\|_F)$

2. **Cholesky分解的增量更新**：
   对于稀疏更新，使用 Sherman-Morrison 公式：
   $$(\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$$

**收敛加速技术**：

1. **动量方法**：
   $$\mathbf{p}_u^{(t+1)} = \mathbf{p}_u^{(t)} + \alpha \Delta\mathbf{p}_u^{(t)} + \beta(\mathbf{p}_u^{(t)} - \mathbf{p}_u^{(t-1)})$$

2. **自适应学习率**：
   基于局部Lipschitz常数估计：
   $$\alpha_u = \min\left(1, \frac{c}{\|\nabla_{\mathbf{p}_u} \mathcal{L}\|}\right)$$

3. **异步更新协议**：
   - Hogwild!风格的无锁更新
   - 延迟补偿：$\mathbf{p}_u^{(t)} = \mathbf{p}_u^{(t-\tau)} + \sum_{s=t-\tau}^{t-1} \Delta\mathbf{p}_u^{(s)}$

### 14.2.4 硬件感知优化

现代硬件架构需要算法层面的适配：

1. **缓存优化**：
   - 分块矩阵乘法：块大小 $b = \sqrt{\text{L3 cache} / (3 \times \text{sizeof}(float))}$
   - 列主序存储用于物品因子

2. **SIMD向量化**：
   - 内积计算使用 AVX-512：一次处理16个float
   - 对齐内存分配：`posix_memalign` 确保64字节对齐

3. **GPU加速考虑**：
   - Warp级别的负载均衡
   - 使用 cuBLAS 的 `cublasXgemmBatched` 批量小矩阵运算

## 14.3 负采样的数学原理与偏差校正

### 14.3.1 采样分布的理论选择

负采样是隐式反馈系统的核心技术。给定正样本集 $\mathcal{D}^+$，我们需要从未观察到的 $(u,i)$ 对中采样负例。

**流行的采样分布**：

1. **均匀采样**：
   $$P_{\text{unif}}(i|u) = \frac{1}{|\mathcal{I} \setminus \mathcal{I}_u|}$$
   简单但可能低估流行物品的负面信号。

2. **流行度比例采样**：
   $$P_{\text{pop}}(i|u) = \frac{f_i^\alpha}{\sum_{j \notin \mathcal{I}_u} f_j^\alpha}$$
   其中 $f_i$ 是物品频率，$\alpha \in [0,1]$ 控制偏斜程度。

3. **自适应采样**：
   基于当前模型预测：
   $$P_{\text{adapt}}(i|u) \propto \sigma(\mathbf{p}_u^T\mathbf{q}_i)$$
   类似于重要性采样，聚焦于"困难"负例。

### 14.3.2 重要性采样与无偏估计

使用非均匀采样时，需要校正引入的偏差：

**基础重要性采样**：
$$\mathbb{E}_{i \sim P}[\frac{Q(i)}{P(i)} \cdot \ell(u,i)] = \mathbb{E}_{i \sim Q}[\ell(u,i)]$$

**在负采样中的应用**：
$$\mathcal{L}_{\text{unbiased}} = \sum_{(u,i) \in \mathcal{D}^+} \ell^+(u,i) + \sum_{j=1}^{K} \frac{1}{P(i_j^-|u)} \ell^-(u,i_j^-)$$

其中 $i_j^- \sim P(\cdot|u)$ 是采样的负例。

**方差减少技术**：

1. **控制变量法**：
   $$\tilde{\ell}(u,i) = \ell(u,i) - c(\ell(u,i) - \mu)$$
   其中 $\mu = \mathbb{E}[\ell]$ 可以在线估计。

2. **分层采样**：
   将物品按流行度分桶，每桶独立采样：
   $$\mathcal{I} = \bigcup_{b=1}^{B} \mathcal{I}_b, \quad K_b = K \cdot \frac{|\mathcal{I}_b|}{|\mathcal{I}|}$$

### 14.3.3 自适应采样策略

**动态困难度感知采样**：

1. **基于梯度的重要性**：
   $$P(i|u) \propto \|\nabla_{\mathbf{q}_i} \ell(u,i)\|^2$$
   
2. **不确定性采样**：
   使用参数的后验分布：
   $$P(i|u) \propto \text{Var}[\mathbf{p}_u^T\mathbf{q}_i]$$

**多目标采样**：
同时优化多个目标（如覆盖度、多样性）：
$$P(i|u) = \sum_{k=1}^{K} w_k P_k(i|u)$$
其中权重 $w_k$ 可以动态调整。

### 14.3.4 理论分析：收敛性与样本复杂度

**收敛速率分析**：
在适当的假设下，使用 $K$ 个负样本的SGD收敛速率为：
$$\mathbb{E}[\|\mathbf{w}^{(t)} - \mathbf{w}^*\|^2] \leq \frac{C}{t} + \frac{D}{K}$$

其中第二项反映了负采样引入的偏差。

**样本复杂度界**：
达到 $\epsilon$-最优解所需的负样本数：
$$K = O\left(\frac{n \log n}{\epsilon^2}\right)$$

这解释了为什么实践中 $K=5-10$ 通常足够。

## 14.4 置信度加权的理论基础

### 14.4.1 贝叶斯视角下的置信度建模

从贝叶斯角度，置信度反映了我们对观察到的隐式信号的不确定性。

**生成模型**：
1. 真实偏好：$r_{ui}^* \sim \mathcal{N}(\mathbf{p}_u^T\mathbf{q}_i, \sigma^2)$
2. 观察过程：$r_{ui} = r_{ui}^* + \epsilon_{ui}$
3. 置信度：$c_{ui} = \frac{1}{\text{Var}[\epsilon_{ui}]}$

**后验推断**：
$$p(r_{ui}^*|r_{ui}) \propto \exp\left(-\frac{c_{ui}}{2}(r_{ui}^* - r_{ui})^2 - \frac{1}{2\sigma^2}(r_{ui}^* - \mathbf{p}_u^T\mathbf{q}_i)^2\right)$$

这导出了加权最小二乘的目标函数。

### 14.4.2 时间衰减与频率加权

**时间动态建模**：

1. **指数衰减模型**：
   $$c_{ui}(t) = c_{ui}^{(0)} \exp(-\lambda_t(t - t_{ui}))$$
   
2. **幂律衰减**：
   $$c_{ui}(t) = c_{ui}^{(0)} (1 + t - t_{ui})^{-\alpha}$$

**频率置信度**：
基于观察次数的置信度估计：
$$c_{ui} = \frac{n_{ui}}{n_{ui} + \beta}$$
这是贝塔-二项式模型的后验均值。

**联合建模**：
$$c_{ui}(t, n) = f_{\text{time}}(t) \cdot f_{\text{freq}}(n) \cdot f_{\text{context}}(\mathbf{x})$$

### 14.4.3 在线更新的一致性保证

**增量置信度更新**：
新观察到达时：
$$c_{ui}^{(t+1)} = \frac{c_{ui}^{(t)} \cdot n_{ui}^{(t)} + \Delta c_{ui}}{n_{ui}^{(t)} + 1}$$

**一致性条件**：
1. **单调性**：$c_{ui}^{(t+1)} \geq c_{ui}^{(t)}$ 当有新正交互
2. **有界性**：$c_{ui}^{(t)} \leq c_{\max}$ 防止过拟合
3. **收敛性**：$\lim_{t \to \infty} c_{ui}^{(t)} = c_{ui}^*$ 存在

**理论保证**：
在适当的正则化下，在线更新算法满足：
$$\text{Regret}(T) = O(\sqrt{T \log n})$$

### 14.4.4 多源置信度融合

现代系统需要整合多种信号：

1. **层次贝叶斯模型**：
   $$c_{ui} \sim \text{Gamma}(\alpha_s, \beta_s), \quad s \in \{\text{click}, \text{purchase}, ...\}$$

2. **注意力机制融合**：
   $$c_{ui} = \sum_{s} a_s(\mathbf{u}, \mathbf{i}) \cdot c_{ui}^{(s)}$$
   其中 $a_s$ 是学习的注意力权重。

3. **元学习置信度**：
   使用MAML框架学习置信度函数：
   $$c_{ui} = f_\phi(\mathbf{x}_{ui}), \quad \phi^* = \arg\min_\phi \mathbb{E}_{\mathcal{T}}[\mathcal{L}_{\mathcal{T}}(f_\phi)]$$

## 本章小结

本章深入探讨了大规模协同过滤中的核心矩阵技术。主要贡献包括：

1. **隐式反馈的数学建模**：
   - 置信度加权框架：$\min \sum_{(u,i)} c_{ui}(p_{ui} - \mathbf{p}_u^T\mathbf{q}_i)^2$
   - 从显式评分到隐式反馈的范式转变
   - 多源信号的统一处理框架

2. **ALS-WR的工程化实现**：
   - 计算复杂度从 $O(mnk)$ 降至 $O((m+n)n_z k^2)$
   - 分布式并行的三种模式及其权衡
   - 数值稳定性保障与收敛加速

3. **负采样的理论基础**：
   - 重要性采样实现无偏估计
   - 样本复杂度界：$K = O(\frac{n \log n}{\epsilon^2})$
   - 自适应采样策略的设计原则

4. **置信度的贝叶斯解释**：
   - 不确定性量化的原则性方法
   - 时间动态与多源融合
   - 在线学习的一致性保证

**关键洞察**：
- 隐式反馈的核心挑战是**缺失并非随机**（MNAR）
- 置信度加权提供了处理不确定性的统一框架
- 负采样效率决定了算法的可扩展性
- 硬件感知的优化对性能至关重要

**未来研究方向**：
1. 因果推断方法处理选择偏差
2. 神经网络与矩阵分解的深度融合
3. 联邦学习场景下的隐私保护矩阵分解
4. 量子算法加速超大规模矩阵运算

## 练习题

### 练习 14.1：置信度函数设计
考虑一个视频推荐系统，用户行为包括：点击（click）、观看时长（watch_time）、点赞（like）、分享（share）。设计一个综合置信度函数 $c_{ui}$，满足以下要求：
1. 不同行为类型有不同权重
2. 考虑时间衰减
3. 防止极端值

**提示**：考虑使用对数变换处理长尾分布的观看时长。

<details>
<summary>答案</summary>

一个可能的设计：
$$c_{ui} = 1 + \alpha_1 \cdot \mathbb{1}[\text{click}] + \alpha_2 \cdot \log(1 + \text{watch\_time}) + \alpha_3 \cdot \mathbb{1}[\text{like}] + \alpha_4 \cdot \mathbb{1}[\text{share}]$$

加入时间衰减：
$$c_{ui}(t) = c_{ui} \cdot \exp(-\lambda(t - t_{ui}))$$

防止极端值：
$$\tilde{c}_{ui} = \min(c_{ui}, c_{\max}) \cdot \frac{c_{ui}}{c_{ui} + \beta}$$

其中参数可以通过交叉验证确定，典型值：$\alpha_1 = 1, \alpha_2 = 0.5, \alpha_3 = 3, \alpha_4 = 5, \lambda = 0.01, \beta = 10, c_{\max} = 100$。
</details>

### 练习 14.2：ALS更新的向量化实现
给定用户 $u$ 的交互物品集合 $\mathcal{I}_u = \{i_1, i_2, ..., i_{n_u}\}$ 和对应置信度 $\{c_{ui_1}, c_{ui_2}, ..., c_{ui_{n_u}}\}$，推导向量化的用户因子更新公式，避免显式构造对角矩阵 $\mathbf{C}^u$。

**提示**：利用 $\mathbf{Q}^T\mathbf{C}^u\mathbf{Q} = \sum_{i \in \mathcal{I}_u} c_{ui}\mathbf{q}_i\mathbf{q}_i^T + \mathbf{Q}^T\mathbf{Q}$。

<details>
<summary>答案</summary>

向量化更新公式：
$$\mathbf{A}_u = \mathbf{Q}^T\mathbf{Q} + \sum_{i \in \mathcal{I}_u} (c_{ui} - 1)\mathbf{q}_i\mathbf{q}_i^T + \lambda\mathbf{I}$$

$$\mathbf{b}_u = \sum_{i \in \mathcal{I}_u} c_{ui} p_{ui} \mathbf{q}_i$$

$$\mathbf{p}_u = \mathbf{A}_u^{-1}\mathbf{b}_u$$

计算技巧：
1. 预计算 $\mathbf{Q}^T\mathbf{Q}$ 一次，所有用户共享
2. 使用 BLAS 的 `syr` 操作高效计算秩-1更新
3. 对于稀疏 $\mathcal{I}_u$，复杂度为 $O(|\mathcal{I}_u|k^2 + k^3)$
</details>

### 练习 14.3：负采样的方差分析
证明：使用流行度比例采样 $P(i) \propto f_i^{0.75}$ 相比均匀采样，在估计全局损失时具有更低的方差。假设损失函数为 $\ell(u,i) = (\mathbf{p}_u^T\mathbf{q}_i)^2$。

**提示**：计算两种采样策略下估计量的方差，并比较。

<details>
<summary>答案</summary>

设真实损失为 $L = \sum_{i \notin \mathcal{I}_u} \ell(u,i)$。

均匀采样估计量：
$$\hat{L}_{\text{unif}} = \frac{|\mathcal{I} \setminus \mathcal{I}_u|}{K} \sum_{j=1}^K \ell(u,i_j)$$

其方差：
$$\text{Var}[\hat{L}_{\text{unif}}] = \frac{|\mathcal{I} \setminus \mathcal{I}_u|^2}{K} \text{Var}_{i \sim \text{Unif}}[\ell(u,i)]$$

流行度采样估计量：
$$\hat{L}_{\text{pop}} = \sum_{j=1}^K \frac{\ell(u,i_j)}{P(i_j)}$$

其方差：
$$\text{Var}[\hat{L}_{\text{pop}}] = \frac{1}{K} \sum_{i \notin \mathcal{I}_u} \frac{\ell(u,i)^2}{P(i)} - \frac{L^2}{K}$$

当 $\ell(u,i)$ 与 $f_i$ 正相关时（流行物品得分更高），选择 $P(i) \propto f_i^{0.75}$ 使得 $\ell(u,i)/P(i)$ 更均匀，从而降低方差。
</details>

### 练习 14.4：分布式ALS的通信优化
在分布式ALS中，假设有 $p$ 个计算节点，$m$ 个用户，$n$ 个物品，因子维度 $k$。分析以下三种数据分片策略的通信复杂度：
1. 随机分片
2. 基于图分割的分片
3. 块循环分片

**提示**：考虑每轮迭代需要传输的因子矩阵大小。

<details>
<summary>答案</summary>

1. **随机分片**：
   - 每个节点需要几乎所有物品因子
   - 通信量：$O(pnk)$ per iteration
   
2. **基于图分割**：
   - 最小化跨分区边
   - 设边割比例为 $\rho$
   - 通信量：$O(\rho nk)$，其中 $\rho < 1$
   
3. **块循环分片**：
   - 用户和物品都分块
   - 每个节点负责 $\frac{m}{p} \times \frac{n}{p}$ 的块
   - 通信量：$O(\frac{(m+n)k}{\sqrt{p}})$
   
结论：对于大规模系统，块循环分片在 $p$ 较大时最优。
</details>

### 练习 14.5：增量更新的误差累积（挑战题）
在流式场景中，使用增量Sherman-Morrison更新代替完全重算。分析经过 $T$ 次更新后的数值误差累积，并提出误差控制策略。

**提示**：考虑浮点运算的舍入误差和条件数的影响。

<details>
<summary>答案</summary>

Sherman-Morrison更新：
$$\mathbf{A}^{-1}_{t+1} = \mathbf{A}^{-1}_t - \frac{\mathbf{A}^{-1}_t \mathbf{u}\mathbf{v}^T \mathbf{A}^{-1}_t}{1 + \mathbf{v}^T \mathbf{A}^{-1}_t \mathbf{u}}$$

误差分析：
- 每步相对误差：$\epsilon_{\text{rel}} \approx \kappa(\mathbf{A}) \cdot \epsilon_{\text{machine}}$
- $T$ 步后：$\epsilon_T \approx T \cdot \kappa(\mathbf{A}) \cdot \epsilon_{\text{machine}}$

误差控制策略：
1. **周期性重算**：每 $T_0 = \frac{1}{\kappa(\mathbf{A}) \cdot \epsilon_{\text{machine}}}$ 步完全重算
2. **条件数监控**：当 $\kappa(\mathbf{A}_t) > \tau$ 时触发重算
3. **残差检验**：检查 $\|\mathbf{A}_t \mathbf{A}_t^{-1} - \mathbf{I}\|_F < \delta$
</details>

### 练习 14.6：多目标优化的帕累托前沿（挑战题）
推荐系统需要同时优化准确性（RMSE）、覆盖度（Coverage）和多样性（Diversity）。设计一个多目标矩阵分解框架，并分析帕累托最优解的性质。

**提示**：考虑加权和方法和约束优化方法。

<details>
<summary>答案</summary>

多目标优化框架：
$$\min_{\mathbf{P},\mathbf{Q}} \alpha_1 \cdot \text{RMSE} + \alpha_2 \cdot (1-\text{Coverage}) + \alpha_3 \cdot (1-\text{Diversity})$$

其中：
- $\text{RMSE} = \sqrt{\frac{1}{|\mathcal{D}|}\sum_{(u,i) \in \mathcal{D}} (r_{ui} - \mathbf{p}_u^T\mathbf{q}_i)^2}$
- $\text{Coverage} = \frac{|\{i: \exists u, \mathbf{p}_u^T\mathbf{q}_i > \theta\}|}{n}$
- $\text{Diversity} = 1 - \frac{1}{|\mathcal{U}|}\sum_u \frac{\sum_{i,j \in \text{Top-K}(u)} \text{sim}(i,j)}{K(K-1)}$

帕累托前沿特性：
1. 凸性：在温和假设下，帕累托前沿局部凸
2. 权衡关系：$\frac{\partial \text{Coverage}}{\partial \text{RMSE}} < 0$
3. 锚点解：纯准确性优化 vs 纯多样性优化

实践建议：使用进化算法（如NSGA-II）探索帕累托前沿。
</details>

### 练习 14.7：联邦学习中的隐私保护矩阵分解（开放题）
设计一个满足差分隐私的分布式矩阵分解算法，其中用户数据不能离开本地设备。分析隐私-效用权衡。

**提示**：考虑梯度扰动和安全聚合。

<details>
<summary>答案</summary>

算法框架：
1. **本地更新**：用户 $u$ 在设备上计算：
   $$\mathbf{p}_u^{(t+1)} = \mathbf{p}_u^{(t)} - \eta \nabla_{\mathbf{p}_u} \mathcal{L}_u$$

2. **梯度扰动**：添加高斯噪声实现 $(\epsilon, \delta)$-差分隐私：
   $$\tilde{\nabla} = \nabla + \mathcal{N}(0, \sigma^2 \mathbf{I})$$
   其中 $\sigma = \frac{2\Delta_f \sqrt{2\log(1.25/\delta)}}{\epsilon}$

3. **安全聚合**：
   使用秘密分享或同态加密聚合物品梯度：
   $$\nabla_{\mathbf{q}_i} = \sum_{u \in \mathcal{U}_i} \tilde{\nabla}_{u,i}$$

隐私-效用分析：
- 隐私损失：$\epsilon_{\text{total}} = \sqrt{2T\log(1/\delta)} \cdot \epsilon$
- 效用损失：$\text{RMSE}_{\text{private}} - \text{RMSE}_{\text{non-private}} \approx O(\frac{k\sqrt{T}}{\epsilon n})$

研究方向：
1. 自适应隐私预算分配
2. 个性化隐私级别
3. 与同态加密结合减少噪声
</details>

### 练习 14.8：因果推断视角的去偏（研究题）
将推荐系统的选择偏差建模为因果图，使用do-calculus推导无偏的损失函数。考虑观察概率 $P(O=1|U,I,R)$ 依赖于真实评分 $R$ 的情况。

**提示**：构建因果图 $U \rightarrow R \leftarrow I, R \rightarrow O$。

<details>
<summary>答案</summary>

因果模型：
- $U$：用户特征
- $I$：物品特征
- $R$：真实评分
- $O$：是否观察到

目标：估计 $\mathbb{E}[R|U,I]$

观察到的数据分布：
$$P(R|U,I,O=1) = \frac{P(O=1|U,I,R)P(R|U,I)}{P(O=1|U,I)}$$

使用逆倾向分数加权：
$$\mathbb{E}[R|U,I] = \frac{\mathbb{E}[R \cdot \mathbb{1}[O=1] / P(O=1|U,I,R)|U,I]}{\mathbb{E}[\mathbb{1}[O=1] / P(O=1|U,I,R)|U,I]}$$

实践中的挑战：
1. 估计 $P(O=1|U,I,R)$ 需要反事实推理
2. 倾向分数的极值导致高方差
3. 模型误设的敏感性

研究方向：
- 双鲁棒估计
- 敏感性分析
- 与强化学习的结合
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 数值稳定性陷阱

**问题**：直接计算 $\mathbf{Q}^T\mathbf{C}^u\mathbf{Q}$ 导致数值不稳定
```
# 错误做法
A = Q.T @ diag(C_u) @ Q  # 当C_u包含极大值时条件数爆炸
```

**正确做法**：
- 使用增量更新：$\mathbf{A} = \mathbf{Q}^T\mathbf{Q} + \sum_i (c_{ui}-1)\mathbf{q}_i\mathbf{q}_i^T$
- 添加自适应正则化：$\lambda = \max(\lambda_0, \epsilon \cdot \text{trace}(\mathbf{A}))$

### 2. 负采样偏差

**问题**：均匀负采样导致流行物品的负信号不足
- 热门物品被大量用户忽略，但采样概率与冷门物品相同
- 导致模型过度推荐流行物品

**解决方案**：
- 使用流行度加权采样：$P(i) \propto f_i^{0.75}$
- 实施重要性权重校正

### 3. 冷启动过拟合

**问题**：新用户/物品的因子向量快速过拟合到少量交互
- 置信度函数对初始交互赋予过高权重
- 正则化不足导致因子向量norm过大

**缓解策略**：
- 动态正则化：$\lambda_u = \lambda_0 \cdot \max(1, n_0/n_u)$
- 使用先验信息初始化因子

### 4. 分布式训练的数据倾斜

**问题**：用户活跃度的幂律分布导致计算负载不均
- 某些节点处理超级活跃用户，成为瓶颈
- 简单的哈希分片加剧问题

**解决方案**：
- 基于计算量的动态负载均衡
- 将超级用户的计算分散到多个节点

### 5. 时间动态的过度拟合

**问题**：过于激进的时间衰减导致历史信息丢失
- 指数衰减参数设置过大
- 短期行为主导，长期偏好被忽略

**平衡方法**：
- 使用多尺度时间建模
- 保留用户的"核心偏好"不衰减

### 6. 隐式到显式的误导转换

**问题**：将隐式反馈二值化时丢失重要信息
```
# 危险的简化
p_ui = 1 if r_ui > 0 else 0  # 忽略了交互强度
```

**更好的做法**：
- 保留连续值：$p_{ui} = \log(1 + r_{ui})$
- 使用置信度区分不同强度的信号

### 7. 并行更新的竞态条件

**问题**：Hogwild!式并行更新在高竞争区域失效
- 热门物品的因子被频繁更新
- 梯度延迟导致收敛不稳定

**缓解措施**：
- 使用延迟补偿SGD
- 对高竞争参数使用锁或原子操作

### 8. 评估指标的误导性

**问题**：离线指标（如RMSE）与在线业务指标脱节
- RMSE优化可能降低推荐多样性
- 忽略了位置偏差和展示偏差

**全面评估**：
- 同时监控准确性、覆盖度、多样性
- 进行在线A/B测试验证

## 最佳实践检查清单

### 数据预处理
- [ ] 识别并处理异常用户行为（如机器人）
- [ ] 对长尾分布的交互强度进行对数变换
- [ ] 实施合理的时间窗口截断
- [ ] 验证用户-物品交互矩阵的稀疏度

### 算法设计
- [ ] 置信度函数考虑多种信号类型
- [ ] 正则化参数根据用户/物品活跃度自适应
- [ ] 负采样策略平衡效率与无偏性
- [ ] 实现增量更新以支持在线学习

### 数值计算
- [ ] 使用数值稳定的矩阵运算（避免显式求逆）
- [ ] 监控条件数，必要时增加正则化
- [ ] 利用稀疏性减少计算复杂度
- [ ] 实施周期性的完全重算以控制误差累积

### 分布式实现
- [ ] 选择适合数据特征的分片策略
- [ ] 实现高效的因子广播/聚合机制
- [ ] 处理节点故障和网络延迟
- [ ] 监控和均衡计算负载

### 性能优化
- [ ] 利用SIMD指令加速向量运算
- [ ] 优化内存访问模式（缓存友好）
- [ ] 使用异步I/O减少等待时间
- [ ] Profile识别性能瓶颈

### 模型评估
- [ ] 设计考虑时间因素的训练/测试集划分
- [ ] 评估多个维度：准确性、覆盖度、多样性、新颖性
- [ ] 进行消融实验验证各组件贡献
- [ ] 监控模型在不同用户群体上的公平性

### 生产部署
- [ ] 实现模型版本管理和回滚机制
- [ ] 设置异常检测和报警
- [ ] 支持热更新without服务中断
- [ ] 记录详细日志用于问题诊断

### 持续改进
- [ ] 收集在线反馈用于模型迭代
- [ ] 定期重新评估超参数
- [ ] 探索新的特征和信号源
- [ ] 跟踪学术界最新进展并评估适用性