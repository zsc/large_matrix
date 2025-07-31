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

**张量构建的关键决策**：

1. **时间离散化策略**：
   - **固定窗口**：将连续时间划分为等长区间
   - **自适应窗口**：根据交互密度动态调整窗口大小
   - **多尺度建模**：同时维护小时、天、周等多个粒度的张量

2. **缺失值处理**：
   - **显式缺失**：未观测位置不参与计算
   - **隐式负反馈**：采用加权策略，$w_{uit} = \alpha + \beta \cdot \text{confidence}_{uit}$
   - **时间衰减填充**：$\hat{r}_{uit} = \sum_{s<t} \gamma^{t-s} r_{uis}$

时间粒度的选择至关重要：
- **细粒度**（小时级）：捕获日内模式，如午餐时段的外卖推荐
- **中粒度**（天级）：捕获周内模式，如周末vs工作日的差异
- **粗粒度**（周/月级）：捕获季节性趋势和长期兴趣演化

时间建模的高级技巧：
- **循环时间编码**：将时间映射到周期空间，如 $(\sin(2\pi t/T), \cos(2\pi t/T))$
- **多尺度融合**：同时建模多个时间粒度，使用耦合张量分解
- **时间衰减权重**：$w_t = \exp(-\lambda(t_{\text{now}} - t))$ 强调近期交互
- **变点检测**：自动识别用户行为模式的突变点
- **季节性分解**：$\mathcal{R} = \mathcal{R}_{\text{trend}} + \mathcal{R}_{\text{seasonal}} + \mathcal{R}_{\text{residual}}$

**时间张量的特殊结构**：

1. **Toeplitz结构**：当交互模式具有时间不变性时
   $$\mathcal{R}_{:,:,t} \approx f(\mathcal{R}_{:,:,t-1}, \mathcal{R}_{:,:,t-2}, ...)$$

2. **低秩加稀疏分解**：
   $$\mathcal{R} = \mathcal{L} + \mathcal{S}$$
   其中 $\mathcal{L}$ 捕获稳定模式，$\mathcal{S}$ 捕获异常事件（如促销、节假日）

3. **动态张量补全**：
   $$\min_{\mathcal{X}} \|\mathcal{P}_\Omega(\mathcal{X} - \mathcal{R})\|_F^2 + \lambda_1\|\mathcal{X}\|_* + \lambda_2\sum_{t}\|\mathcal{X}_{:,:,t} - \mathcal{X}_{:,:,t-1}\|_F^2$$
   第三项强制时间平滑性。

**案例2：多模态内容推荐**
对于包含文本、图像、视频的多模态物品，构建四阶张量：
$$\mathcal{X} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}| \times |\mathcal{M}| \times |\mathcal{F}|}$$
其中 $\mathcal{M}$ 是模态类型，$\mathcal{F}$ 是特征维度。

模态融合的关键考虑：
- **早期融合**：在张量层面直接建模多模态交互
- **晚期融合**：分别对每个模态进行张量分解，然后融合结果
- **注意力机制**：学习用户对不同模态的偏好权重

**多模态张量的构建方法**：

1. **特征对齐**：
   - 使用预训练模型（CLIP、BERT等）提取统一维度的特征
   - 跨模态对齐：$\mathbf{f}_{\text{aligned}} = \text{CCA}(\mathbf{f}_{\text{text}}, \mathbf{f}_{\text{image}})$
   - 维度规范化：将不同模态映射到相同的特征空间

2. **模态交互建模**：
   $$\mathcal{X}_{uimf} = \sum_{k=1}^{K} \alpha_{umk} \cdot \beta_{ik} \cdot \gamma_{fk}$$
   其中 $\alpha_{umk}$ 表示用户 $u$ 对模态 $m$ 中第 $k$ 个潜在因子的偏好。

3. **缺失模态处理**：
   - **零填充**：简单但可能引入偏差
   - **均值填充**：使用同类物品的平均特征
   - **生成填充**：训练跨模态生成模型填补缺失

**高级建模技巧**：

1. **模态特定的正则化**：
   $$\mathcal{L} = \|\mathcal{X} - \hat{\mathcal{X}}\|_F^2 + \sum_{m} \lambda_m \mathcal{R}_m(\mathbf{U}^{(m)})$$
   其中 $\mathcal{R}_m$ 是模态特定的正则化器（如文本的稀疏性、图像的平滑性）。

2. **分层张量结构**：
   - 顶层：用户-物品-模态交互
   - 中层：模态内的特征交互
   - 底层：原始特征表示

3. **动态模态选择**：
   根据用户历史行为动态调整不同模态的重要性：
   $$w_{um}^{(t)} = \frac{\exp(\mathbf{u}_u^T \mathbf{m}_m + b_{um}^{(t)})}{\sum_{m'} \exp(\mathbf{u}_u^T \mathbf{m}_{m'} + b_{um'}^{(t)})}$$

**案例3：社交推荐网络**
五阶张量建模 "谁-什么-何时-何地-与谁"：
$$\mathcal{Y} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}| \times |\mathcal{T}| \times |\mathcal{L}| \times |\mathcal{G}|}$$

其中：
- $\mathcal{L}$：地理位置（POI推荐）
- $\mathcal{G}$：社交群组或共同参与者

这种高阶建模能够捕获复杂的情境依赖：
- 和朋友一起时的电影偏好 vs 独自观看
- 工作地点的午餐选择 vs 家附近的晚餐选择
- 周末聚会的音乐品味 vs 日常通勤的播放列表

**社交张量的特殊挑战**：

1. **超高维数的处理**：
   - 维度诅咒加剧：$O(\prod_{i=1}^{5} d_i)$ 的参数量
   - 解决方案：使用张量列车分解（Tensor Train Decomposition）
   $$\mathcal{Y}_{i_1,i_2,i_3,i_4,i_5} \approx \mathbf{G}_1(i_1)\mathbf{G}_2(i_2)\mathbf{G}_3(i_3)\mathbf{G}_4(i_4)\mathbf{G}_5(i_5)$$
   其中 $\mathbf{G}_k$ 是三阶核张量（除了边界）。

2. **社交影响力建模**：
   $$r_{uitlg} = \alpha \cdot r_{uit}^{\text{personal}} + \beta \cdot \sum_{v \in g} w_{uv} \cdot r_{vit}^{\text{social}}$$
   其中 $w_{uv}$ 是社交影响力权重，可以通过图神经网络学习。

3. **群体决策建模**：
   - **平均策略**：$\hat{r}_g = \frac{1}{|g|}\sum_{u \in g} r_u$
   - **最小痛苦策略**：$\hat{r}_g = \min_{u \in g} r_u$
   - **灵活权衡**：$\hat{r}_g = \prod_{u \in g} r_u^{\theta_u}$，其中 $\theta_u$ 是用户在群体中的决策权重

4. **时空依赖性**：
   $$P(\text{visit } l | u, t, g) = \frac{\exp(\mathbf{u}_u^T \mathbf{l}_l + \mathbf{t}_t^T \mathbf{l}_l + \sum_{v \in g} \mathbf{v}_v^T \mathbf{l}_l)}{\sum_{l'} \exp(\cdot)}$$

**高效近似方法**：

1. **分层分解**：
   - 先分解低阶子张量：$\mathcal{Y}_{:,:,:} = \text{CP}(\mathcal{U}, \mathcal{I}, \mathcal{T})$
   - 再建模高阶交互：$\mathcal{Y}_{:,:,:,l,g} = f(\mathcal{Y}_{:,:,:}, \mathbf{L}_l, \mathbf{G}_g)$

2. **采样策略**：
   - 重要性采样：根据交互频率采样
   - 负采样：从未观测的维度组合中采样

**案例4：多任务学习张量**
对于需要同时优化多个目标的推荐系统（点击率、转化率、停留时间等）：
$$\mathcal{Z} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}| \times |\mathcal{T}| \times |\mathcal{O}|}$$

其中 $\mathcal{O}$ 是目标函数集合。这允许：
- 发现不同目标间的相关性模式
- 通过张量分解实现多任务知识共享
- 为不同业务目标动态调整推荐策略

**多任务张量分解的数学框架**：

1. **联合优化目标**：
   $$\min_{\{\mathbf{U}^{(n,o)}\}} \sum_{o \in \mathcal{O}} \lambda_o \|\mathcal{Z}_{:,:,:,o} - \text{CP}(\mathbf{U}^{(1,o)}, \mathbf{U}^{(2,o)}, \mathbf{U}^{(3,o)})\|_F^2$$
   
   其中部分因子矩阵在任务间共享：
   $$\mathbf{U}^{(n,o)} = \mathbf{U}^{(n)}_{\text{shared}} + \mathbf{U}^{(n,o)}_{\text{specific}}$$

2. **任务相关性建模**：
   $$\mathbf{R}_{\mathcal{O}} = \text{corr}(\mathcal{Z}_{:,:,:,o_1}, \mathcal{Z}_{:,:,:,o_2})$$
   
   使用任务相关性矩阵指导共享程度：
   - 高相关任务：更多共享参数
   - 负相关任务：独立参数或对抗学习

3. **Pareto最优平衡**：
   不同于简单的加权和，寻找Pareto最优解：
   $$\mathcal{P} = \{\theta : \nexists \theta' \text{ s.t. } L_o(\theta') \leq L_o(\theta) \, \forall o \text{ and } L_{o'}(\theta') < L_{o'}(\theta) \text{ for some } o'\}$$

4. **动态权重调整**：
   $$\lambda_o^{(t)} = \lambda_o^{(0)} \cdot \exp\left(-\frac{\nabla L_o}{\|\nabla L_o\|_2}\right)$$
   
   梯度范数较大的任务获得更高权重。

**实际应用中的优化技巧**：

1. **分阶段训练**：
   - 阶段1：独立训练各任务获得初始化
   - 阶段2：联合微调共享参数
   - 阶段3：精细调整任务特定参数

2. **不确定性量化**：
   使用贝叶斯张量分解量化预测不确定性：
   $$p(\mathcal{Z}|\mathcal{X}) = \int p(\mathcal{Z}|\mathcal{U})p(\mathcal{U}|\mathcal{X})d\mathcal{U}$$

3. **在线更新**：
   当新任务加入时，仅更新相关的张量切片：
   $$\mathcal{Z}_{:,:,:,o_{\text{new}}} = \text{transfer\_learning}(\mathcal{Z}_{:,:,:,o_{\text{similar}}})$$

### 16.1.4 稀疏性挑战与机遇

推荐系统中的张量极度稀疏，观测率通常低于 0.01%。这带来了计算和统计两方面的挑战：

**稀疏性的根本原因**：
1. **长尾分布**：大部分用户只与少数物品交互
2. **时空局部性**：用户在特定时间和地点的活动有限
3. **选择性偏差**：用户倾向于与感兴趣的内容交互
4. **平台限制**：推荐系统只展示有限选项

**计算挑战**：
- 存储开销：稠密存储不现实
- 运算效率：大量零元素参与计算
- 内存访问：随机访问模式导致cache miss
- 负载均衡：非零元素分布不均导致并行效率低下

**统计挑战**：
- 过拟合风险：参数远多于观测
- 冷启动问题：新用户/物品缺乏数据
- 负采样偏差：未观测不等于负样本
- 噪声敏感性：稀疏数据中的异常值影响更大

然而，稀疏性也带来机遇：
- 低秩假设更可能成立
- 可利用隐式正则化
- 计算可大幅优化
- 支持在线增量学习

**稀疏性的数学视角**：

1. **核范数正则化的隐式偏好**：
   稀疏观测下的张量补全等价于：
   $$\min_{\mathcal{X}} \|\mathcal{P}_\Omega(\mathcal{X} - \mathcal{M})\|_F^2 + \lambda \|\mathcal{X}\|_*$$
   其中 $\|\cdot\|_*$ 是张量核范数，$\mathcal{P}_\Omega$ 是观测位置的投影算子。

2. **采样复杂度界**：
   对于秩为 $r$ 的 $n_1 \times n_2 \times n_3$ 张量，可靠恢复所需的样本数：
   $$m \geq C \cdot r \cdot (n_1 + n_2 + n_3) \cdot \log^2(n_1 n_2 n_3)$$
   这远小于总元素数 $n_1 n_2 n_3$。

3. **相干性（Coherence）条件**：
   成功恢复需要张量的奇异向量不能过度集中：
   $$\mu(\mathcal{X}) = \max_i \frac{n}{r} \|\mathbf{u}_i\|_\infty^2 \leq \mu_0$$
   高相干性（如某些用户特别活跃）会增加恢复难度。
   
   **相干性的实际含义**：
   - **低相干性**：信息均匀分布，易于恢复
   - **高相干性**：信息集中在少数维度，难以恢复
   - **处理方法**：重加权采样、正则化约束

**实用的稀疏性处理策略**：

1. **自适应采样**：
   - 基于不确定性的主动学习采样
   - leverage score采样提高信息量
   - 重要性采样减少方差

2. **辅助信息利用**：
   - 引入用户/物品特征作为side information
   - 利用隐式反馈（浏览、收藏等）
   - 迁移学习从相关域借力

3. **稀疏性感知的优化**：
   - 只在观测位置计算损失和梯度
   - 使用coordinate descent减少计算
   - 采用importance sampling加速收敛

4. **鲁棒性增强**：
   - Huber损失处理异常值
   - 矩阵补全的鲁棒PCA扩展
   - 贝叶斯方法量化不确定性

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

**数值稳定性保证**：

1. **正则化的ALS更新**：
   $$\mathbf{A}^{(n)} = \mathbf{X}_{(n)} \mathbf{Z}^{(n)} (\mathbf{V}^{(n)} + \lambda \mathbf{I})^{-1}$$
   
   选择 $\lambda$ 的自适应策略：
   - 基于条件数：$\lambda = \epsilon \cdot \sigma_\max(\mathbf{V}^{(n)})$
   - 基于噪声水平：$\lambda = \sigma^2 / \text{SNR}$
   - 贝叶斯视角：$\lambda$ 对应先验强度

2. **QR分解稳定化**：
   当 $\mathbf{V}^{(n)}$ 接近奇异时，使用QR分解：
   ```
   [Q, R] = qr([Z^(n); sqrt(λ)I], 0)
   A^(n) = X_(n) * (Q(1:end-R,:) / R)
   ```

3. **行归一化策略**：
   每次迭代后归一化因子矩阵，将尺度吸收到 $\lambda_r$：
   $$\mathbf{a}_r^{(n)} \leftarrow \frac{\mathbf{a}_r^{(n)}}{\|\mathbf{a}_r^{(n)}\|}, \quad \lambda_r \leftarrow \lambda_r \prod_n \|\mathbf{a}_r^{(n)}\|$$

**收敛加速技术**：

1. **线搜索**：
   不直接接受ALS更新，而是寻找最优步长：
   $$\mathbf{A}^{(n)}_{\text{new}} = (1-\alpha)\mathbf{A}^{(n)}_{\text{old}} + \alpha \mathbf{A}^{(n)}_{\text{ALS}}$$

2. **动量方法**：
   借鉴深度学习的动量技巧：
   $$\mathbf{V}_t^{(n)} = \beta \mathbf{V}_{t-1}^{(n)} + (1-\beta)\nabla f_t$$
   $$\mathbf{A}_t^{(n)} = \mathbf{A}_{t-1}^{(n)} - \eta \mathbf{V}_t^{(n)}$$

3. **Anderson加速**：
   利用历史迭代信息构造更好的更新方向，特别适合ALS这类定点迭代。

**大规模实现优化**：

1. **分块更新**：
   将因子矩阵 $\mathbf{A}^{(n)}$ 按行分块，每次只更新一个块：
   - 减少内存需求
   - 提高缓存利用率
   - 支持在线学习

2. **采样ALS**：
   每次迭代只使用部分非零元素：
   - 均匀采样：简单但可能有偏
   - 重要性采样：基于leverage score
   - 分层采样：保证每个用户/物品被采样

3. **异步ALS**：
   不等待所有模式更新完成：
   - Hogwild!风格的无锁更新
   - 延迟补偿机制
   - 理论保证：在温和条件下收敛

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

**高级优化技术**：

1. **增量SVD更新**：
   不每次都计算完整SVD，而是使用增量方法：
   ```
   已有：UΣV^T ≈ Y_(n)
   新数据到达：ΔY
   更新：[U', Σ', V'] = incremental_svd(U, Σ, V, ΔY)
   ```
   
   复杂度从 $O(I_n^2 J_n)$ 降至 $O(I_n R_n^2)$，其中 $J_n = \prod_{k \neq n} I_k$。

2. **随机化HOOI**：
   使用随机投影加速SVD计算：
   ```
   Ω = randn(J_n, k)  # k = R_n + oversampling
   Q = orth(Y_(n) * Ω)
   B = Q^T * Y_(n)
   [U_B, S, V] = svd(B)
   U^(n) = Q * U_B(:, 1:R_n)
   ```

3. **块Tucker分解**：
   将大张量分割成小块，分别处理：
   $$\mathcal{X} = \sum_{b=1}^{B} \mathcal{X}_b$$
   $$\mathcal{X}_b \approx \mathcal{G}_b \times_1 \mathbf{U}_b^{(1)} \times_2 \mathbf{U}_b^{(2)} \times_3 \mathbf{U}_b^{(3)}$$
   
   优势：
   - 每个块可并行处理
   - 内存需求大幅降低
   - 支持流式处理

4. **自适应秩选择**：
   不需要预先指定 $(R_1, R_2, ..., R_N)$：
   ```
   对于每个模式 n:
       计算奇异值衰减率
       R_n = argmin{r: Σ_{i=1}^r s_i^2 / Σ_{i=1}^{I_n} s_i^2 > θ}
   ```
   
   其中 $\theta$ 是能量保留阈值（如 0.95）。

**稀疏Tucker分解**：

1. **稀疏核心张量**：
   在核心张量上施加稀疏性约束：
   $$\min_{\mathcal{G}, \{\mathbf{U}^{(n)}\}} \|\mathcal{X} - \mathcal{G} \times \{\mathbf{U}^{(n)}\}\|_F^2 + \lambda \|\mathcal{G}\|_1$$
   
   使用软阈值算子：$\mathcal{G} \leftarrow \text{soft}(\mathcal{G}, \lambda)$

2. **稀疏因子矩阵**：
   促进因子矩阵的列稀疏性：
   $$\min \|\mathcal{X} - \text{tucker}(\mathcal{G}, \{\mathbf{U}^{(n)}\})\|_F^2 + \sum_n \lambda_n \|\mathbf{U}^{(n)}\|_{2,1}$$
   
   其中 $\|\cdot\|_{2,1}$ 是组LASSO范数。

**并行化策略**：

1. **模式并行**：
   ```
   parallel for n = 1:N
       计算 Y^(n) = X × {U^(k)^T}_{k≠n}
       更新 U^(n) = svd(Y^(n)_(n), R_n)
   end
   ```
   
   注意：需要同步以保证一致性。

2. **数据并行**：
   将张量按某个模式分割：
   ```
   X = [X_1; X_2; ...; X_P]  # 沿mode-1分割
   每个节点p计算：
       Y_p^(n) = X_p × {U^(k)^T}_{k≠n}
   全局归约：
       Y^(n) = gather(Y_1^(n), ..., Y_P^(n))
   ```

3. **流水线并行**：
   不同模式的更新可以流水线化：
   ```
   时刻1: 更新U^(1)
   时刻2: 更新U^(2), 传输U^(1)
   时刻3: 更新U^(3), 传输U^(2), 使用U^(1)
   ...
   ```

### 16.2.5 随机化加速

**随机采样Tucker分解**：
1. 对每个模式随机采样纤维
2. 在采样的子张量上执行Tucker分解
3. 使用采样分解初始化完整分解

理论保证：在一定条件下，采样误差以高概率被控制在：
$$\|\mathcal{X} - \hat{\mathcal{X}}\|_F \leq (1 + \epsilon) \|\mathcal{X} - \mathcal{X}_k\|_F$$

其中 $\mathcal{X}_k$ 是最优的秩-$(R_1, \ldots, R_N)$ 近似。

**详细的随机化算法**：

1. **Fiber采样**：
   ```
   对于每个模式 n:
       选择采样数 s_n = O(R_n log R_n / ε^2)
       采样概率 p_i ∝ ||X(i,:,...,:)||_F^2  # leverage score
       采样索引集 S_n
   构建采样张量：
       X_S = X[S_1, S_2, ..., S_N]
   ```

2. **多阶投影**：
   使用随机投影矩阵压缩张量：
   $$\tilde{\mathcal{X}} = \mathcal{X} \times_1 \boldsymbol{\Omega}^{(1)} \times_2 \boldsymbol{\Omega}^{(2)} \times_3 \cdots \times_N \boldsymbol{\Omega}^{(N)}$$
   
   其中 $\boldsymbol{\Omega}^{(n)} \in \mathbb{R}^{s_n \times I_n}$ 是随机投影矩阵（如高斯矩阵或稀疏矩阵）。

3. **两阶段算法**：
   ```
   # 阶段1：快速近似
   [G_approx, {U_approx^(n)}] = tucker_als(τilde{X}, {R_n}, max_iter=5)
   
   # 阶段2：精细化
   [G, {U^(n)}] = tucker_als(X, {R_n}, init={U_approx^(n)}, max_iter=20)
   ```

**采样复杂度分析**：

对于大小为 $I_1 \times I_2 \times \cdots \times I_N$ 的张量：
- 完整Tucker-ALS：$O(\sum_n I_n \prod_{k \neq n} I_k)$
- 随机采样：$O(\sum_n s_n \prod_{k \neq n} s_k)$
- 加速比：$\prod_n (I_n/s_n)$

当 $s_n = O(\sqrt{I_n})$ 时，可获得 $O(2^{N/2})$ 倍加速。

**实用的采样策略**：

1. **自适应采样**：
   ```
   初始：s_n = min(50, I_n/10)
   while 不满足精度要求:
       s_n *= 1.5
       重新采样和计算
   ```

2. **分层采样**：
   - 将每个模式分成若干层
   - 每层独立采样
   - 保证重要用户/物品被采样

3. **混合采样**：
   - 重要fiber：确定性选择
   - 普通fiber：随机采样
   - 稀疏fiber：可能跳过

**随机化CP分解**：

1. **随机SGD**：
   ```
   for each epoch:
       随机采样一批非零元素 B
       for (i1, i2, ..., iN, v) in B:
           # 计算预测误差
           pred = ∑_r ∏_n A^(n)[i_n, r]
           err = v - pred
           
           # 更新因子
           for n in 1:N:
               grad = err * ∏_{k≠n} A^(k)[i_k, :]
               A^(n)[i_n, :] += η * grad
   ```

2. **采样ALS**：
   每次ALS更新只使用部分非零元素：
   ```
   采样率 q ∈ (0, 1]
   for each mode n:
       S = sample(nnz(X), q * nnz(X))
       A^(n) = update_with_samples(X, S, {A^(k)}_{k≠n})
   ```

3. **Sketched CP**：
   使用Count Sketch加速：
   ```
   # 预计算sketch
   for n in 1:N:
       sketch[n] = CountSketch(A^(n), hash_size)
   
   # 快速近似计算
   V^(n) ≈ FFT(conv(sketch[1], ..., sketch[n-1], sketch[n+1], ..., sketch[N]))
   ```

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

**高级优化实现**：

1. **块化MTTKRP**：
   ```c
   // 将非零元素按mode-n索引分块
   #define BLOCK_SIZE 64
   
   for (block = 0; block < num_blocks; block++) {
       // 预加载该块需要的因子矩阵行
       prefetch_factor_rows(block);
       
       #pragma omp parallel for
       for (idx = block_start[block]; idx < block_end[block]; idx++) {
           i_n = coords[idx].mode_n;
           val = values[idx];
           
           // SIMD化的内层循环
           #pragma omp simd
           for (r = 0; r < R; r++) {
               prod = val;
               for (k = 0; k < N; k++) {
                   if (k != n) {
                       prod *= A[k][coords[idx].indices[k]][r];
                   }
               }
               Y_private[tid][i_n][r] += prod;
           }
       }
   }
   
   // 归约私有结果
   reduce_private_results(Y_private, Y);
   ```

2. **缓存感知的重排序**：
   ```
   // Z-order (Morton order) 重排序
   struct NonZero {
       uint64_t morton_code;
       int indices[N];
       float value;
   };
   
   // 计算Morton码
   for (each nonzero) {
       nz.morton_code = interleave_bits(indices);
   }
   
   // 按Morton码排序
   sort(nonzeros, by_morton_code);
   ```

3. **融合计算**：
   将多个MTTKRP操作融合：
   ```
   // 传统：分别计算每个模式
   Y1 = MTTKRP(X, {A2, A3}, mode=1)
   Y2 = MTTKRP(X, {A1, A3}, mode=2)
   Y3 = MTTKRP(X, {A1, A2}, mode=3)
   
   // 融合：一次遍历
   for (each nonzero (i,j,k,v)) {
       for (r = 0; r < R; r++) {
           Y1[i,r] += v * A2[j,r] * A3[k,r];
           Y2[j,r] += v * A1[i,r] * A3[k,r];
           Y3[k,r] += v * A1[i,r] * A2[j,r];
       }
   }
   ```

4. **GPU优化**：
   ```cuda
   __global__ void sparse_mttkrp_kernel(
       int nnz, int* coords, float* vals,
       float** factors, float* output, 
       int mode, int rank
   ) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       if (tid >= nnz) return;
       
       // 共享内存缓存因子矩阵行
       __shared__ float cache[CACHE_SIZE];
       
       int out_idx = coords[tid * N + mode];
       float val = vals[tid];
       
       // Warp级并行
       for (int r = threadIdx.y; r < rank; r += blockDim.y) {
           float prod = val;
           #pragma unroll
           for (int n = 0; n < N; n++) {
               if (n != mode) {
                   int idx = coords[tid * N + n];
                   prod *= factors[n][idx * rank + r];
               }
           }
           atomicAdd(&output[out_idx * rank + r], prod);
       }
   }
   ```

5. **数据布局优化**：
   ```
   // AoS (Array of Structures) vs SoA (Structure of Arrays)
   
   // AoS (差的局部性)
   struct Element {
       int i, j, k;
       float value;
   } elements[nnz];
   
   // SoA (更好的向量化)
   int* indices_i;
   int* indices_j; 
   int* indices_k;
   float* values;
   
   // 进一步：分块SoA
   struct Block {
       int indices[3][BLOCK_SIZE];
       float values[BLOCK_SIZE];
   } blocks[];
   ```

**性能分析与调优**：

1. **瓶颈识别**：
   - 内存带宽：MTTKRP通常受限于内存带宽
   - 计算强度：O(R) flops per byte
   - 不规则访问：随机访问因子矩阵

2. **Roofline模型**：
   ```
   算术强度 = (2*R*nnz) / (nnz*12 + N*I*R*4)
   峰值性能 = min(峰值浮点, 算术强度 * 内存带宽)
   ```

3. **优化效果**：
   - 基础实现：~5% 峰值性能
   - 块化+向量化：~25% 峰值性能
   - GPU优化：~40% 峰值性能
   - 融合计算：2-3x 加速

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

**详细的分布式算法**：

1. **Medium-grained分区**：
   ```
   // 将张量分成P×Q×R个块
   对于节点(p,q,r)：
       存储X[i_p:i_{p+1}, j_q:j_{q+1}, k_r:k_{r+1}]
       本地因子：A1[i_p:i_{p+1},:], A2[j_q:j_{q+1},:], A3[k_r:k_{r+1},:]
   
   MTTKRP算法：
   1. 本地部分计算
   2. All-reduce在适当维度
   3. Scatter结果到相应节点
   ```

2. **通信避免算法**：
   ```
   // 3D张量在P个节点上
   for iteration = 1:max_iter
       // Mode-1更新(无通信)
       A1_local = local_mttkrp(X_local, A2_local, A3_local, mode=1)
       
       // Mode-2更新(需要A1)
       A1_gathered = allgather(A1_local, row_comm)
       A2_local = local_mttkrp(X_local, A1_gathered, A3_local, mode=2)
       
       // Mode-3更新(需要A1,A2)
       A2_gathered = allgather(A2_local, col_comm)
       A3_local = local_mttkrp(X_local, A1_gathered, A2_gathered, mode=3)
   end
   ```

3. **异步更新策略**：
   ```python
   class AsyncDistributedCP:
       def __init__(self, X_local, rank, world_size):
           self.X_local = X_local
           self.factors = [init_factor(dim, R) for dim in dims]
           self.buffer = [None] * N_modes
           self.version = [0] * N_modes
           
       def async_update(self, mode):
           # 使用过期的因子进行更新
           Y = local_mttkrp(self.X_local, self.buffer, mode)
           self.factors[mode] = solve_least_squares(Y)
           
           # 非阻塞发送
           req = comm.Isend(self.factors[mode], dest=all)
           
           # 尝试接收其他节点的更新
           if comm.Iprobe():
               new_factor, source, tag = comm.recv()
               self.buffer[tag] = new_factor
               self.version[tag] += 1
   ```

**负载均衡技术**：

1. **动态重分配**：
   ```
   // 监控每个节点的计算时间
   if load_imbalance() > threshold:
       // 计算新的分区
       new_partition = compute_balanced_partition(nnz_per_node)
       
       // 迁移数据
       migrate_data(old_partition, new_partition)
   ```

2. **工作窃取**：
   ```
   while has_work() or can_steal():
       if has_work():
           process_local_work()
       else:
           victim = select_victim_node()
           stolen_work = steal_from(victim)
           process_stolen_work(stolen_work)
   ```

3. **分层负载均衡**：
   - 节点间：粗粒度均衡
   - 节点内：线程级细粒度均衡

**容错与恢复**：

1. **检查点机制**：
   ```python
   def checkpoint():
       if iteration % checkpoint_interval == 0:
           # 保存因子矩阵
           save_factors_to_disk(self.factors)
           # 保存迭代状态
           save_state(iteration, loss)
   
   def recover():
       # 从最近的检查点恢复
       self.factors = load_factors_from_disk()
       iteration, loss = load_state()
       return iteration
   ```

2. **复制策略**：
   - 关键因子矩阵多副本
   - 使用Reed-Solomon编码
   - 跨机架复制

**通信模式优化**：

1. **Butterfly混合**：
   ```
   // log(P)轮通信完成全局交换
   for round = 0 to log2(P)-1:
       partner = rank XOR (1 << round)
       exchange_with(partner)
       merge_factors()
   ```

2. **分层All-reduce**：
   ```
   // 机架内先归约
   local_reduce(rack_comm)
   
   // 跨机架通信
   if is_rack_leader:
       global_reduce(leader_comm)
   
   // 机架内广播
   broadcast(rack_comm)
   ```

3. **流水线通信**：
   ```
   // 将大消息分成小块
   for chunk in chunks:
       send_async(chunk, next_node)
       if has_received():
           process_chunk(recv_chunk)
   ```

### 16.3.4 GPU加速的稀疏张量运算

**挑战**：
- 不规则的内存访问模式
- 负载不均衡
- 有限的GPU内存
- 原子操作的竞争

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

**高效的GPU实现**：

1. **分层并行策略**：
   ```cuda
   __global__ void hierarchical_mttkrp(
       int nnz, int4* coords, float* vals,
       float* A1, float* A2, float* A3, 
       float* output, int rank, int mode
   ) {
       // Block级别：处理一组非零元素
       __shared__ float s_factors[3][BLOCK_SIZE][RANK];
       
       // Warp级别：协作加载因子矩阵
       int warp_id = threadIdx.x / 32;
       int lane_id = threadIdx.x % 32;
       
       // 加载相关因子矩阵行到共享内存
       cooperative_load_factors(coords, s_factors);
       __syncthreads();
       
       // Thread级别：计算乘积
       for (int elem = blockIdx.x * blockDim.x + threadIdx.x; 
            elem < nnz; elem += gridDim.x * blockDim.x) {
           
           int4 coord = coords[elem];
           float val = vals[elem];
           int out_idx = get_index(coord, mode);
           
           // 小循环展开
           #pragma unroll 4
           for (int r = 0; r < rank; r += 4) {
               float4 prod = make_float4(val, val, val, val);
               
               // 向量化计算
               prod = compute_product_vectorized(
                   coord, s_factors, r, mode, prod
               );
               
               // 分散写回
               scatter_atomic_add(output, out_idx, r, prod);
           }
       }
   }
   ```

2. **Hash-based冲突解决**：
   ```cuda
   __device__ void scatter_atomic_add(
       float* output, int row, int col_start, float4 vals
   ) {
       // 使用hash table减少原子操作冲突
       const int HASH_SIZE = 1024;
       __shared__ float hash_table[HASH_SIZE];
       __shared__ int hash_keys[HASH_SIZE];
       
       int hash = (row * RANK + col_start) % HASH_SIZE;
       
       // 尝试写入hash table
       int old_key = atomicCAS(&hash_keys[hash], -1, row);
       if (old_key == -1 || old_key == row) {
           // 成功获得槽位
           atomicAdd(&hash_table[hash], vals.x + vals.y + vals.z + vals.w);
       } else {
           // 冲突，直接写全局内存
           atomicAdd(&output[row * RANK + col_start], vals.x);
           // ...
       }
   }
   ```

3. **Tensor Core优化**：
   ```cuda
   // 使用wmma API进行矩阵乘法
   #include <mma.h>
   using namespace nvcuda;
   
   __global__ void tensor_core_mttkrp(
       half* A_factors, half* B_factors, 
       float* C_output, int M, int N, int K
   ) {
       // 声明wmma fragments
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
       
       // 初始化累加器
       wmma::fill_fragment(c_frag, 0.0f);
       
       // Tensor Core计算
       for (int k = 0; k < K; k += 16) {
           wmma::load_matrix_sync(a_frag, A_factors + k, 16);
           wmma::load_matrix_sync(b_frag, B_factors + k, 16);
           wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
       }
       
       // 存储结果
       wmma::store_matrix_sync(C_output, c_frag, 16, wmma::mem_row_major);
   }
   ```

4. **动态并行度调整**：
   ```cuda
   __global__ void adaptive_kernel(
       int nnz, int* row_ptr, int* col_idx, float* vals
   ) {
       // 根据行的非零元素数量动态分配线程
       int row = blockIdx.x;
       int nnz_in_row = row_ptr[row+1] - row_ptr[row];
       
       if (nnz_in_row < 32) {
           // 小行：一个warp处理多行
           small_row_kernel(row, row_ptr, col_idx, vals);
       } else if (nnz_in_row < 1024) {
           // 中等行：一个block处理一行
           medium_row_kernel(row, row_ptr, col_idx, vals);
       } else {
           // 大行：多个block处理一行
           large_row_kernel(row, row_ptr, col_idx, vals);
       }
   }
   ```

5. **内存管理优化**：
   ```cuda
   class GPUMemoryPool {
   private:
       void* pool;
       size_t pool_size;
       std::vector<Block> free_blocks;
       
   public:
       void* allocate(size_t size) {
           // 从内存池分配
           auto it = std::find_if(free_blocks.begin(), free_blocks.end(),
               [size](const Block& b) { return b.size >= size; });
           
           if (it != free_blocks.end()) {
               void* ptr = it->ptr;
               it->ptr += size;
               it->size -= size;
               return ptr;
           }
           
           // 回退到cudaMalloc
           return fallback_allocate(size);
       }
       
       void batch_prefetch(const std::vector<TensorBlock>& blocks) {
           // 预取下一批数据
           #pragma omp parallel for
           for (int i = 0; i < blocks.size(); i++) {
               cudaMemPrefetchAsync(blocks[i].data, blocks[i].size, 
                                   device_id, stream[i]);
           }
       }
   };
   ```

**性能分析与调优**：

1. **Occupancy优化**：
   ```
   // 动态调整block大小
   int block_size;
   int min_grid_size;
   cudaOccupancyMaxPotentialBlockSize(
       &min_grid_size, &block_size, kernel, 0, 0
   );
   ```

2. **内存带宽优化**：
   - 合并内存访问
   - 使用纹理内存缓存只读数据
   - 避免bank conflict

3. **性能指标**：
   - COO格式：~200 GFLOPS (V100)
   - CSF格式：~150 GFLOPS (更好的内存局部性)
   - Tensor Core：~400 GFLOPS (限制条件下)

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