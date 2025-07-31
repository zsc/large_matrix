# 第13章：动态低秩近似

在大规模数据流和在线学习场景中，我们经常需要维护矩阵的低秩近似，同时适应数据的动态变化。本章深入探讨流式环境下的矩阵分解更新、自适应秩选择、在线补全算法，以及这些技术与现代深度学习中模型压缩的深刻联系。我们将特别关注时间-空间-精度的三方权衡，以及在非平稳环境中的理论保证。

## 学习目标

- 掌握增量SVD的核心算法及其数值稳定性分析
- 理解在线环境下的秩选择理论与实践
- 学会设计适应性遗忘机制处理概念漂移
- 建立动态低秩近似与神经网络压缩的联系
- 掌握流式矩阵计算的regret分析框架

## 13.1 流式矩阵近似基础

### 13.1.1 问题设定与挑战

考虑数据流设定：时刻$t$观察到向量$\mathbf{x}_t \in \mathbb{R}^n$或矩阵更新$\Delta\mathbf{A}_t$。目标是维护当前数据矩阵$\mathbf{A}_t$的秩$k$近似$\tilde{\mathbf{A}}_t$：

$$\mathbf{A}_t = \sum_{i=1}^t \mathbf{x}_i\mathbf{x}_i^T \quad \text{或} \quad \mathbf{A}_t = \mathbf{A}_0 + \sum_{i=1}^t \Delta\mathbf{A}_i$$

核心挑战：
1. **空间限制**：只能存储$O(nk)$而非$O(n^2)$的数据
2. **单遍处理**：每个数据点只能访问一次
3. **实时更新**：更新复杂度需远低于批处理SVD的$O(n^3)$
4. **精度保证**：与最优秩$k$近似的误差有界

**数学形式化**：定义流式低秩近似问题为
$$\min_{\tilde{\mathbf{A}}_t: \text{rank}(\tilde{\mathbf{A}}_t) \leq k} \|\mathbf{A}_t - \tilde{\mathbf{A}}_t\|_F^2$$

在约束条件下：
- 空间复杂度：$O(nk + \text{poly}(k))$
- 更新时间：$O(nk^2)$每个数据点
- 仅使用过去观测$\{\mathbf{x}_1, \ldots, \mathbf{x}_t\}$

**实际应用场景**：
- **推荐系统**：用户-物品交互矩阵的实时更新
- **网络监控**：流量矩阵的异常检测
- **金融数据**：协方差矩阵的在线估计
- **计算机视觉**：视频流的背景建模

### 13.1.2 性能度量与理论界限

定义累积regret：
$$R_T = \sum_{t=1}^T \|\mathbf{A}_t - \tilde{\mathbf{A}}_t\|_F^2 - \sum_{t=1}^T \|\mathbf{A}_t - \mathbf{A}_t^{(k)}\|_F^2$$

其中$\mathbf{A}_t^{(k)}$是$\mathbf{A}_t$的最优秩$k$近似。

**定理13.1**（流式SVD的regret界）：存在算法使得
$$R_T \leq O(k\log T \cdot \sum_{j=k+1}^n \lambda_j(\mathbf{A}_T))$$

这表明额外误差仅对数增长于时间$T$。

**证明要点**：
1. 利用矩阵扰动理论分析单步误差
2. 通过telescoping sum技术累积误差界
3. 应用在线学习的regret分析框架

**更细致的性能度量**：

**相对误差**：
$$\text{RelErr}_t = \frac{\|\mathbf{A}_t - \tilde{\mathbf{A}}_t\|_F}{\|\mathbf{A}_t\|_F}$$

**子空间距离**（更敏感的度量）：
$$d_{\text{sub}}(\mathbf{U}_t, \tilde{\mathbf{U}}_t) = \|\mathbf{U}_t\mathbf{U}_t^T - \tilde{\mathbf{U}}_t\tilde{\mathbf{U}}_t^T\|_F$$

**投影误差**（应用导向）：
$$\text{ProjErr}_t = \mathbb{E}_{\mathbf{x}}[\|\mathbf{x} - \tilde{\mathbf{U}}_t\tilde{\mathbf{U}}_t^T\mathbf{x}\|^2]$$

**定理13.2**（子空间追踪）：在温和条件下，
$$d_{\text{sub}}(\mathbf{U}_t, \tilde{\mathbf{U}}_t) \leq \frac{C}{\lambda_k(\mathbf{A}_t) - \lambda_{k+1}(\mathbf{A}_t)} \cdot \text{err}_t$$

其中分母是特征值间隙，决定了子空间分离的难度。

### 13.1.3 基础增量框架

**算法13.1**：通用增量低秩近似框架
```
输入：初始分解 U₀Σ₀V₀ᵀ，流数据 {xₜ}
维护：当前近似 UₜΣₜVₜᵀ，秩为k
For t = 1, 2, ...
  1. 接收新数据xₜ（或ΔAₜ）
  2. 计算残差：r = xₜ - UₜUₜᵀxₜ
  3. 更新子空间：
     - 若‖r‖ > θ：扩展U空间
     - 否则：仅更新系数
  4. 可选：秩截断保持k
  5. 可选：应用遗忘因子
```

关键设计选择：
- **扩展策略**：何时增加秩vs.投影到现有空间
- **截断策略**：如何选择保留的奇异值/向量
- **数值稳定化**：周期性正交化的时机

**详细的扩展策略分析**：

**策略1：固定阈值**
```
if ‖r‖ > θ:
    扩展子空间
else:
    仅更新系数
```
优点：简单直观
缺点：阈值选择困难，对数据尺度敏感

**策略2：相对阈值**
```
if ‖r‖/‖xₜ‖ > θ_rel:
    扩展子空间
```
优点：尺度不变性
适用：数据范数变化大的场景

**策略3：累积误差**
```
维护累积残差能量E_res
if E_res > θ_energy:
    扩展并重置E_res
```
优点：考虑历史信息
适用：缓慢变化的数据流

**截断策略比较**：

1. **硬阈值截断**：保留前$k$个奇异值
   - 优点：秩严格受控
   - 缺点：可能丢失重要信息

2. **软阈值截断**：$\sigma_i \leftarrow \max(\sigma_i - \tau, 0)$
   - 优点：平滑过渡
   - 应用：去噪场景

3. **能量保留截断**：保留95%能量的最小秩
   - 优点：自适应数据复杂度
   - 缺点：秩可能波动

### 13.1.4 与在线凸优化的联系

将低秩近似视为约束优化：
$$\min_{\mathbf{U}\in\mathbb{R}^{n\times k}} \sum_{t=1}^T \ell_t(\mathbf{U}) = \sum_{t=1}^T \|\mathbf{x}_t - \mathbf{U}\mathbf{U}^T\mathbf{x}_t\|^2$$

这建立了与在线梯度下降、镜像下降等经典算法的联系。特别地，Oja's rule可视为此问题的随机梯度解法。

**Oja's算法详解**：
更新规则：
$$\mathbf{U}_{t+1} = \mathbf{U}_t + \eta_t(\mathbf{x}_t\mathbf{x}_t^T - \mathbf{U}_t\mathbf{U}_t^T\mathbf{x}_t\mathbf{x}_t^T)\mathbf{U}_t$$

**收敛性质**：
- 步长$\eta_t = c/t$时，$\mathbb{E}[\|\mathbf{U}_t\mathbf{U}_t^T - \mathbf{U}_*\mathbf{U}_*^T\|_F^2] = O(1/t)$
- 需要特征值间隙条件：$\lambda_k > \lambda_{k+1}$

**推广：在线矩阵流形优化**

考虑Stiefel流形约束：$\mathbf{U}^T\mathbf{U} = \mathbf{I}_k$

**Riemannian SGD**：
```
1. 计算欧氏梯度：∇f(Uₜ)
2. 投影到切空间：grad = ∇f - Uₜ(Uₜᵀ∇f)
3. 沿测地线更新：Uₜ₊₁ = Retr(Uₜ, -ηₜ·grad)
```

其中Retraction可选：
- QR分解：$\text{Retr}(\mathbf{U}, \boldsymbol{\xi}) = \text{qr}(\mathbf{U} + \boldsymbol{\xi})$
- Cayley变换：计算更精确但更昂贵

**与经典在线算法的对比**：

| 算法 | 更新复杂度 | 收敛率 | 约束处理 |
|------|-----------|--------|----------|
| Oja's rule | $O(nk)$ | $O(1/t)$ | 渐近满足 |
| 在线梯度下降 | $O(nk)$ | $O(\sqrt{T})$ | 投影步 |
| Riemannian SGD | $O(nk^2)$ | $O(1/t)$ | 精确满足 |
| Block power | $O(nkb)$ | $O(1/t^2)$ | 渐近满足 |

**研究方向**：
- 非凸在线优化的收敛性分析
- 自适应步长的理论保证
- 分布式流式PCA的通信复杂度下界
- 带约束的在线矩阵分解（如非负、稀疏）
- 量子算法加速的可能性

## 13.2 增量SVD算法深度剖析

### 13.2.1 秩一更新的扰动分析

给定SVD分解$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，考虑秩一更新$\mathbf{A}' = \mathbf{A} + \mathbf{c}\mathbf{d}^T$。

**引理13.1**（Bunch-Nielsen）：更新后的SVD可通过以下步骤计算：
1. 投影：$\mathbf{p} = \mathbf{U}^T\mathbf{c}$，$\mathbf{q} = \mathbf{V}^T\mathbf{d}$
2. 正交补：$\mathbf{r}_c = \mathbf{c} - \mathbf{U}\mathbf{p}$，$\mathbf{r}_d = \mathbf{d} - \mathbf{V}\mathbf{q}$
3. 构造中间矩阵：
   $$\mathbf{K} = \begin{bmatrix} \boldsymbol{\Sigma} + \mathbf{p}\mathbf{q}^T & \|\mathbf{r}_d\|\mathbf{p} \\ \|\mathbf{r}_c\|\mathbf{q}^T & \mathbf{r}_c^T\mathbf{r}_d \end{bmatrix}$$
4. 对$\mathbf{K}$进行SVD得到更新

计算复杂度：$O(nk + k^3)$，远优于重新计算的$O(n^2k)$。

**详细推导**：

将$\mathbf{A}' = \mathbf{A} + \mathbf{c}\mathbf{d}^T$重写为：
$$\mathbf{A}' = [\mathbf{U}, \mathbf{c}] \begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{0} \\ \mathbf{0}^T & 1 \end{bmatrix} [\mathbf{V}, \mathbf{d}]^T$$

但$[\mathbf{U}, \mathbf{c}]$和$[\mathbf{V}, \mathbf{d}]$可能不正交。通过正交化：

$$[\mathbf{U}, \mathbf{c}] = [\mathbf{U}, \tilde{\mathbf{u}}]\begin{bmatrix} \mathbf{I} & \mathbf{p} \\ \mathbf{0}^T & \rho_c \end{bmatrix}$$

其中$\tilde{\mathbf{u}} = \mathbf{r}_c/\|\mathbf{r}_c\|$，$\rho_c = \|\mathbf{r}_c\|$。

**稳定性分析**：

**定理13.3**：设$\kappa = \|\mathbf{c}\|\|\mathbf{d}\|/\sigma_{\min}(\mathbf{A})$，则相对误差满足：
$$\frac{\|\tilde{\mathbf{A}}' - \mathbf{A}'\|_F}{\|\mathbf{A}'\|_F} \leq \epsilon_{\text{machine}} \cdot O(\kappa^2)$$

这表明当更新幅度相对于最小奇异值过大时，算法可能不稳定。

**特殊情况处理**：

1. **当$\mathbf{r}_c \approx 0$或$\mathbf{r}_d \approx 0$**：
   - 更新在现有子空间内
   - 仅更新$\boldsymbol{\Sigma}$，不改变$\mathbf{U}, \mathbf{V}$
   - 避免不必要的秩增加

2. **当$\mathbf{A}$不是满秩时**：
   - 需要特殊处理零奇异值
   - 使用伪逆或正则化技术

### 13.2.2 Brand算法及其稳定化

**算法13.2**：Brand's增量SVD
```
输入：当前SVD UΣVᵀ (秩k)，新列c
1. 计算投影系数：m = Uᵀc
2. 计算正交分量：p = c - Um, ρ = ‖p‖
3. 若ρ > ε：
   构造扩展矩阵并SVD分解
4. 否则：
   仅更新Σ中的系数
5. 定期重正交化U, V
```

**数值稳定性改进**：
- Gram-Schmidt vs. Householder正交化
- 迭代refinement技术
- 基于条件数的自适应重计算

**完整的Brand算法实现细节**：

```
function updateSVD(U, Σ, V, c):
    // 步骤1: 投影到当前子空间
    m = Uᵀc
    p = c - Um
    ρ = ‖p‖
    
    // 步骤2: 判断是否需要扩展秩
    if ρ > ε_threshold:
        // 步骤3a: 构造扩展矩阵
        P = p/ρ  // 归一化
        Q = zeros(k+1, 1)
        Q[k+1] = 1
        
        // 构造(k+1)×(k+1)矩阵
        K = [Σ  m]
            [0  ρ]
            
        // 步骤4a: 对K进行SVD
        [U', Σ', V'] = svd(K)
        
        // 步骤5a: 更新大矩阵
        U_new = [U P] @ U'
        V_new = [V Q] @ V'
        Σ_new = Σ'
    else:
        // 步骤3b: 仅更新系数
        K = Σ + m*1ᵀ  // 秩一更新
        [U', Σ', V'] = svd(K)
        U_new = U @ U'
        V_new = V @ V'
        Σ_new = Σ'
    
    return U_new, Σ_new, V_new
```

**正交化策略比较**：

1. **修正Gram-Schmidt (MGS)**
   - 优点：在线更新，内存高效
   - 缺点：数值稳定性较差
   - 适用：中等精度要求

2. **Householder变换**
   - 优点：数值稳定性最佳
   - 缺点：计算量大，不易并行
   - 适用：高精度要求

3. **快速Givens旋转**
   - 优点：局部更新，适合稀疏
   - 缺点：串行性质
   - 适用：特定结构

**自适应重正交化**：

```
function checkOrthogonality(U):
    orth_error = ‖UᵀU - I‖_F
    if orth_error > ε_reorth:
        U = reorthogonalize(U)
    return U
```

重正交化时机：
- 每隔$O(1/\epsilon_{\text{machine}})$次更新
- 当正交性误差超过阈值
- 在秩截断操作后

### 13.2.3 高阶更新：块算法

对于批量更新$\mathbf{A}' = \mathbf{A} + \mathbf{C}\mathbf{D}^T$（$\mathbf{C}, \mathbf{D} \in \mathbb{R}^{n \times b}$）：

**定理13.2**：块更新可分解为
$$\mathbf{A}' = \begin{bmatrix} \mathbf{U} & \tilde{\mathbf{U}} \end{bmatrix} \tilde{\boldsymbol{\Sigma}} \begin{bmatrix} \mathbf{V} & \tilde{\mathbf{V}} \end{bmatrix}^T$$

其中$\tilde{\mathbf{U}}, \tilde{\mathbf{V}}$来自$\mathbf{C}, \mathbf{D}$的QR分解。

这在处理mini-batch更新时特别高效。

**块更新算法详解**：

```
function blockUpdateSVD(U, Σ, V, C, D):
    // 步骤1: QR分解输入矩阵
    [Q_C, R_C] = qr(C - U(UᵀC))  // 正交化C
    [Q_D, R_D] = qr(D - V(VᵀD))  // 正交化D
    
    // 步骤2: 构造扩展矩阵
    K = [Σ      UᵀC]
        [VᵀD    R_C @ R_Dᵀ]
    
    // 步骤3: 对K进行SVD (size: (k+b)×(k+b))
    [U_K, Σ_K, V_K] = svd(K)
    
    // 步骤4: 更新原始矩阵
    U_new = [U, Q_C] @ U_K
    V_new = [V, Q_D] @ V_K
    Σ_new = Σ_K
    
    // 步骤5: 可选秩截断
    if rank(Σ_new) > k_max:
        [U_new, Σ_new, V_new] = truncate(U_new, Σ_new, V_new, k_max)
    
    return U_new, Σ_new, V_new
```

**效率分析**：

复杂度分解：
- QR分解：$O(nb^2)$
- 构造K矩阵：$O(kb + b^2)$
- K的SVD：$O((k+b)^3)$
- 矩阵乘法：$O(n(k+b)^2)$

总复杂度：$O(nb^2 + n(k+b)^2 + (k+b)^3)$

**与逐个更新的对比**：
- 逐个更新$b$次：$O(nbk + bk^3)$
- 块更新：$O(nb^2 + n(k+b)^2)$
- 当$b \ll k$时，块更新显著更快

**特殊块结构的优化**：

1. **对称更新**：$\mathbf{A}' = \mathbf{A} + \mathbf{C}\mathbf{C}^T$
   - 只需一次QR分解
   - 保持对称性

2. **低秩更新**：当$b \ll k$时
   - 使用Sherman-Morrison-Woodbury公式
   - 避免扩展到$(k+b) \times (k+b)$

3. **稀疏更新**：当$\mathbf{C}, \mathbf{D}$稀疏时
   - 利用稀疏性减少计算
   - 特殊的索引结构

### 13.2.4 分布式增量SVD

在分布式设置下，不同节点观察数据流的不同部分：

**算法13.3**：异步分布式SVD更新
```
节点i维护局部近似Uᵢ
1. 局部更新：处理本地数据流
2. 周期通信：交换子空间信息
3. 共识步骤：对齐不同节点的子空间
4. 全局更新：合并得到全局U
```

关键挑战：
- 子空间对齐的高效算法
- 通信与精度的权衡
- 拜占庭节点的鲁棒性

**分布式架构详解**：

```
// 节点i的本地算法
function nodeUpdate(local_data_stream):
    // 本地状态
    U_local = []  // n×k_local
    Σ_local = []  // k_local×k_local
    buffer = []   // 数据缓冲
    
    while true:
        // 阶段1: 本地处理
        batch = collect_batch(local_data_stream)
        U_local, Σ_local = incrementalSVD(U_local, Σ_local, batch)
        
        // 阶段2: 周期通信
        if time_to_communicate():
            // 发送本地子空间的紧凑表示
            sketch = compute_sketch(U_local, Σ_local)
            broadcast(sketch)
            
            // 接收其他节点的sketch
            sketches = receive_all()
            
            // 阶段3: 子空间对齐
            U_global = align_subspaces(sketches)
            
            // 阶段4: 本地投影更新
            U_local = project_to_global(U_local, U_global)
```

**子空间对齐算法**：

**方法1：Grassmannian平均**
```
function align_subspaces(U_list):
    // 计算Grassmannian重心
    U_mean = grassmannian_mean(U_list)
    return U_mean
    
function grassmannian_mean(U_list):
    // 迭代算法
    U_avg = U_list[0]
    for iter in 1:max_iters:
        // 计算到各子空间的测地线
        tangents = []
        for U in U_list:
            v = log_map(U_avg, U)
            tangents.append(v)
        
        // 平均切向量
        v_mean = mean(tangents)
        
        // 沿测地线更新
        U_avg = exp_map(U_avg, v_mean)
    return U_avg
```

**方法2：分布式特征分解**
```
function distributed_eigen_alignment():
    // 构造分布式协方差矩阵
    C_global = sum_all_nodes(U_local @ Σ_local² @ U_localᵀ)
    
    // 分布式特征分解
    [V, Λ] = distributed_eig(C_global, k)
    
    return V[:, 1:k]
```

**通信优化技术**：

1. **量化压缩**
   ```
   function quantize_subspace(U, bits):
       // 随机旋转
       R = random_orthogonal(k)
       U_rot = U @ R
       
       // 量化
       U_quant = quantize(U_rot, bits)
       
       return U_quant, R
   ```

2. **随机投影**
   ```
   function sketch_subspace(U, Σ, sketch_size):
       // Johnson-Lindenstrauss投影
       S = randn(n, sketch_size) / sqrt(sketch_size)
       Y = Sᵀ @ U @ Σ
       
       return Y  // sketch_size × k
   ```

3. **分层通信**
   - 频繁交换主要成分
   - 稀疏交换次要成分
   - 基于重要性的自适应频率

**拜占庭鲁棒性**：

```
function byzantine_robust_aggregation(sketches, f):
    // f = 拜占庭节点数量上界
    
    // 方法1: 中位数聚合
    median_sketch = geometric_median(sketches)
    
    // 方法2: 修剪均值
    distances = compute_distances(sketches)
    good_nodes = trim_outliers(distances, f)
    robust_mean = mean(sketches[good_nodes])
    
    return robust_mean
```

**研究方向**：
- 基于草图的分布式SVD通信压缩
- 去中心化共识SVD算法
- 异构数据分布下的理论分析
- 动态网络拓扑下的适应性算法
- 差分隐私保护的分布式PCA

## 13.3 自适应秩选择

### 13.3.1 在线模型选择理论

流式环境下的秩选择面临独特挑战：需在观察完整数据前做出决策。核心权衡：
- 秩过小：欠拟合，丢失重要信息
- 秩过大：计算/存储开销，过拟合噪声

**定义13.1**（在线秩选择regret）：
$$R_k(T) = \sum_{t=1}^T \|\mathbf{A}_t - \tilde{\mathbf{A}}_{t,k_t}\|_F^2 - \min_{k^*} \sum_{t=1}^T \|\mathbf{A}_t - \tilde{\mathbf{A}}_{t,k^*}\|_F^2$$

其中$k_t$是算法在时刻$t$选择的秩。

### 13.3.2 能量阈值方法

**算法13.4**：自适应能量阈值
```
参数：能量保留比例τ ∈ (0,1)
维护：奇异值{σᵢ}及对应向量
1. 更新奇异值分解
2. 计算累积能量：Eⱼ = Σᵢ₌₁ʲ σᵢ² / Σᵢ σᵢ²
3. 选择秩：k* = min{j : Eⱼ ≥ τ}
4. 自适应调整τ基于历史性能
```

**定理13.3**：在平稳假设下，自适应能量阈值达到
$$\mathbb{E}[R_k(T)] \leq O(\sqrt{T\log K})$$
其中$K$是最大允许秩。

### 13.3.3 基于预测的秩选择

利用奇异值衰减模式预测未来：

**模型假设**：
1. 幂律衰减：$\sigma_i \sim i^{-\alpha}$
2. 指数衰减：$\sigma_i \sim e^{-\beta i}$
3. 混合模型：组合多种衰减模式

**算法13.5**：预测性秩选择
```
1. 在线估计衰减参数α或β
2. 预测未来L步的奇异值分布
3. 选择秩k最小化预期误差：
   k* = argmin_k E[Σₜ₊₁ᵗ⁺ᴸ ‖Aₜ - Ãₜ,ₖ‖²]
4. 使用滑动窗口更新参数估计
```

### 13.3.4 贝叶斯秩选择

将秩视为隐变量，使用在线贝叶斯推断：

**概率模型**：
$$p(\mathbf{A}|k) = \mathcal{MN}(\mathbf{0}, \mathbf{I}_n, \boldsymbol{\Sigma}_k)$$
其中$\boldsymbol{\Sigma}_k$是秩$k$协方差。

**在线推断**：
1. 维护秩的后验分布$p(k|D_{1:t})$
2. 使用粒子滤波或变分推断更新
3. 选择MAP估计或后验期望

**研究方向**：
- 非平稳环境下的秩追踪
- 多尺度秩选择（不同时间尺度不同秩）
- 与在线核学习的联系

## 13.4 在线矩阵补全

### 13.4.1 流式观测模型

考虑部分观测的流式设定：
- 时刻$t$：观测位置$(i_t, j_t)$的值$M_{i_t,j_t}$
- 目标：实时预测未观测位置
- 约束：低秩假设$\text{rank}(\mathbf{M}) \leq r$

**形式化**：在线学习协议
```
For t = 1, 2, ..., T:
  1. 算法预测M̂ₜ（秩≤r）
  2. 对手选择(iₜ, jₜ)
  3. 观测真实值Mᵢₜ,ⱼₜ
  4. 遭受损失ℓₜ = (M̂ₜ[iₜ,jₜ] - Mᵢₜ,ⱼₜ)²
```

### 13.4.2 在线梯度下降方法

矩阵补全的因子分解形式：$\mathbf{M} \approx \mathbf{U}\mathbf{V}^T$

**算法13.6**：流式矩阵分解
```
初始化：U₀, V₀ ∈ ℝⁿˣʳ
For each观测(i,j,Mᵢⱼ):
  1. 预测：M̂ᵢⱼ = ⟨uᵢ, vⱼ⟩
  2. 梯度计算：
     ∇ᵤᵢ = -2(Mᵢⱼ - M̂ᵢⱼ)vⱼ
     ∇ᵥⱼ = -2(Mᵢⱼ - M̂ᵢⱼ)uᵢ
  3. 更新：
     uᵢ ← uᵢ - ηₜ∇ᵤᵢ
     vⱼ ← vⱼ - ηₜ∇ᵥⱼ
  4. 可选正则化/投影步骤
```

**收敛性分析**：
- 凸松弛下：$O(\sqrt{T})$ regret
- 非凸但满足RIP：线性收敛到局部最优
- 一般情况：依赖初始化质量

### 13.4.3 Riemannian优化视角

将低秩约束视为流形约束：
$$\mathcal{M}_r = \{\mathbf{X} \in \mathbb{R}^{m \times n} : \text{rank}(\mathbf{X}) = r\}$$

**算法13.7**：在线Riemannian梯度下降
```
维护：当前点Xₜ ∈ 𝓜ᵣ
1. 计算欧氏梯度∇f(Xₜ)
2. 投影到切空间：gradₜ = Proj_{TXₜ}(∇f)
3. 沿测地线更新：
   Xₜ₊₁ = Retr_Xₜ(-ηₜ gradₜ)
```

关键优势：
- 自动满足秩约束
- 更好的收敛性质
- 自然处理矩阵的内在几何

### 13.4.4 带边信息的在线补全

实际应用常有辅助信息（如用户特征、时间戳）：

**模型扩展**：
$$M_{ij} = \langle \mathbf{u}_i, \mathbf{v}_j \rangle + f(\mathbf{x}_i, \mathbf{y}_j; \boldsymbol{\theta})$$

其中$f$可以是：
- 线性：$\boldsymbol{\theta}^T[\mathbf{x}_i; \mathbf{y}_j]$
- 神经网络：更复杂的特征交互
- 核方法：非线性但计算高效

**在线学习策略**：
1. 交替优化潜在因子和特征函数
2. 端到端梯度下降
3. 两阶段：先学特征再补全

**研究方向**：
- 冷启动场景的理论保证
- 非均匀采样下的无偏估计
- 对抗性corruptions的鲁棒性

## 13.5 遗忘机制与概念漂移

### 13.5.1 指数遗忘模型

处理非平稳数据流的经典方法：

**更新规则**：
$$\mathbf{A}_t = \gamma \mathbf{A}_{t-1} + \mathbf{x}_t\mathbf{x}_t^T$$

其中$\gamma \in (0,1)$是遗忘因子。

**性质分析**：
- 有效窗口长度：$\approx 1/(1-\gamma)$
- 特征值衰减：$\lambda_i(t) = \gamma\lambda_i(t-1) + \text{新贡献}$
- 偏差-方差权衡：$\gamma$越小适应越快但方差越大

### 13.5.2 自适应遗忘因子

**算法13.8**：基于变化检测的自适应遗忘
```
维护：多个时间尺度的统计量
1. 短期统计：Sₛ = 最近k个样本
2. 长期统计：Sₗ = 指数平滑历史
3. 检测变化：D = d(Sₛ, Sₗ)
4. 调整遗忘：
   if D > threshold:
     γ ← max(γ - δ, γₘᵢₙ)  # 加快遗忘
   else:
     γ ← min(γ + δ, γₘₐₓ)  # 减慢遗忘
```

**理论保证**：在分段平稳假设下，自适应算法达到
$$\text{Regret} \leq O(\sqrt{T(1 + N_c)})$$
其中$N_c$是变化点数量。

### 13.5.3 滑动窗口技术

保持固定大小$W$的数据窗口：

**精确滑窗SVD**：
```
数据结构：循环缓冲区存储最近W个样本
更新策略：
1. 加入新样本：秩1更新
2. 移除旧样本：秩1降级（downdating）
3. 周期性从头计算以控制误差累积
```

**近似滑窗方法**：
- Sketching：维护数据的线性草图
- 采样：概率保留历史样本
- 分层：多分辨率时间表示

### 13.5.4 多尺度遗忘

不同奇异向量可能有不同的时间尺度：

**模型**：
$$\mathbf{u}_i(t) = \gamma_i \mathbf{u}_i(t-1) + \text{update}$$

其中$\gamma_i$依赖于第$i$个模式的稳定性。

**自动尺度发现**：
1. 监控每个奇异向量的变化率
2. 稳定模式：大$\gamma_i$（慢遗忘）
3. 快变模式：小$\gamma_i$（快遗忘）

**研究方向**：
- 连续时间遗忘过程的SDE建模
- 遗忘机制的信息论分析
- 与持续学习(continual learning)的统一框架

## 13.6 与神经网络压缩的联系

### 13.6.1 权重矩阵的动态低秩分解

深度网络中的权重矩阵$\mathbf{W} \in \mathbb{R}^{m \times n}$常呈现低秩结构：

**在线压缩框架**：
```
训练过程中：
1. 监控权重矩阵的有效秩
2. 当检测到低秩结构时分解：W ≈ UV^T
3. 继续以因子形式训练
4. 周期性评估是否需要调整秩
```

**与剪枝的对比**：
- 剪枝：离散的0/1决策
- 低秩：连续的子空间选择
- 统一视角：都在减少有效自由度

### 13.6.2 增量SVD用于在线压缩

**算法13.9**：训练时动态压缩
```
每E个epoch:
1. 对当前权重W计算SVD
2. 评估能量分布确定有效秩k
3. 压缩：W ← UₖΣₖVₖᵀ
4. 可选：切换到因子形式继续训练
5. 监控压缩对性能的影响
```

**关键考虑**：
- 压缩时机：loss平台期vs.固定周期
- 秩的自适应：基于梯度信息或验证性能
- 与优化器状态的协调（如Adam的动量）

### 13.6.3 彩票假说的低秩视角

**彩票假说**：随机初始化的网络包含能独立训练到同等精度的稀疏子网络。

**低秩重述**：
- 稀疏mask $\leftrightarrow$ 低秩投影
- 寻找"中奖彩票" $\leftrightarrow$ 发现主要子空间
- 迭代剪枝 $\leftrightarrow$ 渐进秩减少

**实验观察**：
1. 早期训练快速形成低秩结构
2. 后期训练主要是细化已有结构
3. 不同层的秩演化模式不同

### 13.6.4 持续学习中的子空间管理

在序列任务学习中防止遗忘：

**策略13.1**：正交子空间分配
```
对每个新任务t:
1. 识别之前任务使用的子空间Uₚᵣₑᵥ
2. 在正交补空间中学习：U⊥
3. 合并：Uₜₒₜₐₗ = [Uₚᵣₑᵥ | Uₙₑw]
4. 当空间耗尽时压缩或合并
```

**策略13.2**：弹性权重巩固(EWC)的低秩版本
- 原始EWC：保护重要参数
- 低秩EWC：保护重要子空间
- 计算Fisher信息矩阵的低秩近似

**研究方向**：
- 任务相似度与子空间重叠的关系
- 最优子空间分配策略
- 与元学习的理论联系

## 本章小结

本章系统介绍了动态低秩近似的理论与算法：

**核心概念**：
1. **流式SVD更新**：Brand算法及其变体实现$O(nk)$复杂度的增量更新
2. **自适应秩选择**：能量阈值、预测模型、贝叶斯方法处理模型选择
3. **在线矩阵补全**：梯度方法、Riemannian优化处理流式观测
4. **遗忘机制**：指数衰减、滑动窗口、自适应遗忘处理概念漂移
5. **神经网络联系**：低秩结构在深度学习压缩和持续学习中的应用

**关键公式**：
- Bunch-Nielsen更新：$\mathbf{A}' = \mathbf{A} + \mathbf{c}\mathbf{d}^T$的SVD增量计算
- 在线regret界：$R_T \leq O(k\log T \cdot \sum_{j>k}\lambda_j)$
- 遗忘因子更新：$\mathbf{A}_t = \gamma\mathbf{A}_{t-1} + \mathbf{x}_t\mathbf{x}_t^T$
- 能量保留准则：$k^* = \min\{j : \sum_{i=1}^j \sigma_i^2 \geq \tau \sum_i \sigma_i^2\}$

**实践要点**：
- 数值稳定性需要周期性重正交化
- 秩选择与计算资源的权衡
- 非平稳环境需要自适应机制
- 分布式场景的通信优化

## 练习题

### 基础题

**习题13.1**：推导秩一更新的计算复杂度
给定$n \times n$矩阵的SVD和秩一更新$\mathbf{u}\mathbf{v}^T$，证明更新后的SVD可在$O(n^2)$时间内计算。
<details>
<summary>提示</summary>
利用Woodbury矩阵恒等式和SVD的扰动理论。
</details>

**习题13.2**：设计滑动窗口SVD算法
实现保持最近$W$个样本的精确SVD，分析添加新样本和删除旧样本的复杂度。
<details>
<summary>提示</summary>
考虑downdating技术和数值稳定性问题。
</details>

**习题13.3**：比较不同遗忘因子
对于$\gamma \in \{0.9, 0.95, 0.99\}$，计算有效窗口长度，并分析其对突变检测的影响。
<details>
<summary>提示</summary>
有效窗口长度约为$1/(1-\gamma)$。
</details>

**习题13.4**：在线矩阵补全的收敛性
证明在强凸假设下，在线梯度下降达到$O(\log T)$ regret。
<details>
<summary>提示</summary>
使用在线凸优化的标准分析技术。
</details>

### 挑战题

**习题13.5**：非均匀采样的偏差校正
在矩阵补全中，若位置$(i,j)$以概率$p_{ij}$被观测，设计无偏估计算法。
<details>
<summary>提示</summary>
使用重要性采样和inverse propensity weighting。
</details>

**习题13.6**：多尺度遗忘的最优性
证明对于分层时间序列数据，多尺度遗忘优于单一遗忘因子。构造具体反例。
<details>
<summary>提示</summary>
考虑包含快变和慢变成分的合成数据。
</details>

**习题13.7**：分布式流式PCA的通信下界
$m$个节点观测数据流，证明达到$\epsilon$-近似需要$\Omega(mk/\epsilon)$通信量。
<details>
<summary>提示</summary>
使用信息论下界和通信复杂度理论。
</details>

**习题13.8**：神经网络压缩的理论保证
给定预训练网络，证明存在秩$k$分解使得精度损失不超过$\epsilon$，其中$k$如何依赖于$\epsilon$？
<details>
<summary>提示</summary>
结合矩阵扰动理论和神经网络的Lipschitz性质。
</details>

## 常见陷阱与错误 (Gotchas)

1. **数值误差累积**：长时间运行的增量算法会累积舍入误差
   - 解决方案：周期性从零开始重算
   - 监控正交性：检查$\|\mathbf{U}^T\mathbf{U} - \mathbf{I}\|$

2. **秩爆炸问题**：没有及时截断会导致秩不断增长
   - 设置最大秩限制
   - 使用能量阈值自动截断

3. **遗忘因子选择**：固定遗忘因子在变化环境下表现差
   - 使用自适应方法
   - 监控预测误差调整

4. **并发更新冲突**：分布式环境下的数据竞争
   - 使用lock-free数据结构
   - 设计异步友好的算法

5. **初始化敏感性**：在线算法可能收敛到次优解
   - 使用多次随机初始化
   - 借鉴离线算法的初始化策略

## 最佳实践检查清单

### 算法设计阶段
- [ ] 明确数据流特性（平稳性、到达率、噪声水平）
- [ ] 确定精度vs.效率的权衡点
- [ ] 选择合适的秩选择策略
- [ ] 设计异常检测和恢复机制

### 实现阶段
- [ ] 使用数值稳定的更新公式
- [ ] 实现高效的数据结构（循环缓冲、优先队列）
- [ ] 添加正交性检查和自动修正
- [ ] 支持checkpoint和恢复

### 调优阶段
- [ ] 监控关键指标（近似误差、计算时间、内存使用）
- [ ] 根据数据特性调整超参数
- [ ] 评估不同遗忘策略的效果
- [ ] 测试极端情况（突变、缺失数据）

### 部署阶段
- [ ] 设置自动报警机制
- [ ] 实现优雅降级策略
- [ ] 准备离线重算方案
- [ ] 记录详细日志用于调试
