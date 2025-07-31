# 第1章：二阶优化的统一框架

二阶优化方法通过利用曲率信息来加速收敛，是大规模机器学习中的核心技术。本章建立一个统一的数学框架，揭示Newton法、Gauss-Newton法和Natural Gradient之间的深刻联系，并探讨这些方法在现代深度学习中的应用与挑战。

## 1.1 Newton法、Gauss-Newton法与Natural Gradient的数学联系

### 1.1.1 统一视角：从泰勒展开到几何解释

考虑优化问题 $\min_{\mathbf{w}} f(\mathbf{w})$，其中 $\mathbf{w} \in \mathbb{R}^n$。二阶方法的核心思想是利用目标函数的局部二次近似：

$$f(\mathbf{w} + \Delta\mathbf{w}) \approx f(\mathbf{w}) + \nabla f(\mathbf{w})^T \Delta\mathbf{w} + \frac{1}{2} \Delta\mathbf{w}^T \mathbf{H} \Delta\mathbf{w}$$

其中 $\mathbf{H}$ 是某种形式的曲率矩阵。不同方法的区别在于如何定义和近似这个曲率矩阵。

**Newton法**：直接使用Hessian矩阵 $\mathbf{H} = \nabla^2 f(\mathbf{w})$

二次模型：$m_N(\Delta\mathbf{w}) = f(\mathbf{w}) + \nabla f(\mathbf{w})^T \Delta\mathbf{w} + \frac{1}{2} \Delta\mathbf{w}^T \nabla^2 f(\mathbf{w}) \Delta\mathbf{w}$

更新规则：$\Delta\mathbf{w} = -[\nabla^2 f(\mathbf{w})]^{-1} \nabla f(\mathbf{w})$

**Gauss-Newton法**：对于最小二乘问题 $f(\mathbf{w}) = \frac{1}{2}\|\mathbf{r}(\mathbf{w})\|^2$，使用一阶近似

完整Hessian：$\nabla^2 f(\mathbf{w}) = \mathbf{J}^T\mathbf{J} + \sum_{i=1}^m r_i(\mathbf{w})\nabla^2 r_i(\mathbf{w})$

Gauss-Newton近似：$\mathbf{H}_{GN} = \mathbf{J}^T\mathbf{J}$（忽略二阶项）

其中 $\mathbf{J} = \nabla \mathbf{r}(\mathbf{w})$ 是残差的Jacobian矩阵。

**Natural Gradient**：从信息几何角度，使用Fisher信息矩阵

参数空间的Riemannian度量：$ds^2 = \mathbf{d}\mathbf{w}^T \mathbf{F}(\mathbf{w}) \mathbf{d}\mathbf{w}$

Fisher信息矩阵：$\mathbf{F} = \mathbb{E}_{p(\mathbf{x}|\mathbf{w})}[\nabla \log p(\mathbf{x}|\mathbf{w}) \nabla \log p(\mathbf{x}|\mathbf{w})^T]$

Natural gradient：$\tilde{\nabla}f(\mathbf{w}) = \mathbf{F}^{-1}(\mathbf{w})\nabla f(\mathbf{w})$

**几何解释的深化**：

1. **欧氏空间 vs 统计流形**：
   - Newton法：假设参数空间是平坦的欧氏空间
   - Natural Gradient：考虑参数化引起的流形弯曲
   - 度量张量：$g_{ij}(\mathbf{w}) = \mathbb{E}[\partial_i \ell(\mathbf{x};\mathbf{w}) \partial_j \ell(\mathbf{x};\mathbf{w})]$

2. **KL散度的二阶近似**：
   $$D_{KL}(p(\cdot|\mathbf{w})||p(\cdot|\mathbf{w}+\Delta\mathbf{w})) \approx \frac{1}{2}\Delta\mathbf{w}^T\mathbf{F}(\mathbf{w})\Delta\mathbf{w}$$
   
   这解释了为什么Natural Gradient在概率模型中特别有效。

3. **坐标变换的不变性**：
   - Newton法：对仿射变换不变
   - Natural Gradient：对任意可微同胚参数化不变
   - 实践意义：对网络参数的重新缩放具有鲁棒性

### 1.1.2 数学等价性的条件

**定理 1.1**（Gauss-Newton与Natural Gradient的等价性）
对于指数族分布的负对数似然最小化问题，当模型正确指定且在最优解处时，Gauss-Newton Hessian等于Fisher信息矩阵。

**证明要点**：
1. 对于负对数似然 $f(\mathbf{w}) = -\log p(\mathbf{y}|\mathbf{x}, \mathbf{w})$
2. 在最优解处，期望Hessian等于Fisher信息矩阵（Bartlett恒等式）
3. Gauss-Newton忽略的二阶项在最优解处为零

**更深入的等价性分析**：

**定理 1.1a**（广义线性模型中的精确等价）
对于广义线性模型(GLM)，若链接函数是canonical的，则：
$$\mathbf{H}_{GN} = \mathbf{X}^T\mathbf{W}\mathbf{X} = \mathbf{F}$$
其中 $\mathbf{W} = \text{diag}(w_1, ..., w_n)$ 是权重矩阵。

**定理 1.1b**（深度网络中的近似等价）
在宽度趋于无穷的神经网络中，输出层的Gauss-Newton矩阵收敛到Neural Tangent Kernel (NTK)：
$$\lim_{m \to \infty} \mathbf{H}_{GN} = \mathbf{K}_{NTK}$$

**等价性破坏的情形**：

1. **模型误设(Model Misspecification)**：
   - 真实数据分布 $q(\mathbf{x})$ 不在模型族 $\{p(\mathbf{x}|\mathbf{w})\}$ 中
   - 此时 $\mathbf{F} \neq \mathbb{E}[\nabla^2 \ell]$，等价性不成立
   - 实践建议：使用sandwich estimator校正

2. **有限样本效应**：
   - Empirical Fisher: $\hat{\mathbf{F}}_n = \frac{1}{n}\sum_{i=1}^n \mathbf{g}_i\mathbf{g}_i^T$
   - 与期望Fisher的差异：$\|\hat{\mathbf{F}}_n - \mathbf{F}\| = O_p(n^{-1/2})$
   - 小样本修正：使用bootstrap或jackknife估计

3. **非渐近区域**：
   - 远离最优解时，Gauss-Newton丢失的二阶项可能很大
   - 量化：$\|\mathbf{H} - \mathbf{H}_{GN}\| \leq L\|\mathbf{r}(\mathbf{w})\|$
   - 自适应策略：基于残差大小混合使用

### 1.1.3 实践中的统一框架

在实际应用中，我们可以将这些方法统一为求解线性系统：
$$\mathbf{G}_k \Delta\mathbf{w}_k = -\nabla f(\mathbf{w}_k)$$

其中 $\mathbf{G}_k$ 是广义曲率矩阵：
- Newton: $\mathbf{G}_k = \nabla^2 f(\mathbf{w}_k)$
- Gauss-Newton: $\mathbf{G}_k = \mathbf{J}_k^T\mathbf{J}_k$  
- Natural Gradient: $\mathbf{G}_k = \mathbf{F}_k + \lambda\mathbf{I}$ (带阻尼)
- Levenberg-Marquardt: $\mathbf{G}_k = \mathbf{J}_k^T\mathbf{J}_k + \lambda_k\text{diag}(\mathbf{J}_k^T\mathbf{J}_k)$

**深入分析：预条件子的选择**

这个统一框架的核心在于如何选择合适的预条件子 $\mathbf{G}_k$。关键考虑因素包括：

1. **正定性保证**：
   - 谱修正：$\mathbf{G}_k = \mathbf{U}\max(\mathbf{\Lambda}, \epsilon\mathbf{I})\mathbf{U}^T$
   - 对角加载：$\mathbf{G}_k + \lambda\mathbf{I}$，其中 $\lambda > -\lambda_{\min}(\mathbf{G}_k)$
   - Cholesky修正：尝试分解，失败时增加对角项

2. **条件数控制**：
   - 目标：$\kappa(\mathbf{G}_k) = \lambda_{\max}/\lambda_{\min} \leq \kappa_{\max}$
   - 谱截断：将小特征值替换为阈值
   - 预条件迭代法的收敛速度：$\rho \approx 1 - 2/(\sqrt{\kappa} + 1)$

3. **计算复杂度**：
   - 直接法：$O(n^3)$ for Cholesky分解
   - 迭代法：$O(n^2k)$ for $k$ 次CG迭代
   - 低秩方法：$O(nr^2)$ for 秩-$r$ 近似

4. **近似质量**：
   - 局部模型精度：$\|f(\mathbf{w}+\Delta\mathbf{w}) - m(\Delta\mathbf{w})\| \leq O(\|\Delta\mathbf{w}\|^3)$
   - 曲率近似误差：$\|\mathbf{G}_k - \nabla^2 f\| \leq \epsilon_G$
   - 收敛速度影响：超线性 vs 线性收敛

**高级变体与扩展**：

1. **Kronecker-Factored Curvature (K-FAC)**：
   $$\mathbf{G}_k = \mathbf{A}_k \otimes \mathbf{B}_k + \lambda\mathbf{I}$$
   
   优势分析：
   - 存储：从 $O(n^2)$ 降至 $O(n)$
   - 求逆：利用 $(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$
   - 近似质量：对具有Kronecker结构的网络是精确的

2. **Quasi-Newton预条件**：
   $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k\mathbf{y}_k^T}{\mathbf{y}_k^T\mathbf{s}_k} - \frac{\mathbf{B}_k\mathbf{s}_k\mathbf{s}_k^T\mathbf{B}_k}{\mathbf{s}_k^T\mathbf{B}_k\mathbf{s}_k}$$
   
   其中 $\mathbf{s}_k = \mathbf{w}_{k+1} - \mathbf{w}_k$, $\mathbf{y}_k = \nabla f_{k+1} - \nabla f_k$

3. **Sketched Curvature**：
   $$\mathbf{G}_k = \mathbf{S}_k^T\nabla^2 f(\mathbf{w}_k)\mathbf{S}_k$$
   
   随机投影选择：
   - Gaussian sketching: $\mathbf{S}_{ij} \sim \mathcal{N}(0, 1/d)$
   - Sparse embedding: 稀疏 $\{-1, 0, +1\}$ 矩阵
   - Subsampled randomized Hadamard transform (SRHT)

**自适应框架的数学基础**：

考虑带权重的曲率组合：
$$\mathbf{G}_k = \sum_{i=1}^m \alpha_i^{(k)} \mathbf{G}_i^{(k)}$$

其中权重 $\alpha_i^{(k)}$ 可通过以下方式确定：

1. **贝叶斯方法**：
   - 先验：$p(\mathbf{G}) = \prod_i p(\mathbf{G}_i)^{\alpha_i}$
   - 后验更新：基于观测的步长质量
   - 计算：使用变分推断或MCMC

2. **在线学习**：
   - Regret最小化：$\min_{\alpha} \sum_{t=1}^T \ell_t(\alpha)$
   - 专家算法：每个曲率矩阵作为一个专家
   - 权重更新：指数权重或Follow-the-Leader

3. **谱分析**：
   - 特征值分解：$\mathbf{G}_i = \mathbf{U}_i\mathbf{\Lambda}_i\mathbf{U}_i^T$
   - 权重选择：基于条件数、谱gap等指标
   - 动态调整：追踪特征值变化

**实现考虑与优化**：

1. **数值稳定性技巧**：
   ```
   # 稳定的Cholesky分解
   while True:
       try:
           L = cholesky(G + diag_shift * I)
           break
       except:
           diag_shift *= 10
   ```

2. **高效线性求解**：
   - 预条件共轭梯度(PCG)
   - 多重网格方法
   - 不完全分解预条件子

3. **分布式实现**：
   - 数据并行：分片计算梯度和曲率
   - 模型并行：分块矩阵运算
   - 通信优化：梯度压缩和延迟更新

**研究线索**：
- 自适应选择曲率矩阵的元学习方法
- 结合不同曲率近似的混合算法理论分析
- 在非凸优化中的收敛性保证强化
- 基于硬件感知的曲率矩阵设计（GPU/TPU优化）
- 分布式环境下的曲率矩阵近似与通信优化
- 量子算法加速曲率矩阵计算的可能性
- 神经架构搜索(NAS)中的二阶方法应用

## 1.2 Fisher信息矩阵与Hessian的关系

### 1.2.1 理论联系

Fisher信息矩阵和Hessian之间存在深刻的数学联系，这种联系在概率模型的参数估计中尤为重要。

**定理 1.2**（Fisher-Hessian关系）
对于概率模型 $p(\mathbf{x}|\mathbf{w})$，负对数似然的期望Hessian等于Fisher信息矩阵：
$$\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}|\mathbf{w})}[\nabla^2(-\log p(\mathbf{x}|\mathbf{w}))] = \mathbf{F}(\mathbf{w})$$

**证明核心**：利用score function的性质
$$\mathbb{E}[\nabla \log p(\mathbf{x}|\mathbf{w})] = 0$$
$$\text{Var}[\nabla \log p(\mathbf{x}|\mathbf{w})] = \mathbf{F}(\mathbf{w})$$

**深层联系的多个视角**：

1. **信息几何视角**：
   - Fisher信息定义了参数空间的Riemannian度量
   - Hessian在该度量下是Levi-Civita联络的表示
   - 测地线方程：$\ddot{\mathbf{w}}^k + \Gamma_{ij}^k \dot{\mathbf{w}}^i \dot{\mathbf{w}}^j = 0$

2. **统计物理类比**：
   - Fisher信息 ~ 系统的"刚度"（对扰动的响应）
   - Hessian ~ 能量景观的局部曲率
   - 温度参数连接两者：$\mathbf{H} = \beta\mathbf{F} + \text{涨落项}$

3. **信息论解释**：
   - Fisher信息量化参数的可辨识性
   - Cramér-Rao界：$\text{Var}(\hat{\mathbf{w}}) \geq \mathbf{F}^{-1}$
   - 效率：估计量接近此界的程度

**推广到非标准情形**：

**定理 1.2a**（加权Fisher信息）
对于加权损失 $L(\mathbf{w}) = \mathbb{E}_{q(\mathbf{x})}[\ell(p(\mathbf{x}|\mathbf{w}))]$：
$$\nabla^2 L(\mathbf{w}) = \mathbf{F}_q(\mathbf{w}) + \text{bias term}$$
其中 $\mathbf{F}_q$ 是关于分布 $q$ 的加权Fisher信息。

**定理 1.2b**（条件Fisher信息）
对于条件模型 $p(\mathbf{y}|\mathbf{x}, \mathbf{w})$：
$$\mathbf{F}_{cond} = \mathbb{E}_{\mathbf{x},\mathbf{y}}[\nabla_{\mathbf{w}} \log p(\mathbf{y}|\mathbf{x},\mathbf{w}) \nabla_{\mathbf{w}} \log p(\mathbf{y}|\mathbf{x},\mathbf{w})^T]$$

### 1.2.2 实际差异与近似策略

尽管理论上存在联系，但在实践中二者常有显著差异：

1. **有限样本效应**：
   - 样本Hessian：$\hat{\mathbf{H}}_n = \frac{1}{n}\sum_{i=1}^n \nabla^2 \ell_i(\mathbf{w})$
   - 偏差：$\mathbb{E}[\hat{\mathbf{H}}_n] - \mathbf{F} = O(n^{-1})$
   - 方差：$\text{Var}(\hat{\mathbf{H}}_n) = O(n^{-1})$
   - 修正方法：Bartlett校正、Bootstrap方差估计

2. **模型误设(Misspecification)**：
   - 真实分布：$q(\mathbf{x})$ vs 模型族：$\{p(\mathbf{x}|\mathbf{w})\}$
   - Sandwich估计量：$\mathbf{V} = \mathbf{H}^{-1}\mathbf{F}\mathbf{H}^{-1}$
   - 稳健推断：使用$\mathbf{V}$而非$\mathbf{H}^{-1}$或$\mathbf{F}^{-1}$
   - 诊断：比较$\|\mathbf{H} - \mathbf{F}\|$的大小

3. **非凸性与负曲率**：
   - Hessian谱：可能包含负特征值
   - Fisher信息：始终半正定（$\mathbf{F} \succeq 0$）
   - 修正策略：
     * 谱截断：$\mathbf{H}^+ = \sum_{\lambda_i > 0} \lambda_i \mathbf{u}_i\mathbf{u}_i^T$
     * 绝对值修正：$|\mathbf{H}| = \mathbf{U}|\mathbf{\Lambda}|\mathbf{U}^T$
     * 凸组合：$\alpha\mathbf{H} + (1-\alpha)\mathbf{F}$

**高级近似技术**：

1. **Empirical Fisher变体**：
   
   **标准Empirical Fisher**：
   $$\hat{\mathbf{F}} = \frac{1}{N}\sum_{i=1}^N \mathbf{g}_i\mathbf{g}_i^T, \quad \mathbf{g}_i = \nabla \log p(\mathbf{x}_i|\mathbf{w})$$
   
   **Centered Empirical Fisher**：
   $$\hat{\mathbf{F}}_c = \frac{1}{N}\sum_{i=1}^N (\mathbf{g}_i - \bar{\mathbf{g}})(\mathbf{g}_i - \bar{\mathbf{g}})^T$$
   
   **Natural Empirical Fisher**（用于深度学习）：
   $$\hat{\mathbf{F}}_{nat} = \frac{1}{N}\sum_{i=1}^N \nabla_{\mathbf{w}} \log p(\mathbf{y}_i|\mathbf{x}_i,\mathbf{w}) \nabla_{\mathbf{w}} \log p(\mathbf{y}_i|\mathbf{x}_i,\mathbf{w})^T$$

2. **Generalized Gauss-Newton (GGN)**：
   
   对于复合损失 $L(\mathbf{w}) = \ell(f(\mathbf{w}))$：
   $$\mathbf{G}_{GGN} = \mathbf{J}^T \nabla^2\ell(f(\mathbf{w})) \mathbf{J}$$
   
   特殊情况：
   - 平方损失：$\mathbf{G}_{GGN} = \mathbf{J}^T\mathbf{J}$（标准Gauss-Newton）
   - 交叉熵：$\mathbf{G}_{GGN} = \mathbf{J}^T\text{diag}(p(1-p))\mathbf{J}$

3. **自适应混合策略**：
   
   **动态权重方案**：
   $$\mathbf{G}_k = \alpha_k\mathbf{H}_k + (1-\alpha_k)\mathbf{F}_k$$
   
   权重选择准则：
   - 基于条件数：$\alpha_k = \min(1, \kappa_{max}/\kappa(\mathbf{H}_k))$
   - 基于进展：$\alpha_k = \rho_k$（实际vs预测下降比）
   - 基于噪声水平：$\alpha_k = 1/(1 + \sigma_k^2)$

### 1.2.3 计算效率考虑

Fisher信息矩阵的计算通常比完整Hessian更高效：

**基础优势**：
- 只需要一阶导数（通过外积）
- 可以使用Monte Carlo近似
- 适合分布式计算和在线更新
- 自动保证半正定性

**计算复杂度对比**：
| 方法 | 时间复杂度 | 空间复杂度 | 并行性 |
|------|------------|------------|--------|
| Full Hessian | $O(n^2 \cdot \text{cost}(\nabla^2))$ | $O(n^2)$ | 困难 |
| Fisher (外积) | $O(n^2 \cdot \text{cost}(\nabla))$ | $O(n^2)$ | 容易 |
| Block Fisher | $O(b \cdot n^2/b^2)$ | $O(n^2/b)$ | 高度并行 |
| Low-rank Fisher | $O(nr \cdot \text{cost}(\nabla))$ | $O(nr)$ | 容易 |

**高级计算优化**：

1. **结构化Fisher计算**：
   
   **层级分解**（用于深度网络）：
   $$\mathbf{F} = \begin{pmatrix}
   \mathbf{F}_{11} & \mathbf{F}_{12} & \cdots \\
   \mathbf{F}_{21} & \mathbf{F}_{22} & \cdots \\
   \vdots & \vdots & \ddots
   \end{pmatrix}$$
   
   - 块对角近似：忽略层间相关性
   - 三对角近似：只保留相邻层
   - Kronecker近似：$\mathbf{F}_l \approx \mathbf{A}_l \otimes \mathbf{G}_l$

2. **动量方法加速Fisher估计**：
   
   **指数移动平均**：
   $$\mathbf{F}_t = \beta\mathbf{F}_{t-1} + (1-\beta)\mathbf{g}_t\mathbf{g}_t^T$$
   
   **二阶动量**（类似Adam）：
   $$\mathbf{M}_t = \beta_1\mathbf{M}_{t-1} + (1-\beta_1)\mathbf{g}_t$$
   $$\mathbf{F}_t = \beta_2\mathbf{F}_{t-1} + (1-\beta_2)\mathbf{g}_t\mathbf{g}_t^T$$

3. **采样策略优化**：
   
   **重要性采样**：
   - 选择信息量大的样本：$p_i \propto \|\mathbf{g}_i\|^2$
   - 无偏估计：$\hat{\mathbf{F}} = \sum_{i \in S} \frac{1}{Np_i}\mathbf{g}_i\mathbf{g}_i^T$
   
   **分层采样**：
   - 按梯度范数分层
   - 每层内均匀采样
   - 减少方差同时保持代表性

**高效计算策略**：

1. **分块计算与稀疏性利用**：
   对于结构化模型（如深度网络），Fisher信息矩阵常呈现分块对角或块三对角结构：
   $$\mathbf{F} = \begin{pmatrix}
   \mathbf{F}_{11} & \mathbf{F}_{12} & \cdots \\
   \mathbf{F}_{21} & \mathbf{F}_{22} & \cdots \\
   \vdots & \vdots & \ddots
   \end{pmatrix}$$
   
   可以利用的计算优化：
   - 只计算和存储非零块
   - 使用稀疏线性代数库（如`scipy.sparse.linalg`）
   - 并行计算不同块

2. **低秩近似技术**：
   
   **SVD分解**：$\mathbf{F} \approx \mathbf{U}_k\mathbf{\Sigma}_k\mathbf{U}_k^T$
   - 只保留前 $k$ 个主成分
   - 存储复杂度从 $O(n^2)$ 降至 $O(nk)$
   
   **Nyström近似**：
   $$\mathbf{F} \approx \mathbf{F}_{:,S}\mathbf{F}_{S,S}^{-1}\mathbf{F}_{S,:}$$
   其中 $S$ 是随机选择的列子集。
   
   **梯度外积的流式更新**：
   $$\mathbf{F}_t = (1-\beta)\mathbf{F}_{t-1} + \beta \mathbf{g}_t\mathbf{g}_t^T$$
   使用指数移动平均维护低秩分解。

3. **Monte Carlo Fisher估计**：
   
   **无偏估计器**：
   $$\hat{\mathbf{F}} = \frac{1}{m}\sum_{i=1}^m \nabla_{\mathbf{w}} \log p(\mathbf{x}_i|\mathbf{w}) \nabla_{\mathbf{w}} \log p(\mathbf{x}_i|\mathbf{w})^T$$
   
   **方差缩减技术**：
   - 重要性采样：选择信息量大的样本
   - 控制变量：$\hat{\mathbf{F}}_{CV} = \hat{\mathbf{F}} + c(\mathbf{F}_0 - \hat{\mathbf{F}}_0)$
   - SVRG类方法：周期性计算完整Fisher作为锚点

4. **分布式计算框架**：
   
   **数据并行**：每个节点计算局部Fisher，然后聚合
   $$\mathbf{F} = \frac{1}{N}\sum_{j=1}^{P} n_j \mathbf{F}_j$$
   
   **模型并行**：分块计算Fisher的不同部分
   - 通信优化：只传输必要的边界信息
   - 异步更新：容忍一定的延迟

**与Hessian计算的混合策略**：

在某些情况下，结合Fisher信息和Hessian信息可以获得更好的性能：

1. **Generalized Gauss-Newton (GGN)**：
   $$\mathbf{G}_{GGN} = \mathbf{J}^T\mathbf{H}_{\text{loss}}\mathbf{J}$$
   其中 $\mathbf{H}_{\text{loss}}$ 是损失函数关于输出的Hessian。

2. **Fisher-Hessian平均**：
   $$\mathbf{G} = \alpha\mathbf{F} + (1-\alpha)\mathbf{H}^+$$
   其中 $\mathbf{H}^+$ 是Hessian的正定部分。

3. **条件切换**：
   - 当 $\|\nabla f\|$ 大时使用Fisher（更稳定）
   - 接近最优时使用Hessian（更精确）

**研究线索**：
- Fisher信息矩阵的低秩近似与压缩
- 基于梯度历史的Fisher估计
- 量子Fisher信息在经典优化中的应用
- 神经正切核(NTK)与Fisher信息的联系
- 自适应采样策略for Fisher估计
- 隐私保护的分布式Fisher计算

## 1.3 Trust Region方法在深度学习中的复兴

### 1.3.1 从经典到现代：Trust Region的演变

Trust Region方法通过限制步长在模型可信的区域内，提供了比线搜索更稳健的全局化策略。在深度学习中，这种方法正经历复兴。

**经典Trust Region子问题**：
$$\min_{\|\Delta\mathbf{w}\| \leq \delta} f(\mathbf{w}) + \nabla f(\mathbf{w})^T \Delta\mathbf{w} + \frac{1}{2} \Delta\mathbf{w}^T \mathbf{H} \Delta\mathbf{w}$$

### 1.3.2 深度学习中的适应性改进

**挑战与解决方案**：

1. **高维问题**：使用Krylov子空间方法（如CG-Steihaug）
2. **随机性**：结合方差缩减技术（SVRG, SARAH）
3. **非凸性**：自适应调整trust region形状

**现代变体**：
- **TRON** (Trust Region Newton)：专为大规模问题设计
- **TR-APG** (Trust Region Accelerated Proximal Gradient)：结合一阶加速
- **Saddle-Free Newton**：检测并逃离鞍点的trust region方法

### 1.3.3 与其他方法的结合

Trust Region框架可以与多种技术结合：
- **与Natural Gradient结合**：使用Fisher度量定义trust region
- **与动量方法结合**：在trust region内加入动量项
- **与自适应学习率结合**：per-parameter trust region (类似Adam)

**高级结合策略**：

1. **Riemannian Trust Region**：
   在流形优化中，trust region的定义需要考虑流形的几何结构：
   $$\min_{\eta \in T_x\mathcal{M}} m(\eta) \quad \text{s.t.} \quad \|\eta\|_x \leq \delta$$
   其中 $T_x\mathcal{M}$ 是切空间，$\|\cdot\|_x$ 是Riemannian度量。
   
   应用场景：
   - 正定矩阵优化（使用Log-Euclidean度量）
   - Stiefel流形上的优化（正交约束）
   - 低秩矩阵流形（固定秩约束）

2. **Stochastic Trust Region**：
   处理随机梯度和Hessian估计的不确定性：
   
   **概率Trust Region**：
   $$\mathbb{P}[\|\Delta\mathbf{w}\| \leq \delta] \geq 1-\epsilon$$
   
   **自适应半径调整**：
   $$\delta_{k+1} = \begin{cases}
   \gamma_{\text{inc}} \delta_k & \text{if } \rho_k > \eta_1 \text{ and } \|\Delta\mathbf{w}_k\| = \delta_k \\
   \gamma_{\text{dec}} \delta_k & \text{if } \rho_k < \eta_2 \\
   \delta_k & \text{otherwise}
   \end{cases}$$
   
   其中 $\rho_k$ 是模型预测质量的随机估计。

3. **Adaptive Shape Trust Region**：
   使用椭球而非球形trust region：
   $$\{\Delta\mathbf{w} : \Delta\mathbf{w}^T\mathbf{M}_k\Delta\mathbf{w} \leq \delta^2\}$$
   
   **度量矩阵选择**：
   - 对角缩放：$\mathbf{M}_k = \text{diag}(|\nabla f_i|^{\alpha})$
   - 曲率感知：$\mathbf{M}_k = (\mathbf{H}_k + \lambda\mathbf{I})^{1/2}$
   - 历史信息：$\mathbf{M}_k = \sum_{i=1}^t \beta^{t-i}\mathbf{g}_i\mathbf{g}_i^T$

4. **Momentum-Enhanced Trust Region**：
   结合动量加速的trust region方法：
   
   **Heavy-ball Trust Region**：
   $$\min_{\Delta\mathbf{w}} m(\Delta\mathbf{w}) + \mu \langle \Delta\mathbf{w}, \mathbf{v}_{k-1} \rangle$$
   $$\text{s.t.} \quad \|\Delta\mathbf{w}\| \leq \delta$$
   
   其中 $\mathbf{v}_{k-1}$ 是历史动量。
   
   **Nesterov-style预测**：
   先进行动量步，然后在预测点构建trust region模型。

5. **Multi-level Trust Region**：
   针对具有层次结构的问题（如深度网络）：
   
   - 不同层使用不同的trust region半径
   - 基于层的敏感度自适应调整
   - 考虑层间相互作用的耦合trust region

**实现优化技巧**：

1. **Subproblem求解加速**：
   - 使用Lanczos方法的早停
   - 基于历史信息的暖启动
   - 近似求解的理论保证

2. **并行化策略**：
   - 多个trust region半径的并行尝试
   - 分布式子问题求解
   - 异步trust region更新

3. **内存效率**：
   - 使用implicit representations避免存储完整Hessian
   - Hessian-free实现using only Hessian-vector products
   - 增量式trust region模型更新

**研究线索**：
- 基于局部曲率的自适应trust region形状
- 分布式trust region算法的通信效率
- 隐式trust region方法（通过正则化实现）
- 非欧几里德空间的trust region理论
- 与强化学习中的PPO/TRPO的理论联系
- 量子优化中的trust region类比

## 1.4 鞍点逃逸的理论与实践

### 1.4.1 鞍点的数学刻画

在高维非凸优化中，鞍点（特征值有正有负的驻点）比局部极小值更常见。严格鞍点满足：
$$\lambda_{\min}(\nabla^2 f(\mathbf{w}^*)) < -\epsilon < 0$$

### 1.4.2 逃逸机制的理论分析

**定理 1.3**（Perturbed Gradient Descent的逃逸保证）
在温和条件下，带扰动的梯度下降能以高概率在多项式时间内逃离严格鞍点。

**主要逃逸策略**：
1. **随机扰动**：在梯度中加入噪声
2. **负曲率方向**：显式计算并利用负特征值方向
3. **二阶方法**：Newton类方法自然逃离鞍点

### 1.4.3 实用算法与技巧

**Negative Curvature Descent**：
1. 使用Lanczos迭代找到近似最小特征值方向
2. 当检测到负曲率时，沿该方向下降
3. 否则执行标准梯度步

**加速逃逸技术**：
- **Neon2**: 结合负曲率和加速梯度
- **Noisy Natural Gradient**: 在Natural Gradient中加入各向异性噪声
- **Cubic Regularization**: 使用三次正则化自动处理负曲率

### 1.4.4 深度网络中的特殊考虑

深度网络的鞍点具有特殊结构：
- 大部分鞍点是由对称性导致的
- 网络深度增加时鞍点数量指数增长
- 但大部分鞍点的负特征值数量较少

**鞍点结构的深入分析**：

1. **对称性导致的鞍点**：
   
   **置换对称性**：神经元的可交换性导致大量等价鞍点
   - 隐层神经元的重排列
   - 权重符号的翻转（对于对称激活函数）
   - 这类鞍点的Hessian具有大量零特征值
   
   **缩放对称性**：在某些架构中存在
   $$\mathbf{W}_2 \leftarrow \alpha\mathbf{W}_2, \quad \mathbf{W}_1 \leftarrow \mathbf{W}_1/\alpha$$
   保持网络功能不变但创建鞍点轨迹。

2. **鞍点的谱特性**：
   
   **经验观察**：
   - 负特征值数量通常为 $O(\sqrt{n})$，其中 $n$ 是参数数量
   - 大部分特征值聚集在零附近
   - 极端特征值（最大正/最小负）主导优化动态
   
   **理论刻画**：
   对于随机初始化的网络，Hessian谱遵循：
   $$\rho(\lambda) \approx \frac{1}{2\pi\sigma^2}\sqrt{4\sigma^2 - \lambda^2}$$
   即Wigner半圆律的变体。

3. **高效逃逸策略**：
   
   **Power Method的加速变体**：
   ```
   输入: 当前点 w, 容忍度 ε
   1. 初始化随机向量 v
   2. for k = 1 to K:
      a. ṽ = Hv (使用Hessian-vector product)
      b. μ = v^T ṽ / ||v||²
      c. if μ < -ε: return v as escape direction
      d. v = ṽ / ||ṽ||
   3. return null (no negative curvature found)
   ```
   
   **Lanczos加速**：
   - 构建Krylov子空间 $\mathcal{K}_k(\mathbf{H}, \mathbf{v})$
   - 在子空间中找最小特征值
   - 通常 $k \ll n$ 即可找到好的近似

4. **与优化器的集成**：
   
   **SGD with Negative Curvature**：
   $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \mathbf{g}_t - \alpha_t \mathbf{v}_t$$
   其中 $\mathbf{v}_t$ 是负曲率方向（如果存在）。
   
   **Adam-NC (Adam with Negative Curvature)**：
   - 维护梯度和负曲率方向的独立动量
   - 自适应混合两个方向
   - 在鞍点附近自动增大负曲率权重

5. **理论保证的增强**：
   
   **Fast Escaping via Coupling**：
   同时运行多个带不同扰动的副本：
   - 耦合强度随时间衰减
   - 最快逃离的副本带动其他副本
   - 理论上改进逃逸时间复杂度
   
   **Correlated Negative Curvature**：
   利用参数间的相关结构：
   - 块对角近似找到相关的负曲率方向
   - 减少计算同时保持逃逸效率

**高级技术与前沿方向**：

1. **拓扑数据分析(TDA)方法**：
   - 使用持续同调(Persistent Homology)分析损失地形
   - 识别鞍点的连通分量
   - 设计拓扑感知的逃逸路径

2. **随机矩阵理论的应用**：
   - 预测不同深度/宽度下的鞍点密度
   - 设计架构以减少坏鞍点
   - 理解over-parameterization的益处

3. **与泛化的联系**：
   - Flat vs Sharp鞍点的泛化差异
   - 逃逸路径的隐式正则化
   - PAC-Bayes框架下的分析

**研究线索**：
- 利用网络结构加速鞍点逃逸
- 鞍点的拓扑结构与泛化性能的关系
- 量子退火启发的逃逸算法
- 神经架构搜索中的鞍点考虑
- 联邦学习环境下的分布式鞍点逃逸
- 对抗鲁棒性与鞍点结构的关系

## 本章小结

本章建立了二阶优化方法的统一框架，核心要点包括：

1. **统一视角**：Newton法、Gauss-Newton法和Natural Gradient都可视为使用不同曲率矩阵的二阶方法
2. **Fisher-Hessian联系**：在期望意义下和特定条件下，Fisher信息矩阵等价于Hessian
3. **Trust Region复兴**：现代深度学习重新发现了Trust Region方法的价值，特别是在处理非凸性方面
4. **鞍点逃逸**：理论保证与实用算法的结合，使二阶方法在非凸优化中更加可靠

关键公式回顾：
- 广义更新规则：$\mathbf{G}_k \Delta\mathbf{w}_k = -\nabla f(\mathbf{w}_k)$
- Fisher信息矩阵：$\mathbf{F} = \mathbb{E}[\nabla \log p \nabla \log p^T]$
- Trust Region子问题：$\min_{\|\Delta\mathbf{w}\| \leq \delta} m(\Delta\mathbf{w})$
- 严格鞍点条件：$\lambda_{\min}(\nabla^2 f) < -\epsilon$

## 练习题

### 基础题

**习题1.1** 证明对于线性回归问题 $f(\mathbf{w}) = \frac{1}{2}\|\mathbf{Xw} - \mathbf{y}\|^2$，Gauss-Newton法在一步内收敛到全局最优解。

*Hint*: 计算Gauss-Newton Hessian并验证它与真实Hessian相等。

<details>
<summary>答案</summary>

对于线性回归，残差 $\mathbf{r}(\mathbf{w}) = \mathbf{Xw} - \mathbf{y}$，Jacobian为 $\mathbf{J} = \mathbf{X}$。
因此 $\mathbf{H}_{GN} = \mathbf{X}^T\mathbf{X} = \nabla^2 f$，即Gauss-Newton Hessian等于真实Hessian。
由于目标函数是凸二次的，Newton法（此时等价于Gauss-Newton）一步收敛。
</details>

**习题1.2** 对于Softmax回归，推导Fisher信息矩阵的显式表达式。

*Hint*: 利用Softmax函数的性质和分类分布的score function。

<details>
<summary>答案</summary>

设Softmax概率为 $p_k = \frac{\exp(\mathbf{w}_k^T\mathbf{x})}{\sum_j \exp(\mathbf{w}_j^T\mathbf{x})}$。
Fisher信息矩阵的块 $(i,j)$ 为：
- 当 $i = j$: $\mathbf{F}_{ii} = \mathbb{E}[p_i(1-p_i)\mathbf{x}\mathbf{x}^T]$
- 当 $i \neq j$: $\mathbf{F}_{ij} = -\mathbb{E}[p_ip_j\mathbf{x}\mathbf{x}^T]$
</details>

**习题1.3** 实现并比较Steepest Descent和Conjugate Gradient求解Trust Region子问题的效率。

*Hint*: 使用Steihaug-Toint CG算法，注意边界情况的处理。

<details>
<summary>答案</summary>

关键实现要点：
1. CG迭代中监测是否到达trust region边界
2. 如果到达边界，求解二次方程找到边界上的交点
3. 检测负曲率方向，此时沿该方向到达边界
4. 对于典型问题，CG通常在远少于 $n$ 次迭代内收敛
</details>

### 挑战题

**习题1.4** 设计一个自适应算法，根据局部曲率自动在Newton、Gauss-Newton和Natural Gradient之间切换。

*Hint*: 考虑使用条件数、负特征值比例等指标。

<details>
<summary>答案</summary>

算法框架：
1. 计算便宜的曲率指标（如梯度范数变化率）
2. 如果检测到强非凸性（负特征值），使用修正Newton
3. 对于近似凸的区域，比较GN近似误差
4. 当模型接近概率解释时，考虑Natural Gradient
5. 使用滑动平均平滑切换决策
</details>

**习题1.5** 分析在什么条件下，Trust Region方法的收敛速度优于线搜索方法。构造具体例子。

*Hint*: 考虑病态Hessian和非凸函数的情况。

<details>
<summary>答案</summary>

Trust Region优势情形：
1. 病态问题：当Hessian条件数很大时，线搜索可能需要很多次函数评估
2. 强非凸性：Trust Region能更好处理负曲率
3. 噪声梯度：Trust Region对步长的鲁棒性更好
具体例子：Rosenbrock函数在狭长谷底，Trust Region能自适应调整步长方向
</details>

**习题1.6** 证明对于过参数化的神经网络，在适当初始化下，Gauss-Newton法的更新方向接近Natural Gradient。

*Hint*: 利用神经切线核(NTK)理论。

<details>
<summary>答案</summary>

在宽度趋于无穷的极限下：
1. 网络函数线性化，Jacobian保持近似不变
2. Gauss-Newton矩阵 $\mathbf{J}^T\mathbf{J}$ 收敛到NTK
3. 在分类任务中，Fisher信息也收敛到相同的核
4. 因此两种方法给出相近的更新方向
关键假设：适当的初始化scale和输出层的参数化
</details>

**习题1.7**（开放问题）如何设计一个统一的预条件子，能够自适应地在不同优化阶段提供最合适的曲率信息？

*Hint*: 考虑元学习、在线学习理论和矩阵学习。

<details>
<summary>答案</summary>

研究方向：
1. 基于历史信息的在线曲率矩阵学习
2. 使用强化学习选择预条件策略
3. 低秩加对角结构的自适应分解
4. 考虑计算预算的最优预条件子设计
5. 理论分析：regret bounds和收敛性保证
</details>

**习题1.8** 探讨量子计算对二阶优化方法可能带来的加速。特别关注矩阵求逆和特征值计算。

*Hint*: 研究HHL算法和量子相位估计。

<details>
<summary>答案</summary>

潜在加速点：
1. HHL算法：在特定条件下指数加速线性系统求解
2. 量子相位估计：加速特征值计算，有助于鞍点检测
3. 量子采样：改进随机二阶方法的方差
挑战：量子态制备、读出开销、有限的量子比特数
现实考虑：近期量子设备上的变分量子算法
</details>

## 常见陷阱与错误

1. **数值不稳定**
   - 错误：直接求逆病态矩阵
   - 正确：使用Cholesky分解或CG迭代，加入适当正则化

2. **忽视计算复杂度**
   - 错误：在高维问题中构造完整Hessian
   - 正确：使用Hessian-vector product，避免显式构造

3. **Trust Region参数设置**
   - 错误：固定trust region半径
   - 正确：基于模型预测质量自适应调整

4. **鞍点检测误判**
   - 错误：仅依据梯度范数判断
   - 正确：结合梯度范数和最小特征值信息

5. **Natural Gradient实现错误**
   - 错误：使用批量Fisher而非期望Fisher
   - 正确：理解Empirical Fisher vs True Fisher的区别

## 最佳实践检查清单

### 算法选择
- [ ] 问题是否有概率解释？考虑Natural Gradient
- [ ] 是否是最小二乘结构？优先Gauss-Newton
- [ ] 计算预算如何？权衡精确度与效率
- [ ] 非凸程度？需要鞍点逃逸机制

### 实现细节
- [ ] 使用迭代法而非直接求逆
- [ ] 实现Hessian-vector product而非完整矩阵
- [ ] 加入数值稳定性保护（条件数检查）
- [ ] 监控收敛诊断指标

### 性能优化
- [ ] 利用问题结构（稀疏性、低秩性）
- [ ] 实现高效的线性代数后端
- [ ] 考虑混合精度计算
- [ ] 并行化矩阵运算

### 理论保证
- [ ] 验证收敛条件是否满足
- [ ] 检查步长是否在理论界内
- [ ] 监控优化轨迹的稳定性
- [ ] 记录关键诊断信息供分析