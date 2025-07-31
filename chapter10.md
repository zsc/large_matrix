# 第10章：Riemannian优化基础

许多大规模矩阵优化问题天然地定义在具有特定几何结构的约束集上。传统的欧氏空间优化方法需要反复投影以维持约束，而Riemannian优化直接在约束流形上工作，将约束隐式地编码在流形的几何结构中。本章介绍流形优化的核心概念，重点关注在机器学习和数值计算中常见的矩阵流形。我们将探讨如何利用流形的内在几何结构设计高效的优化算法，以及这些方法在低秩矩阵补全等实际问题中的应用。

## 10.1 矩阵流形上的几何结构

矩阵流形是满足特定代数约束的矩阵集合，配备了使其成为光滑流形的微分结构。理解这些流形的几何性质是设计高效优化算法的基础。

### 10.1.1 常见的矩阵流形

**Stiefel流形** $\mathcal{V}_k(\mathbb{R}^n)$：所有$n \times k$列正交矩阵的集合
$$\mathcal{V}_k(\mathbb{R}^n) = \{\mathbf{X} \in \mathbb{R}^{n \times k} : \mathbf{X}^T\mathbf{X} = \mathbf{I}_k\}$$

当$k = n$时，Stiefel流形退化为正交群$O(n)$。Stiefel流形在主成分分析、正交Procrustes问题和特征值计算中广泛应用。

**Grassmann流形** $\mathcal{G}_k(\mathbb{R}^n)$：$\mathbb{R}^n$中所有$k$维子空间的集合。可以视为Stiefel流形在正交等价关系下的商空间：
$$\mathcal{G}_k(\mathbb{R}^n) = \mathcal{V}_k(\mathbb{R}^n) / O(k)$$

Grassmann流形在子空间追踪、降维和计算机视觉中扮演重要角色。

**固定秩矩阵流形** $\mathcal{M}_r^{m \times n}$：所有秩为$r$的$m \times n$矩阵的集合
$$\mathcal{M}_r^{m \times n} = \{\mathbf{X} \in \mathbb{R}^{m \times n} : \text{rank}(\mathbf{X}) = r\}$$

这是一个$(m + n - r)r$维的嵌入子流形，在低秩矩阵补全和逼近中至关重要。

**对称正定矩阵锥** $\mathcal{S}_{++}^n$：所有$n \times n$对称正定矩阵的集合。虽然技术上是开集而非流形，但可以赋予其Riemannian几何结构，在协方差矩阵估计和度量学习中应用广泛。

### 10.1.2 切空间的刻画

切空间$T_{\mathbf{X}}\mathcal{M}$是流形$\mathcal{M}$在点$\mathbf{X}$处的线性化，包含了所有可行的"无穷小移动方向"。

**Stiefel流形的切空间**：
$$T_{\mathbf{X}}\mathcal{V}_k(\mathbb{R}^n) = \{\boldsymbol{\xi} \in \mathbb{R}^{n \times k} : \mathbf{X}^T\boldsymbol{\xi} + \boldsymbol{\xi}^T\mathbf{X} = \mathbf{0}\}$$

这个条件确保$\mathbf{X} + t\boldsymbol{\xi}$在一阶近似下保持列正交性。

**固定秩流形的切空间**：对于$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$的SVD分解，
$$T_{\mathbf{X}}\mathcal{M}_r^{m \times n} = \{\mathbf{U}\mathbf{M}\mathbf{V}^T + \mathbf{U}_\perp\mathbf{N}\mathbf{V}^T + \mathbf{U}\mathbf{P}^T\mathbf{V}_\perp^T : \mathbf{M} \in \mathbb{R}^{r \times r}, \mathbf{N} \in \mathbb{R}^{(m-r) \times r}, \mathbf{P} \in \mathbb{R}^{(n-r) \times r}\}$$

其中$\mathbf{U}_\perp$和$\mathbf{V}_\perp$分别是$\mathbf{U}$和$\mathbf{V}$的正交补。

### 10.1.3 Riemannian度量

Riemannian度量为每个切空间赋予内积结构，使得我们能够测量切向量的长度和夹角。选择合适的度量对算法性能影响重大。

**标准度量**：最常用的是从环境空间继承的Euclidean度量
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle \boldsymbol{\xi}, \boldsymbol{\eta} \rangle_F = \text{tr}(\boldsymbol{\xi}^T\boldsymbol{\eta})$$

**规范不变度量**：对于Grassmann流形，可以定义对子空间基的选择不变的度量
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\boldsymbol{\xi}^T(\mathbf{I} - \mathbf{X}\mathbf{X}^T)\boldsymbol{\eta})$$

**信息几何度量**：对于正定矩阵锥，Fisher信息度量
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\mathbf{X}^{-1}\boldsymbol{\xi}\mathbf{X}^{-1}\boldsymbol{\eta})$$
提供了对矩阵求逆和缩放的自然不变性。

### 10.1.4 测地线与指数映射

测地线是流形上的"直线"，即局部最短路径。指数映射$\exp_{\mathbf{X}}: T_{\mathbf{X}}\mathcal{M} \to \mathcal{M}$将切向量映射到沿测地线的点。

对于配备标准度量的Stiefel流形，指数映射有闭式表达：
$$\exp_{\mathbf{X}}(\boldsymbol{\xi}) = [\mathbf{X}, \mathbf{Q}]\exp\left(\begin{bmatrix} \mathbf{A} & -\mathbf{R}^T \\ \mathbf{R} & \mathbf{0} \end{bmatrix}\right)\begin{bmatrix} \mathbf{I}_k \\ \mathbf{0} \end{bmatrix}$$
其中$\mathbf{Q}\mathbf{R} = (\mathbf{I} - \mathbf{X}\mathbf{X}^T)\boldsymbol{\xi}$是QR分解，$\mathbf{A} = \mathbf{X}^T\boldsymbol{\xi}$。

然而，指数映射的计算通常开销较大，实践中常用更简单的回缩映射替代。

## 10.2 Riemannian梯度与Hessian

将欧氏空间的梯度和Hessian概念推广到流形上，是构建流形优化算法的关键步骤。

### 10.2.1 Riemannian梯度的计算

给定定义在流形$\mathcal{M}$上的光滑函数$f: \mathcal{M} \to \mathbb{R}$，其在点$\mathbf{X}$的Riemannian梯度$\text{grad} f(\mathbf{X})$是唯一满足以下条件的切向量：
$$g_{\mathbf{X}}(\text{grad} f(\mathbf{X}), \boldsymbol{\xi}) = \text{D}f(\mathbf{X})[\boldsymbol{\xi}], \quad \forall \boldsymbol{\xi} \in T_{\mathbf{X}}\mathcal{M}$$

其中$\text{D}f(\mathbf{X})[\boldsymbol{\xi}]$是$f$在$\mathbf{X}$沿方向$\boldsymbol{\xi}$的方向导数。

**投影方法**：若$f$可以扩展到环境空间，则Riemannian梯度是Euclidean梯度在切空间上的正交投影：
$$\text{grad} f(\mathbf{X}) = \text{Proj}_{T_{\mathbf{X}}\mathcal{M}}(\nabla f(\mathbf{X}))$$

对于Stiefel流形，投影算子为：
$$\text{Proj}_{T_{\mathbf{X}}\mathcal{V}}(\mathbf{Z}) = \mathbf{Z} - \mathbf{X}\text{sym}(\mathbf{X}^T\mathbf{Z})$$
其中$\text{sym}(\mathbf{A}) = (\mathbf{A} + \mathbf{A}^T)/2$。

### 10.2.2 Riemannian Hessian

Riemannian Hessian描述梯度场在流形上的变化率。对于切向量$\boldsymbol{\xi} \in T_{\mathbf{X}}\mathcal{M}$，Hessian作用定义为：
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \nabla_{\boldsymbol{\xi}} \text{grad} f(\mathbf{X})$$

其中$\nabla$是Levi-Civita联络。计算涉及三个步骤：

1. **梯度场的扩展**：将$\text{grad} f$从单点扩展到邻域
2. **方向导数**：计算扩展梯度场沿$\boldsymbol{\xi}$的导数
3. **投影**：将结果投影回切空间

对于嵌入子流形，Weingarten公式提供了实用的计算方法：
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \text{Proj}_{T_{\mathbf{X}}\mathcal{M}}(\text{D}(\nabla f)(\mathbf{X})[\boldsymbol{\xi}]) - \text{II}_{\mathbf{X}}(\boldsymbol{\xi}, \text{grad} f(\mathbf{X}))$$

其中$\text{II}_{\mathbf{X}}$是第二基本形式，刻画了流形的外在曲率。

### 10.2.3 二阶优化算法

**Riemannian Newton法**：求解切空间中的Newton方程
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\eta}] = -\text{grad} f(\mathbf{X})$$

然后沿回缩更新：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(\alpha_k \boldsymbol{\eta}_k)$

**Riemannian信赖域法**：在切空间中求解子问题
$$\min_{\boldsymbol{\eta} \in T_{\mathbf{X}}\mathcal{M}} m_{\mathbf{X}}(\boldsymbol{\eta}) = f(\mathbf{X}) + g_{\mathbf{X}}(\text{grad} f(\mathbf{X}), \boldsymbol{\eta}) + \frac{1}{2}g_{\mathbf{X}}(\text{Hess} f(\mathbf{X})[\boldsymbol{\eta}], \boldsymbol{\eta})$$
subject to $\|\boldsymbol{\eta}\|_{\mathbf{X}} \leq \Delta$

信赖域方法对非凸问题特别有效，能够利用负曲率方向。

### 10.2.4 曲率的影响

流形的内在曲率影响优化算法的收敛性。对于具有正截面曲率的流形（如球面），梯度流可能比欧氏情况收敛更快；而负曲率可能导致不稳定性。

**Rayleigh商优化**：在单位球面上最小化$f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$时，球面的正曲率加速了向最小特征向量的收敛。

**曲率修正**：某些算法通过估计和补偿曲率效应来改善性能，如Riemannian加速梯度方法中的动量项修正。

## 10.3 回缩与向量传输

虽然指数映射提供了理论上优雅的流形更新方式，但其计算成本往往过高。回缩（retraction）提供了计算友好的替代方案，而向量传输（vector transport）则是实现高效迭代算法的关键组件。

### 10.3.1 回缩映射

回缩$\mathcal{R}: T\mathcal{M} \to \mathcal{M}$是满足以下条件的光滑映射：
1. $\mathcal{R}_{\mathbf{X}}(\mathbf{0}) = \mathbf{X}$（保持原点）
2. $\text{D}\mathcal{R}_{\mathbf{X}}(\mathbf{0})[\boldsymbol{\xi}] = \boldsymbol{\xi}$（局部刚性）

回缩提供了从切空间"回到"流形的方式，且保证了一阶近似的正确性。

**常用回缩映射**：

*QR回缩*（Stiefel流形）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \text{qf}(\mathbf{X} + \boldsymbol{\xi})$$
其中$\text{qf}(\cdot)$表示QR分解的Q因子。计算复杂度$O(nk^2)$。

*极回缩*（Stiefel流形）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = (\mathbf{X} + \boldsymbol{\xi})(\mathbf{I} + \boldsymbol{\xi}^T\boldsymbol{\xi})^{-1/2}$$
基于极分解，保持了更多几何性质，但计算稍贵。

*投影回缩*（固定秩流形）：设$\mathbf{Y} = \mathbf{X} + \boldsymbol{\xi}$，计算其截断SVD
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \mathcal{P}_r(\mathbf{Y}) = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
其中$\mathcal{P}_r$是秩$r$最佳近似算子。

*测地回缩*（正定矩阵）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \mathbf{X}^{1/2}\exp(\mathbf{X}^{-1/2}\boldsymbol{\xi}\mathbf{X}^{-1/2})\mathbf{X}^{1/2}$$

### 10.3.2 向量传输

向量传输$\mathcal{T}_{\boldsymbol{\eta}}: T_{\mathbf{X}}\mathcal{M} \to T_{\mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})}\mathcal{M}$将切向量从一个切空间"运输"到另一个。这对于需要历史信息的算法（如共轭梯度、BFGS）至关重要。

**平行移动vs向量传输**：
- 平行移动保持向量在联络意义下"平行"，但计算复杂
- 向量传输只需满足较弱的相容性条件，允许更高效的实现

**常用向量传输**：

*投影传输*：最简单的方法，直接投影到目标切空间
$$\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}) = \text{Proj}_{T_{\mathbf{Y}}\mathcal{M}}(\boldsymbol{\zeta}), \quad \mathbf{Y} = \mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})$$

*微分回缩传输*：利用回缩的微分结构
$$\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}) = \text{D}\mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})[\boldsymbol{\zeta}]$$

对于QR回缩，有高效的实现利用QR分解的微分。

*平行移动的近似*：对于Stiefel流形，存在$O(nk^2)$的近似平行移动算法，在保持数值稳定性的同时提供良好的理论性质。

### 10.3.3 算法实现考虑

**计算复杂度权衡**：
- 指数映射：理论最优但计算最贵，$O(n^3)$或更高
- 极回缩：良好的几何性质，$O(nk^2 + k^3)$
- QR回缩：快速稳定，$O(nk^2)$
- 投影传输：最快但可能损失信息，$O(nk^2)$

**数值稳定性**：
- QR分解提供excellent数值稳定性
- 极分解在$\boldsymbol{\xi}$较大时可能需要迭代求解
- 正交化过程需要注意舍入误差累积

**算法选择指南**：
1. 对于大规模问题，优先考虑QR回缩和投影传输
2. 当问题具有特殊结构时（如稀疏性），设计专门的回缩
3. 在病态问题中，可能需要更精确的向量传输以保持收敛性

### 10.3.4 收敛性保证

使用回缩和向量传输的算法仍能保持良好的收敛性质：

**一阶方法**：Riemannian梯度下降with任意回缩保持$O(1/k)$收敛率（凸）和到驻点的收敛（非凸）。

**二阶方法**：Riemannian Newton和信赖域方法保持局部二次收敛，前提是回缩和向量传输满足：
- 回缩是二阶的：$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \exp_{\mathbf{X}}(\boldsymbol{\xi}) + O(\|\boldsymbol{\xi}\|^3)$
- 向量传输与回缩相容

**加速方法**：Riemannian加速梯度需要仔细设计向量传输以保持动量的正确性。recent工作表明，即使使用简单的投影传输，也能达到$O(1/k^2)$的收敛率。

## 10.4 在低秩矩阵补全中的应用

低秩矩阵补全是流形优化最成功的应用之一。给定部分观测的矩阵，目标是恢复完整的低秩矩阵。传统的核范数最小化方法需要完整的SVD，而流形方法直接在固定秩矩阵流形上工作，显著提升了可扩展性。

### 10.4.1 问题表述

给定观测集$\Omega \subset \{1,\ldots,m\} \times \{1,\ldots,n\}$和对应的观测值$\{M_{ij}: (i,j) \in \Omega\}$，低秩矩阵补全问题为：
$$\min_{\mathbf{X} \in \mathcal{M}_r^{m \times n}} f(\mathbf{X}) = \frac{1}{2}\sum_{(i,j) \in \Omega} (X_{ij} - M_{ij})^2$$

关键洞察：将问题限制在固定秩流形上避免了秩约束的组合特性。

**参数化选择**：
1. **SVD参数化**：$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，其中$\mathbf{U} \in \mathcal{V}_r(\mathbb{R}^m)$，$\mathbf{V} \in \mathcal{V}_r(\mathbb{R}^n)$，$\boldsymbol{\Sigma} \in \mathbb{R}^{r \times r}_{++}$
2. **因子参数化**：$\mathbf{X} = \mathbf{L}\mathbf{R}^T$，其中$\mathbf{L} \in \mathbb{R}^{m \times r}$，$\mathbf{R} \in \mathbb{R}^{n \times r}$
3. **商流形表示**：考虑等价类$[(\mathbf{L}, \mathbf{R})] = \{(\mathbf{L}\mathbf{Q}, \mathbf{R}\mathbf{Q}): \mathbf{Q} \in GL(r)\}$

### 10.4.2 Riemannian梯度计算

对于SVD参数化，Euclidean梯度的稀疏结构为：
$$\nabla f(\mathbf{X}) = \mathcal{P}_\Omega(\mathbf{X} - \mathbf{M})$$
其中$\mathcal{P}_\Omega$是到观测位置的投影。

Riemannian梯度在切空间中的表示：
$$\text{grad} f(\mathbf{X}) = \mathbf{U}\mathbf{U}^T\nabla f(\mathbf{X})\mathbf{V}\mathbf{V}^T + \mathbf{U}_\perp\mathbf{U}_\perp^T\nabla f(\mathbf{X})\mathbf{V}\mathbf{V}^T + \mathbf{U}\mathbf{U}^T\nabla f(\mathbf{X})\mathbf{V}_\perp\mathbf{V}_\perp^T$$

关键优化：利用稀疏性，只需计算$\nabla f(\mathbf{X})$与相关子空间的作用，复杂度$O(|\Omega|r)$而非$O(mnr)$。

### 10.4.3 算法实现

**Riemannian梯度下降**：
1. 计算sparse梯度：$\mathbf{G} = \mathcal{P}_\Omega(\mathbf{X}_k - \mathbf{M})$
2. 投影到切空间：$\boldsymbol{\xi}_k = \text{Proj}_{T_{\mathbf{X}_k}}(\mathbf{G})$
3. 线搜索确定步长$\alpha_k$
4. 回缩更新：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(-\alpha_k\boldsymbol{\xi}_k)$

**Riemannian共轭梯度**：
1. 初始化：$\boldsymbol{\eta}_0 = -\text{grad} f(\mathbf{X}_0)$
2. 迭代：
   - 线搜索：$\alpha_k = \arg\min_\alpha f(\mathcal{R}_{\mathbf{X}_k}(\alpha\boldsymbol{\eta}_k))$
   - 更新：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(\alpha_k\boldsymbol{\eta}_k)$
   - 传输：$\boldsymbol{\xi}_{k+1} = -\text{grad} f(\mathbf{X}_{k+1})$
   - 共轭方向：$\boldsymbol{\eta}_{k+1} = \boldsymbol{\xi}_{k+1} + \beta_k\mathcal{T}_{\alpha_k\boldsymbol{\eta}_k}(\boldsymbol{\eta}_k)$

其中$\beta_k$可选择Fletcher-Reeves或Polak-Ribière公式。

### 10.4.4 收敛性与复杂度分析

**采样复杂度**：在适当的非相干性假设下，当观测数$|\Omega| \geq Cr(m+n)\log(m+n)$时，with high probability可以精确恢复秩$r$矩阵。

**迭代复杂度**：
- 每次迭代：$O(|\Omega|r + (m+n)r^2)$
- 内存需求：$O((m+n)r + |\Omega|)$
- 收敛率：线性收敛到局部最优（在适当假设下）

**实践加速技巧**：
1. **自适应秩增加**：从小秩开始，逐步增加以避免局部最优
2. **预条件**：利用Hessian的块结构设计高效预条件子
3. **并行化**：梯度计算和回缩步骤都易于并行
4. **warm start**：利用相近问题的解作为初始化

### 10.4.5 扩展与变体

**鲁棒矩阵补全**：处理离群值
$$\min_{\mathbf{X} \in \mathcal{M}_r, \mathbf{S}} \sum_{(i,j) \in \Omega} \rho(X_{ij} + S_{ij} - M_{ij}) + \lambda\|S\|_1$$
其中$\rho$是鲁棒损失函数。流形约束自然地正则化了问题。

**时序矩阵补全**：利用时间平滑性
$$\min_{\{\mathbf{X}_t\} \in \mathcal{M}_r} \sum_t \left[\sum_{(i,j) \in \Omega_t} (X_{t,ij} - M_{t,ij})^2 + \mu\|\mathbf{X}_t - \mathbf{X}_{t-1}\|_F^2\right]$$

流形结构使得时间平滑项的计算更加高效。

**多任务学习**：共享低秩结构
$$\min_{\mathbf{U} \in \mathcal{V}_r, \{\mathbf{V}_k\}} \sum_{k=1}^K \sum_{(i,j) \in \Omega_k} ([\mathbf{U}\mathbf{V}_k^T]_{ij} - M_{k,ij})^2$$

Stiefel流形约束确保了共享子空间的正交性。

## 本章小结

本章介绍了Riemannian优化的核心概念和实践技术：

**关键概念**：
- 矩阵流形提供了编码约束的自然几何结构
- Riemannian梯度通过投影Euclidean梯度到切空间获得：$\text{grad} f = \text{Proj}_{T_\mathbf{X}\mathcal{M}}(\nabla f)$
- 回缩映射提供了计算高效的流形更新方式：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(\alpha_k\boldsymbol{\xi}_k)$
- 向量传输实现了切向量在不同切空间之间的转移

**主要流形及其应用**：
- Stiefel流形$\mathcal{V}_k(\mathbb{R}^n)$：正交约束优化、特征值问题
- Grassmann流形$\mathcal{G}_k(\mathbb{R}^n)$：子空间学习、降维
- 固定秩流形$\mathcal{M}_r^{m \times n}$：低秩矩阵补全、模型压缩
- 正定矩阵锥$\mathcal{S}_{++}^n$：协方差估计、度量学习

**算法设计原则**：
1. 利用问题的几何结构选择合适的流形和度量
2. 在计算效率和理论性质之间权衡选择回缩和传输
3. 考虑流形曲率对收敛性的影响
4. 充分利用问题的稀疏性和特殊结构

**未来发展方向**：
- 大规模分布式流形优化
- 随机和在线流形算法
- 深度学习中的流形结构利用
- 与其他约束优化方法的融合

## 练习题

### 基础题

**习题10.1**：证明Stiefel流形的切空间表征
$$T_{\mathbf{X}}\mathcal{V}_k(\mathbb{R}^n) = \{\boldsymbol{\xi} \in \mathbb{R}^{n \times k} : \mathbf{X}^T\boldsymbol{\xi} + \boldsymbol{\xi}^T\mathbf{X} = \mathbf{0}\}$$

*提示*：利用约束$\mathbf{X}^T\mathbf{X} = \mathbf{I}$对时间求导。

<details>
<summary>答案</summary>

设$\mathbf{X}(t)$是流形上过$\mathbf{X}(0) = \mathbf{X}$的光滑曲线，满足$\mathbf{X}(t)^T\mathbf{X}(t) = \mathbf{I}$。对时间求导：
$$\frac{d}{dt}[\mathbf{X}(t)^T\mathbf{X}(t)]|_{t=0} = \dot{\mathbf{X}}(0)^T\mathbf{X} + \mathbf{X}^T\dot{\mathbf{X}}(0) = 0$$

由于切向量$\boldsymbol{\xi} = \dot{\mathbf{X}}(0)$，得到切空间条件。反之，任何满足此条件的$\boldsymbol{\xi}$都可以通过测地线或回缩构造相应的曲线。
</details>

**习题10.2**：计算函数$f(\mathbf{X}) = \text{tr}(\mathbf{A}^T\mathbf{X})$在Stiefel流形上的Riemannian梯度。

*提示*：先计算Euclidean梯度，然后投影到切空间。

<details>
<summary>答案</summary>

Euclidean梯度：$\nabla f(\mathbf{X}) = \mathbf{A}$

Riemannian梯度：
$$\text{grad} f(\mathbf{X}) = \text{Proj}_{T_{\mathbf{X}}\mathcal{V}}(\mathbf{A}) = \mathbf{A} - \mathbf{X}\text{sym}(\mathbf{X}^T\mathbf{A})$$

验证：$\mathbf{X}^T\text{grad} f + (\text{grad} f)^T\mathbf{X} = \mathbf{X}^T\mathbf{A} - \mathbf{X}^T\mathbf{A} - \mathbf{A}^T\mathbf{X} + \mathbf{A}^T\mathbf{X} = 0$。
</details>

**习题10.3**：证明QR分解定义的映射$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \text{qf}(\mathbf{X} + \boldsymbol{\xi})$是Stiefel流形上的回缩。

*提示*：验证回缩的两个定义条件。

<details>
<summary>答案</summary>

1. $\mathcal{R}_{\mathbf{X}}(\mathbf{0}) = \text{qf}(\mathbf{X}) = \mathbf{X}$（因为$\mathbf{X}$已经列正交）

2. 设$\mathbf{X} + t\boldsymbol{\xi} = \mathbf{Q}(t)\mathbf{R}(t)$，则：
   $$\frac{d}{dt}\mathcal{R}_{\mathbf{X}}(t\boldsymbol{\xi})|_{t=0} = \frac{d\mathbf{Q}(t)}{dt}|_{t=0}$$
   
   由QR分解的唯一性和连续性，当$t \to 0$时，$\mathbf{R}(t) \to \mathbf{I}$，因此$\frac{d\mathbf{Q}}{dt}|_{t=0} = \boldsymbol{\xi}$。
</details>

### 挑战题

**习题10.4**：设计固定秩流形$\mathcal{M}_r^{m \times n}$上的高效回缩映射，要求避免完整SVD计算。

*提示*：考虑利用切空间的特殊结构和增量SVD技术。

<details>
<summary>答案</summary>

对于$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$和切向量$\boldsymbol{\xi} = \mathbf{U}\mathbf{M}\mathbf{V}^T + \mathbf{U}_\perp\mathbf{N}\mathbf{V}^T + \mathbf{U}\mathbf{P}^T\mathbf{V}_\perp^T$：

1. 计算$\mathbf{Y} = \mathbf{X} + \boldsymbol{\xi}$的紧凑表示：
   $$\mathbf{Y} = [\mathbf{U}, \mathbf{U}_\perp]\begin{bmatrix}\boldsymbol{\Sigma} + \mathbf{M} & \mathbf{P}^T \\ \mathbf{N} & \mathbf{0}\end{bmatrix}[\mathbf{V}, \mathbf{V}_\perp]^T$$

2. 对中间$(2r) \times (2r)$矩阵进行SVD：
   $$\begin{bmatrix}\boldsymbol{\Sigma} + \mathbf{M} & \mathbf{P}^T \\ \mathbf{N} & \mathbf{0}\end{bmatrix} = \mathbf{U}_s\boldsymbol{\Sigma}_s\mathbf{V}_s^T$$

3. 回缩结果：$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = ([\mathbf{U}, \mathbf{U}_\perp]\mathbf{U}_s)_{:,1:r}\boldsymbol{\Sigma}_{s,1:r}([\mathbf{V}, \mathbf{V}_\perp]\mathbf{V}_s)_{:,1:r}^T$

复杂度：$O((m+n)r^2)$，避免了$O(mn\min(m,n))$的完整SVD。
</details>

**习题10.5**：分析Grassmann流形上的测地距离与主角度的关系，并设计基于此的线搜索策略。

*提示*：Grassmann流形上两个子空间之间的测地距离由主角度决定。

<details>
<summary>答案</summary>

设$\mathcal{X}, \mathcal{Y}$是两个$k$维子空间，主角度$0 \leq \theta_1 \leq \cdots \leq \theta_k \leq \pi/2$定义为：
$$\cos\theta_i = \max_{\mathbf{x} \in \mathcal{X}, \mathbf{y} \in \mathcal{Y}} \mathbf{x}^T\mathbf{y}$$
subject to正交约束。

测地距离：$d_g(\mathcal{X}, \mathcal{Y}) = \|\boldsymbol{\theta}\|_2 = \sqrt{\sum_{i=1}^k \theta_i^2}$

线搜索策略：
1. 对于搜索方向$\boldsymbol{\eta}$，参数化测地线$\gamma(t) = \text{Exp}_{\mathcal{X}}(t\boldsymbol{\eta})$
2. 利用主角度的单调性质，采用黄金分割搜索
3. 终止准则基于角度变化：$\|\boldsymbol{\theta}(t) - \boldsymbol{\theta}(t-\Delta t)\| < \epsilon$

此策略特别适合子空间追踪和正交Procrustes问题。
</details>

**习题10.6**：推导固定秩流形上的Riemannian Hessian-向量积的高效计算方法，用于信赖域子问题求解。

*提示*：利用二阶导数的链式法则和投影算子的导数。

<details>
<summary>答案</summary>

对于$f(\mathbf{X})$和切向量$\boldsymbol{\xi}, \boldsymbol{\eta} \in T_{\mathbf{X}}\mathcal{M}_r$：

1. Euclidean Hessian作用：$\mathbf{H}[\boldsymbol{\eta}] = \text{D}^2f(\mathbf{X})[\boldsymbol{\eta}]$

2. 曲率校正项：利用Weingarten映射
   $$\mathcal{W}_{\boldsymbol{\xi}}(\boldsymbol{\eta}) = -\text{Proj}_{T_{\mathbf{X}}\mathcal{M}}(\text{D}(\text{Proj}_{N_{\mathbf{X}}\mathcal{M}})(\mathbf{X})[\boldsymbol{\eta}][\boldsymbol{\xi}])$$

3. Riemannian Hessian：
   $$\text{Hess}f(\mathbf{X})[\boldsymbol{\eta}] = \text{Proj}_{T_{\mathbf{X}}\mathcal{M}}(\mathbf{H}[\boldsymbol{\eta}]) + \mathcal{W}_{\text{grad}f(\mathbf{X})}(\boldsymbol{\eta})$$

高效实现：
- 利用自动微分计算Hessian-向量积
- 缓存投影算子的中间结果
- 对于二次函数，曲率项可以预计算

复杂度：$O(|\Omega|r + (m+n)r^2)$用于矩阵补全问题。
</details>

**习题10.7**：设计流形约束下的随机梯度算法，分析其收敛性质。

*提示*：考虑随机梯度的无偏性和回缩的影响。

<details>
<summary>答案</summary>

Riemannian SGD算法：
1. 采样mini-batch $\mathcal{B}_k$
2. 计算随机梯度：$\tilde{\mathbf{g}}_k = \frac{1}{|\mathcal{B}_k|}\sum_{i \in \mathcal{B}_k}\nabla f_i(\mathbf{X}_k)$
3. 投影：$\tilde{\boldsymbol{\xi}}_k = \text{Proj}_{T_{\mathbf{X}_k}\mathcal{M}}(\tilde{\mathbf{g}}_k)$
4. 更新：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(-\alpha_k\tilde{\boldsymbol{\xi}}_k)$

收敛性分析：
- 无偏性：$\mathbb{E}[\tilde{\boldsymbol{\xi}}_k | \mathbf{X}_k] = \text{grad}f(\mathbf{X}_k)$
- 方差界：需要假设$\mathbb{E}[\|\tilde{\mathbf{g}}_k - \nabla f(\mathbf{X}_k)\|^2] \leq \sigma^2$
- 收敛率：在强凸条件下（相对于流形度量），$\mathbb{E}[f(\mathbf{X}_k) - f^*] = O(1/k)$

关键挑战：
1. 流形曲率影响方差界
2. 回缩的非线性可能破坏某些性质
3. 自适应步长需要考虑度量变化

实践建议：使用方差缩减技术（SVRG、SAGA的流形版本）。
</details>

## 常见陷阱与错误

1. **度量选择不当**
   - 错误：盲目使用Euclidean度量
   - 正确：根据问题的不变性选择合适度量
   - 例如：正定矩阵使用仿射不变度量

2. **忽视数值稳定性**
   - 错误：直接计算$(\mathbf{I} + \boldsymbol{\xi}^T\boldsymbol{\xi})^{-1/2}$
   - 正确：使用Cholesky分解或迭代方法
   - 特别注意：接近奇异时的处理

3. **切空间投影错误**
   - 错误：忘记对称化或使用错误的投影公式
   - 正确：仔细验证投影后确实在切空间内
   - 调试技巧：检查$\|\text{Proj}(\boldsymbol{\xi}) - \boldsymbol{\xi}\|$

4. **回缩选择不当**
   - 错误：总是使用最简单的回缩
   - 正确：权衡计算成本和收敛性能
   - 原则：二阶方法需要更精确的回缩

5. **向量传输缺失**
   - 错误：在CG或BFGS中忽略向量传输
   - 正确：即使使用简单投影也要传输
   - 影响：可能导致收敛变慢或发散

6. **初始化问题**
   - 错误：随机初始化可能不在流形上
   - 正确：使用投影或特定构造确保可行性
   - 技巧：利用问题结构选择好的初始点

7. **步长选择**
   - 错误：使用固定步长
   - 正确：考虑流形曲率的自适应步长
   - 注意：Armijo规则需要适应流形结构

8. **秩选择**
   - 错误：固定秩可能过小或过大
   - 正确：自适应秩选择或交叉验证
   - 策略：从小秩开始逐步增加

## 最佳实践检查清单

### 算法设计阶段
- [ ] 识别问题的自然几何结构
- [ ] 选择合适的流形和参数化
- [ ] 确定计算效率vs理论性质的权衡
- [ ] 设计利用问题稀疏性的策略

### 实现阶段
- [ ] 验证所有操作保持流形约束
- [ ] 实现数值稳定的投影和回缩
- [ ] 添加调试断言检查切空间条件
- [ ] 优化内存使用和计算顺序

### 调优阶段
- [ ] 比较不同回缩的性能
- [ ] 测试各种线搜索策略
- [ ] 调整算法参数（信赖域半径等）
- [ ] 分析不同初始化的影响

### 验证阶段
- [ ] 检查收敛到的解确实在流形上
- [ ] 验证一阶必要条件
- [ ] 与其他方法比较解的质量
- [ ] 评估计算时间和可扩展性

## 研究方向

1. **大规模分布式流形优化**
   - 如何设计通信高效的分布式算法
   - 异步更新在流形上的收敛性保证
   - 联邦学习中的流形约束处理

2. **深度学习中的流形结构**
   - 神经网络权重的内在流形结构
   - 归一化层的流形interpretation
   - 流形正则化的新方法

3. **混合整数流形优化**
   - 离散选择与连续流形约束的结合
   - 组合流形的算法设计
   - 在特征选择中的应用

4. **动态流形优化**
   - 时变流形上的追踪算法
   - 在线学习的流形适应
   - 非平稳环境下的理论保证

5. **高阶流形方法**
   - 三阶及以上信息的利用
   - 流形上的内点法
   - 与传统高阶方法的比较

6. **流形上的随机优化**
   - Heavy-tail噪声下的鲁棒算法
   - 流形MCMC采样
   - 贝叶斯优化的流形扩展

7. **量子流形算法**
   - 量子计算加速的可能性
   - 流形结构的量子表示
   - 混合经典-量子算法