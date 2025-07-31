# 第10章：Riemannian优化基础

许多大规模矩阵优化问题天然地定义在具有特定几何结构的约束集上。传统的欧氏空间优化方法需要反复投影以维持约束，而Riemannian优化直接在约束流形上工作，将约束隐式地编码在流形的几何结构中。本章介绍流形优化的核心概念，重点关注在机器学习和数值计算中常见的矩阵流形。我们将探讨如何利用流形的内在几何结构设计高效的优化算法，以及这些方法在低秩矩阵补全等实际问题中的应用。

## 10.1 矩阵流形上的几何结构

矩阵流形是满足特定代数约束的矩阵集合，配备了使其成为光滑流形的微分结构。理解这些流形的几何性质是设计高效优化算法的基础。流形优化的核心优势在于将约束隐式编码在几何结构中，从而避免了传统方法中反复投影的计算开销。

### 10.1.1 常见的矩阵流形

**Stiefel流形** $\mathcal{V}_k(\mathbb{R}^n)$：所有$n \times k$列正交矩阵的集合
$$\mathcal{V}_k(\mathbb{R}^n) = \{\mathbf{X} \in \mathbb{R}^{n \times k} : \mathbf{X}^T\mathbf{X} = \mathbf{I}_k\}$$

当$k = n$时，Stiefel流形退化为正交群$O(n)$。当$k = 1$时，得到单位球面$S^{n-1}$。Stiefel流形在主成分分析、正交Procrustes问题、特征值计算和深度学习的正交初始化中广泛应用。其维数为$nk - k(k+1)/2$，反映了正交约束的数量。

**研究线索**：Stiefel流形上的概率分布（如von Mises-Fisher分布）在方向统计和贝叶斯推断中有重要应用，但高维情况下的采样算法仍需改进。

**Grassmann流形** $\mathcal{G}_k(\mathbb{R}^n)$：$\mathbb{R}^n$中所有$k$维子空间的集合。可以视为Stiefel流形在正交等价关系下的商空间：
$$\mathcal{G}_k(\mathbb{R}^n) = \mathcal{V}_k(\mathbb{R}^n) / O(k)$$

Grassmann流形的维数为$k(n-k)$，比Stiefel流形低，这种维数缩减来源于消除了子空间基的选择自由度。在子空间追踪、降维、计算机视觉（如多视角几何）和量子信息理论中扮演重要角色。

**固定秩矩阵流形** $\mathcal{M}_r^{m \times n}$：所有秩为$r$的$m \times n$矩阵的集合
$$\mathcal{M}_r^{m \times n} = \{\mathbf{X} \in \mathbb{R}^{m \times n} : \text{rank}(\mathbf{X}) = r\}$$

这是一个$(m + n - r)r$维的嵌入子流形，在低秩矩阵补全、模型压缩和系统识别中至关重要。固定秩流形不是闭集（其闭包包含更低秩的矩阵），这给数值计算带来独特挑战。

**对称正定矩阵锥** $\mathcal{S}_{++}^n$：所有$n \times n$对称正定矩阵的集合。虽然技术上是开集而非流形，但可以赋予其多种Riemannian几何结构：
- 标准欧氏度量：继承自矩阵空间
- 仿射不变度量：$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\mathbf{X}^{-1}\boldsymbol{\xi}\mathbf{X}^{-1}\boldsymbol{\eta})$
- Log-Euclidean度量：通过矩阵对数映射到欧氏空间

在协方差矩阵估计、度量学习、脑电图信号处理和金融风险建模中应用广泛。

**其他重要流形**：
- **对称矩阵流形** $\mathcal{S}^n$：图拉普拉斯优化、谱聚类
- **斜对称矩阵流形** $\mathfrak{so}(n)$：刚体动力学、李群算法
- **单位对角占优矩阵流形**：马尔可夫链、随机矩阵理论
- **双随机矩阵流形**：最优传输、图匹配

**研究线索**：混合约束流形（如同时满足正交性和稀疏性约束）的几何结构和算法设计是一个活跃的研究方向。

### 10.1.2 切空间的刻画

切空间$T_{\mathbf{X}}\mathcal{M}$是流形$\mathcal{M}$在点$\mathbf{X}$处的线性化，包含了所有可行的"无穷小移动方向"。理解切空间的结构对于梯度计算、约束保持和算法设计至关重要。切空间可以通过多种等价方式刻画：速度向量、约束的核空间或局部参数化的导数像。

**Stiefel流形的切空间**：
$$T_{\mathbf{X}}\mathcal{V}_k(\mathbb{R}^n) = \{\boldsymbol{\xi} \in \mathbb{R}^{n \times k} : \mathbf{X}^T\boldsymbol{\xi} + \boldsymbol{\xi}^T\mathbf{X} = \mathbf{0}\}$$

这个条件确保$\mathbf{X} + t\boldsymbol{\xi}$在一阶近似下保持列正交性。等价地，可以写成：
$$T_{\mathbf{X}}\mathcal{V}_k(\mathbb{R}^n) = \{\mathbf{X}\mathbf{A} + \mathbf{X}_\perp\mathbf{B} : \mathbf{A} \in \mathbb{R}^{k \times k}, \mathbf{A}^T = -\mathbf{A}, \mathbf{B} \in \mathbb{R}^{(n-k) \times k}\}$$

其中$\mathbf{X}_\perp \in \mathbb{R}^{n \times (n-k)}$是$\mathbf{X}$的正交补。这种表示揭示了切空间的维数：$k(k-1)/2 + k(n-k) = nk - k(k+1)/2$。

**Grassmann流形的切空间**：作为商流形，其切空间继承自Stiefel流形但需要模去垂直空间：
$$T_{[\mathbf{X}]}\mathcal{G}_k(\mathbb{R}^n) = \{\mathbf{X}_\perp\mathbf{B} : \mathbf{B} \in \mathbb{R}^{(n-k) \times k}\}$$

这里$[\mathbf{X}]$表示包含$\mathbf{X}$列空间的等价类。注意切向量必须与当前子空间正交，维数为$k(n-k)$。

**固定秩流形的切空间**：对于$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$的SVD分解，
$$T_{\mathbf{X}}\mathcal{M}_r^{m \times n} = \{\mathbf{U}\mathbf{M}\mathbf{V}^T + \mathbf{U}_\perp\mathbf{N}\mathbf{V}^T + \mathbf{U}\mathbf{P}^T\mathbf{V}_\perp^T : \mathbf{M} \in \mathbb{R}^{r \times r}, \mathbf{N} \in \mathbb{R}^{(m-r) \times r}, \mathbf{P} \in \mathbb{R}^{(n-r) \times r}\}$$

其中$\mathbf{U}_\perp$和$\mathbf{V}_\perp$分别是$\mathbf{U}$和$\mathbf{V}$的正交补。这种分解反映了三个独立的变化方向：
- $\mathbf{U}\mathbf{M}\mathbf{V}^T$：保持行列空间不变，只改变奇异值和混合
- $\mathbf{U}_\perp\mathbf{N}\mathbf{V}^T$：扩展行空间
- $\mathbf{U}\mathbf{P}^T\mathbf{V}_\perp^T$：扩展列空间

**对称正定矩阵的切空间**：
$$T_{\mathbf{X}}\mathcal{S}_{++}^n = \mathcal{S}^n = \{\boldsymbol{\xi} \in \mathbb{R}^{n \times n} : \boldsymbol{\xi}^T = \boldsymbol{\xi}\}$$

由于$\mathcal{S}_{++}^n$是开集，其切空间就是所有对称矩阵的空间。

**切空间的数值表示**：实践中，存储和操作切向量需要高效的数据结构：
- **压缩表示**：对Stiefel流形，只存储$\mathbf{A}$和$\mathbf{B}$而非完整的$\boldsymbol{\xi}$
- **正交基**：预计算切空间的正交基，便于投影和内积计算
- **隐式表示**：通过线性算子避免显式构造大矩阵

**研究线索**：切空间的几何性质（如截面曲率）如何影响优化算法的收敛速度是一个重要但尚未充分研究的问题。特别是在高维情况下，局部几何的统计性质可能提供新的算法设计思路。

### 10.1.3 Riemannian度量

Riemannian度量为每个切空间赋予内积结构，使得我们能够测量切向量的长度和夹角。选择合适的度量对算法性能影响重大，不同的度量导致不同的梯度、Hessian和测地线，从而影响收敛速度和数值稳定性。

**标准度量**：最常用的是从环境空间继承的Euclidean度量
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle \boldsymbol{\xi}, \boldsymbol{\eta} \rangle_F = \text{tr}(\boldsymbol{\xi}^T\boldsymbol{\eta})$$

优点：计算简单，与欧氏空间算法兼容性好。缺点：可能不反映问题的内在几何，对于病态问题收敛慢。

**规范不变度量**：对于Grassmann流形，可以定义对子空间基的选择不变的度量
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\boldsymbol{\xi}^T(\mathbf{I} - \mathbf{X}\mathbf{X}^T)\boldsymbol{\eta})$$

这确保了度量只依赖于子空间本身而非其特定表示。在主角度意义下，这个度量诱导的测地距离有明确的几何解释。

**信息几何度量**：对于正定矩阵锥，Fisher信息度量（也称仿射不变度量）
$$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\mathbf{X}^{-1}\boldsymbol{\xi}\mathbf{X}^{-1}\boldsymbol{\eta})$$

提供了对矩阵求逆和缩放的自然不变性。这个度量下的测地线是矩阵几何平均，在协方差矩阵插值和扩散张量成像中有重要应用。

**其他重要度量**：
- **Bures-Wasserstein度量**：$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \frac{1}{2}\text{tr}(\boldsymbol{\xi}\mathbf{X}^{-1}\boldsymbol{\eta} + \boldsymbol{\eta}\mathbf{X}^{-1}\boldsymbol{\xi})$，在量子信息和最优传输中使用
- **Log-Euclidean度量**：通过对数映射转换到欧氏空间，$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle d\log_{\mathbf{X}}(\boldsymbol{\xi}), d\log_{\mathbf{X}}(\boldsymbol{\eta}) \rangle$
- **加权度量**：$g_{\mathbf{X}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \text{tr}(\boldsymbol{\xi}^T\mathbf{W}\boldsymbol{\eta})$，其中$\mathbf{W}$编码问题特定的重要性

**度量选择的影响**：
1. **条件数**：适当的度量可以改善优化问题的条件数。例如，对于正定矩阵的优化，仿射不变度量often提供更好的条件数
2. **收敛速度**：Natural gradient（使用Fisher信息度量）在许多统计问题中展现二阶收敛性质
3. **数值稳定性**：某些度量（如Log-Euclidean）通过避免矩阵求逆提高数值稳定性
4. **计算复杂度**：标准度量计算最快，而信息几何度量需要额外的矩阵运算

**研究线索**：自适应度量选择——根据问题的局部几何动态调整度量——是一个有前景的研究方向。这可能结合了不同度量的优点，但理论分析仍然困难。

### 10.1.4 测地线与指数映射

测地线是流形上的"直线"，即局部最短路径。指数映射$\exp_{\mathbf{X}}: T_{\mathbf{X}}\mathcal{M} \to \mathcal{M}$将切向量映射到沿测地线的点。理解测地线的性质对于设计高效的优化算法和分析收敛性至关重要。

**测地线的刻画**：测地线$\gamma(t)$满足测地线方程
$$\nabla_{\dot{\gamma}}\dot{\gamma} = 0$$
其中$\nabla$是Levi-Civita联络。这意味着测地线的"加速度"（在流形意义下）为零。

对于配备标准度量的Stiefel流形，指数映射有闭式表达：
$$\exp_{\mathbf{X}}(\boldsymbol{\xi}) = [\mathbf{X}, \mathbf{Q}]\exp\left(\begin{bmatrix} \mathbf{A} & -\mathbf{R}^T \\ \mathbf{R} & \mathbf{0} \end{bmatrix}\right)\begin{bmatrix} \mathbf{I}_k \\ \mathbf{0} \end{bmatrix}$$
其中$\mathbf{Q}\mathbf{R} = (\mathbf{I} - \mathbf{X}\mathbf{X}^T)\boldsymbol{\xi}$是瘦QR分解，$\mathbf{A} = \mathbf{X}^T\boldsymbol{\xi}$是斜对称矩阵。

**其他流形的指数映射**：
- **正定矩阵（仿射不变度量）**：$\exp_{\mathbf{X}}(\boldsymbol{\xi}) = \mathbf{X}^{1/2}\exp(\mathbf{X}^{-1/2}\boldsymbol{\xi}\mathbf{X}^{-1/2})\mathbf{X}^{1/2}$
- **Grassmann流形**：通过Stiefel流形的指数映射和投影获得
- **固定秩流形**：一般无闭式表达，需要数值求解常微分方程

**计算复杂度分析**：
- Stiefel流形：$O(nk^2 + k^3)$，主要开销在矩阵指数
- 正定矩阵：$O(n^3)$，需要特征分解
- 固定秩流形：$O(r^2(m+n))$每步，需要多步数值积分

然而，指数映射的计算通常开销较大，实践中常用更简单的回缩映射替代。关键观察是：对于一阶方法，只需要保证一阶近似的正确性；对于二阶方法，回缩的二阶近似通常就足够了。

**对数映射与内射半径**：对数映射$\log_{\mathbf{X}}: \mathcal{M} \to T_{\mathbf{X}}\mathcal{M}$是指数映射的（局部）逆。内射半径$\text{inj}(\mathbf{X})$是使得指数映射为微分同胚的最大半径。了解内射半径对于：
- 确定信赖域算法的最大步长
- 分析全局收敛性
- 设计测地线搜索算法

**研究线索**：对于许多实际应用的流形，精确计算或有效近似内射半径仍是开放问题。这限制了某些理论保证的实用性。

## 10.2 Riemannian梯度与Hessian

将欧氏空间的梯度和Hessian概念推广到流形上，是构建流形优化算法的关键步骤。这些概念不仅提供了优化的基本工具，还揭示了流形几何与优化性能之间的深刻联系。

### 10.2.1 Riemannian梯度的计算

给定定义在流形$\mathcal{M}$上的光滑函数$f: \mathcal{M} \to \mathbb{R}$，其在点$\mathbf{X}$的Riemannian梯度$\text{grad} f(\mathbf{X})$是唯一满足以下条件的切向量：
$$g_{\mathbf{X}}(\text{grad} f(\mathbf{X}), \boldsymbol{\xi}) = \text{D}f(\mathbf{X})[\boldsymbol{\xi}], \quad \forall \boldsymbol{\xi} \in T_{\mathbf{X}}\mathcal{M}$$

其中$\text{D}f(\mathbf{X})[\boldsymbol{\xi}]$是$f$在$\mathbf{X}$沿方向$\boldsymbol{\xi}$的方向导数。

**投影方法**：若$f$可以扩展到环境空间，则Riemannian梯度是Euclidean梯度在切空间上的正交投影（相对于所选度量）：
$$\text{grad} f(\mathbf{X}) = \text{Proj}_{T_{\mathbf{X}}\mathcal{M}}^g(\nabla f(\mathbf{X}))$$

对于标准度量，这简化为正交投影。对于Stiefel流形，投影算子为：
$$\text{Proj}_{T_{\mathbf{X}}\mathcal{V}}(\mathbf{Z}) = \mathbf{Z} - \mathbf{X}\text{sym}(\mathbf{X}^T\mathbf{Z})$$
其中$\text{sym}(\mathbf{A}) = (\mathbf{A} + \mathbf{A}^T)/2$。

**具体流形的梯度公式**：

*Grassmann流形*：对于$f([\mathbf{X}])$，
$$\text{grad} f([\mathbf{X}]) = (\mathbf{I} - \mathbf{X}\mathbf{X}^T)\nabla \tilde{f}(\mathbf{X})$$
其中$\tilde{f}$是$f$在Stiefel流形上的提升。

*固定秩流形*：设$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，
$$\text{grad} f(\mathbf{X}) = \mathbf{U}\mathbf{U}^T\nabla f(\mathbf{X})\mathbf{V}\mathbf{V}^T + \mathbf{U}_\perp\mathbf{U}_\perp^T\nabla f(\mathbf{X})\mathbf{V}\mathbf{V}^T + \mathbf{U}\mathbf{U}^T\nabla f(\mathbf{X})\mathbf{V}_\perp\mathbf{V}_\perp^T$$

*正定矩阵（仿射不变度量）*：
$$\text{grad} f(\mathbf{X}) = \mathbf{X}\text{sym}(\nabla f(\mathbf{X}))\mathbf{X}$$

**计算技巧与优化**：
1. **利用稀疏性**：当$\nabla f(\mathbf{X})$稀疏时，投影可以高效计算
2. **避免显式正交补**：使用$\mathbf{I} - \mathbf{U}\mathbf{U}^T$而非存储$\mathbf{U}_\perp$
3. **缓存重用**：在迭代算法中重用QR分解或SVD
4. **自动微分**：现代框架（PyTorch, JAX）可以自动计算流形梯度

**梯度的几何意义**：Riemannian梯度指向函数在流形上增长最快的方向，其模长反映增长率。与欧氏情况不同，这个方向依赖于所选的度量，体现了"最陡"的相对性。

**研究线索**：对于非光滑函数的流形次梯度理论仍在发展中，这对于稀疏优化和鲁棒统计特别重要。

### 10.2.2 Riemannian Hessian

Riemannian Hessian描述梯度场在流形上的变化率。对于切向量$\boldsymbol{\xi} \in T_{\mathbf{X}}\mathcal{M}$，Hessian作用定义为：
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \nabla_{\boldsymbol{\xi}} \text{grad} f(\mathbf{X})$$

其中$\nabla$是Levi-Civita联络，它唯一地由度量决定并保持度量相容性和无挠性。

**计算的三步框架**：

1. **梯度场的扩展**：将$\text{grad} f$从单点扩展到邻域。这可以通过参数化或嵌入实现
2. **方向导数**：计算扩展梯度场沿$\boldsymbol{\xi}$的导数
3. **投影**：将结果投影回切空间，因为协变导数必须是切向量

对于嵌入子流形，Weingarten公式提供了实用的计算方法：
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \text{Proj}_{T_{\mathbf{X}}\mathcal{M}}(\text{D}(\nabla f)(\mathbf{X})[\boldsymbol{\xi}]) - \text{II}_{\mathbf{X}}(\boldsymbol{\xi}, \text{grad} f(\mathbf{X}))$$

其中$\text{II}_{\mathbf{X}}$是第二基本形式，刻画了流形的外在曲率。第二项反映了流形弯曲对Hessian的贡献。

**具体流形的Hessian计算**：

*Stiefel流形*：对于$\boldsymbol{\xi} \in T_{\mathbf{X}}\mathcal{V}$，
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \text{Proj}_{T_{\mathbf{X}}\mathcal{V}}(\text{D}^2f(\mathbf{X})[\boldsymbol{\xi}]) + \mathbf{X}\text{sym}(\boldsymbol{\xi}^T\nabla f(\mathbf{X}))$$

第二项来自第二基本形式，反映了正交约束的影响。

*正定矩阵（仿射不变度量）*：
$$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] = \mathbf{X}\text{sym}(\text{D}^2f(\mathbf{X})[\boldsymbol{\xi}] - \text{D}f(\mathbf{X})[\mathbf{X}^{-1}\boldsymbol{\xi}])\mathbf{X}$$

这个公式展示了度量选择如何影响Hessian的形式。

**Hessian-向量积的高效计算**：直接构造Hessian矩阵在高维情况下不可行。实践中使用：

1. **有限差分近似**：
   $$\text{Hess} f(\mathbf{X})[\boldsymbol{\xi}] \approx \frac{1}{\epsilon}[\text{grad} f(\mathcal{R}_{\mathbf{X}}(\epsilon\boldsymbol{\xi})) - \mathcal{T}_{\epsilon\boldsymbol{\xi}}^{-1}(\text{grad} f(\mathbf{X}))]$$

2. **自动微分**：利用反向模式自动微分计算精确的Hessian-向量积

3. **BFGS近似**：维护Hessian的低秩近似，适用于大规模问题

**Hessian的性质与应用**：
- **自伴性**：相对于所选度量，Hessian是自伴算子
- **临界点分类**：正定Hessian表示局部最小值
- **收敛速度**：Hessian的条件数影响Newton法的收敛速度
- **负曲率利用**：在非凸优化中，负特征值方向可用于逃离鞍点

**研究线索**：流形上的拟Newton方法如何最优地利用曲率信息仍有改进空间。特别是，如何设计保持流形结构的秩-1或秩-2更新公式。

### 10.2.3 二阶优化算法

二阶方法利用Hessian信息实现更快的局部收敛。在流形上实现这些方法需要仔细处理几何结构。

**Riemannian Newton法**：求解切空间中的Newton方程
$$\text{Hess} f(\mathbf{X}_k)[\boldsymbol{\eta}_k] = -\text{grad} f(\mathbf{X}_k)$$

然后沿回缩更新：$\mathbf{X}_{k+1} = \mathcal{R}_{\mathbf{X}_k}(\alpha_k \boldsymbol{\eta}_k)$

关键实现细节：
- **线性系统求解**：利用共轭梯度法求解Newton方程，避免显式构造Hessian
- **正则化**：添加$\lambda\mathbf{I}$确保正定性，$(\text{Hess} f + \lambda\mathbf{I})[\boldsymbol{\eta}] = -\text{grad} f$
- **步长选择**：Armijo回溯或精确线搜索，注意在流形上评估函数值

**Riemannian信赖域法**：在切空间中求解子问题
$$\min_{\boldsymbol{\eta} \in T_{\mathbf{X}}\mathcal{M}} m_{\mathbf{X}}(\boldsymbol{\eta}) = f(\mathbf{X}) + g_{\mathbf{X}}(\text{grad} f(\mathbf{X}), \boldsymbol{\eta}) + \frac{1}{2}g_{\mathbf{X}}(\text{Hess} f(\mathbf{X})[\boldsymbol{\eta}], \boldsymbol{\eta})$$
subject to $\|\boldsymbol{\eta}\|_{\mathbf{X}} \leq \Delta$

信赖域方法对非凸问题特别有效，能够利用负曲率方向。子问题求解策略：
- **Steihaug-Toint CG**：当遇到负曲率或达到信赖域边界时提前终止
- **Lanczos方法**：构造Krylov子空间中的近似解
- **精确求解**：对于小规模问题，可以通过特征分解精确求解

**Riemannian BFGS**：维护Hessian逆的近似$\mathbf{H}_k$，更新公式需要考虑向量传输：
$$\mathbf{H}_{k+1} = \mathcal{T}_{\alpha_k\boldsymbol{\eta}_k}(\mathbf{H}_k) + \text{秩-2修正}$$

其中秩-2修正确保割线条件：$\mathbf{H}_{k+1}\mathbf{y}_k = \mathbf{s}_k$，这里$\mathbf{s}_k$和$\mathbf{y}_k$是传输后的步长和梯度差。

**Limited-memory BFGS (L-BFGS)**：只存储最近$m$对向量$\{\mathbf{s}_i, \mathbf{y}_i\}$，通过两遍递归计算搜索方向。流形版本需要：
- 历史向量的传输
- 初始Hessian近似的选择（可以利用流形度量）
- 处理曲率信息的策略

**收敛性分析**：
- **局部收敛**：Newton法在非退化最小值附近达到二次收敛
- **全局收敛**：信赖域方法保证收敛到一阶驻点
- **复杂度**：每次迭代$O(n^3)$（Newton）或$O(mn)$（L-BFGS）

### 10.2.4 曲率的影响

流形的内在曲率深刻影响优化算法的行为。理解这种影响对于算法设计和分析至关重要。

**截面曲率的作用**：
- **正曲率**（如球面）：梯度流趋向于比欧氏空间更快地收敛，因为测地线会聚
- **负曲率**（如双曲空间）：可能导致不稳定，测地线发散
- **混合曲率**：大多数实际流形，需要局部分析

**Rayleigh商优化**：在单位球面$S^{n-1}$上最小化$f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$。球面的常正曲率$\kappa = 1$加速收敛：
- 梯度流的吸引盆更大
- 逃离鞍点更容易
- 收敛速度依赖于特征值间隙和曲率的相互作用

**曲率修正技术**：
1. **动量修正**：Riemannian加速梯度中，动量项需要根据曲率调整
   $$\boldsymbol{\mu}_{k+1} = \gamma\mathcal{T}_{\alpha_k\boldsymbol{\xi}_k}(\boldsymbol{\mu}_k) - \alpha_k\text{grad} f(\mathbf{X}_{k+1}) + \text{曲率修正项}$$

2. **预条件设计**：利用Ricci曲率信息设计更好的预条件子

3. **自适应算法**：根据局部曲率估计动态调整步长和其他参数

**曲率与优化landscape的关系**：
- **正曲率区域**：通常对应凸区域，优化相对容易
- **负曲率区域**：可能存在多个局部最优，需要全局策略
- **零曲率方向**：可能导致收敛变慢，需要特殊处理

**研究线索**：如何系统地利用流形的全局拓扑和几何信息来设计更智能的优化算法仍是开放问题。特别是，能否通过学习流形的几何来加速优化？

## 10.3 回缩与向量传输

虽然指数映射提供了理论上优雅的流形更新方式，但其计算成本往往过高。回缩（retraction）提供了计算友好的替代方案，而向量传输（vector transport）则是实现高效迭代算法的关键组件。这两个概念的巧妙设计往往决定了流形优化算法的实用性。

### 10.3.1 回缩映射

回缩$\mathcal{R}: T\mathcal{M} \to \mathcal{M}$是满足以下条件的光滑映射：
1. $\mathcal{R}_{\mathbf{X}}(\mathbf{0}) = \mathbf{X}$（保持原点）
2. $\text{D}\mathcal{R}_{\mathbf{X}}(\mathbf{0})[\boldsymbol{\xi}] = \boldsymbol{\xi}$（局部刚性）

回缩提供了从切空间"回到"流形的方式，且保证了一阶近似的正确性。对于二阶方法，可能需要更强的性质：
- **二阶回缩**：$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \exp_{\mathbf{X}}(\boldsymbol{\xi}) + O(\|\boldsymbol{\xi}\|^3)$
- **凸组合性质**：某些回缩保持测地凸组合的性质

**常用回缩映射**：

*QR回缩*（Stiefel流形）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \text{qf}(\mathbf{X} + \boldsymbol{\xi})$$
其中$\text{qf}(\cdot)$表示薄QR分解的Q因子。

优点：
- 数值稳定，基于Householder或Givens变换
- 计算高效：$O(nk^2)$
- 自然处理$k \ll n$的情况

实现细节：
- 使用列主导的修正Gram-Schmidt以提高稳定性
- 当$\mathbf{X} + \boldsymbol{\xi}$接近秩亏时需要特殊处理

*极回缩*（Stiefel流形）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = (\mathbf{X} + \boldsymbol{\xi})(\mathbf{I} + \boldsymbol{\xi}^T\boldsymbol{\xi})^{-1/2}$$

基于极分解，保持了更多几何性质：
- 最短路径性质：在某种意义下最小化$\|\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) - (\mathbf{X} + \boldsymbol{\xi})\|_F$
- 保持奇异值的单调性
- 计算：$O(nk^2 + k^3)$，矩阵平方根可用Denman-Beavers迭代

*投影回缩*（固定秩流形）：设$\mathbf{Y} = \mathbf{X} + \boldsymbol{\xi}$，计算其截断SVD
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \mathcal{P}_r(\mathbf{Y}) = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

关键优化：利用$\boldsymbol{\xi}$的结构避免完整SVD：
1. 若$\boldsymbol{\xi} = \mathbf{U}_X\mathbf{M}\mathbf{V}_X^T + \mathbf{U}_\perp\mathbf{N}\mathbf{V}_X^T + \mathbf{U}_X\mathbf{P}^T\mathbf{V}_\perp^T$
2. 则只需对$(2r) \times (2r)$矩阵进行SVD
3. 总复杂度：$O((m+n)r^2)$而非$O(mn\min(m,n))$

*测地回缩*（正定矩阵）：
$$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \mathbf{X}^{1/2}\exp(\mathbf{X}^{-1/2}\boldsymbol{\xi}\mathbf{X}^{-1/2})\mathbf{X}^{1/2}$$

这实际上就是指数映射，展示了某些流形上回缩可以高效计算。

**回缩的选择原则**：
1. **计算效率**：QR回缩通常最快
2. **数值稳定性**：避免矩阵求逆和病态运算
3. **几何性质**：某些应用需要保持特定的几何不变量
4. **并行性**：考虑GPU实现的可行性

**研究线索**：设计保持更多几何性质同时计算高效的回缩仍是活跃研究方向。特别是对于乘积流形和复合约束的情况。

### 10.3.2 向量传输

向量传输$\mathcal{T}_{\boldsymbol{\eta}}: T_{\mathbf{X}}\mathcal{M} \to T_{\mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})}\mathcal{M}$将切向量从一个切空间"运输"到另一个。这对于需要历史信息的算法（如共轭梯度、BFGS、动量方法）至关重要。向量传输的质量直接影响这些算法的收敛性。

**平行移动vs向量传输**：
- **平行移动**：保持向量在Levi-Civita联络意义下"平行"，保持内积和长度，但计算复杂
- **向量传输**：只需满足较弱的相容性条件，允许更高效的实现
- **关键性质**：对于收敛性，通常只需要传输是等距的加上一些连续性条件

**向量传输的必要条件**：
1. **相容性**：$\mathcal{T}_{\mathbf{0}}(\boldsymbol{\zeta}) = \boldsymbol{\zeta}$
2. **线性**：$\mathcal{T}_{\boldsymbol{\eta}}(a\boldsymbol{\zeta}_1 + b\boldsymbol{\zeta}_2) = a\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}_1) + b\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}_2)$
3. **与回缩相容**：传输后的向量确实在目标切空间中

**常用向量传输**：

*投影传输*：最简单的方法，直接投影到目标切空间
$$\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}) = \text{Proj}_{T_{\mathbf{Y}}\mathcal{M}}(\boldsymbol{\zeta}), \quad \mathbf{Y} = \mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})$$

优点：实现简单，计算快速
缺点：不保持向量长度，可能损失正交性
适用：当迭代次数较少或精度要求不高时

*微分回缩传输*：利用回缩的微分结构
$$\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}) = \text{D}\mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})[\boldsymbol{\zeta}]$$

对于QR回缩，有闭式表达：设$\mathbf{X} + \boldsymbol{\eta} = \mathbf{Q}\mathbf{R}$，则
$$\mathcal{T}_{\boldsymbol{\eta}}(\boldsymbol{\zeta}) = \mathbf{Q}\text{skew}(\mathbf{Q}^T\boldsymbol{\zeta}\mathbf{R}^{-T}) + (\mathbf{I} - \mathbf{Q}\mathbf{Q}^T)\boldsymbol{\zeta}\mathbf{R}^{-T}$$
其中$\text{skew}(\mathbf{A}) = (\mathbf{A} - \mathbf{A}^T)/2$。

*平行移动的近似*：对于Stiefel流形，Ring-Wirth公式提供$O(nk^2)$的近似：
$$\mathcal{T}_{\boldsymbol{\eta}}^{\text{par}}(\boldsymbol{\zeta}) \approx \boldsymbol{\zeta} - \frac{1}{2}\mathbf{Y}(\mathbf{Y}^T\boldsymbol{\zeta} + \boldsymbol{\zeta}^T\mathbf{Y})$$
其中$\mathbf{Y} = \mathcal{R}_{\mathbf{X}}(\boldsymbol{\eta})$。

**固定秩流形的向量传输**：需要特别注意保持切向量的三部分结构。设
$$\boldsymbol{\zeta} = \mathbf{U}_X\mathbf{M}_\zeta\mathbf{V}_X^T + \mathbf{U}_\perp\mathbf{N}_\zeta\mathbf{V}_X^T + \mathbf{U}_X\mathbf{P}_\zeta^T\mathbf{V}_\perp^T$$

传输需要更新$\mathbf{U}_X, \mathbf{V}_X$及其正交补，同时适当变换系数矩阵$\mathbf{M}_\zeta, \mathbf{N}_\zeta, \mathbf{P}_\zeta$。

### 10.3.3 算法实现考虑

**计算复杂度权衡**：
- 指数映射：理论最优但计算最贵，$O(n^3)$或更高
- 极回缩：良好的几何性质，$O(nk^2 + k^3)$
- QR回缩：快速稳定，$O(nk^2)$
- 投影传输：最快但可能损失信息，$O(nk^2)$
- 微分回缩传输：中等复杂度，$O(nk^2)$，保持更多性质

**数值稳定性考虑**：
1. **QR分解**：
   - 使用Householder变换而非Gram-Schmidt
   - 对于高瘦矩阵，考虑TSQR（Tall-Skinny QR）
   - 监控$\mathbf{R}$的条件数

2. **极分解**：
   - Newton-Schulz迭代：$\mathbf{X}_{k+1} = \frac{1}{2}\mathbf{X}_k(3\mathbf{I} - \mathbf{X}_k^T\mathbf{X}_k)$
   - 当$\|\boldsymbol{\xi}\|$较大时切换到特征分解方法
   - 使用Halley方法获得三阶收敛

3. **正交化维护**：
   - 定期重正交化防止数值漂移
   - 使用迭代refinement：$\mathbf{X} \leftarrow \mathbf{X} - \frac{1}{2}\mathbf{X}(\mathbf{X}^T\mathbf{X} - \mathbf{I})$

**算法选择决策树**：
```
问题规模？
├─ 小规模 (n < 1000)
│  └─ 使用指数映射和平行移动
├─ 中等规模 (1000 < n < 10000)
│  ├─ 一阶方法：QR回缩 + 投影传输
│  └─ 二阶方法：极回缩 + 微分传输
└─ 大规模 (n > 10000)
   ├─ 稀疏结构？→ 专门设计的回缩
   ├─ 低秩结构？→ 利用SVD的增量更新
   └─ 一般情况 → QR回缩 + 投影传输
```

**性能优化技巧**：
1. **缓存策略**：
   - 重用QR/SVD分解结果
   - 存储正交补的投影算子
   - 预计算常用的矩阵乘积

2. **并行化**：
   - 矩阵乘法使用BLAS Level 3
   - QR分解的块算法
   - 多个向量的批量传输

3. **自适应精度**：
   - 初期迭代使用低精度回缩
   - 接近收敛时提高精度
   - 基于梯度范数动态调整

**实践建议**：
- 先用简单方法（QR+投影）建立baseline
- 仅在必要时升级到复杂方法
- 始终监控约束违反程度
- 考虑混合策略：不同阶段用不同方法

### 10.3.4 收敛性保证

使用回缩和向量传输的算法仍能保持良好的收敛性质，关键是理解几何近似如何影响收敛分析。

**一阶方法的收敛性**：

*梯度下降*：使用任意回缩的Riemannian梯度下降
- **凸函数**：保持$O(1/k)$收敛率
- **强凸函数**：保持线性收敛率$O(\rho^k)$，其中$\rho < 1$
- **非凸函数**：保证收敛到一阶驻点，$\min_{i \leq k} \|\text{grad} f(\mathbf{X}_i)\| = O(1/\sqrt{k})$

关键假设：回缩在紧集上Lipschitz连续。

**二阶方法的收敛性**：

*Newton法*：需要更强的条件
- **回缩条件**：二阶回缩，即$\mathcal{R}_{\mathbf{X}}(\boldsymbol{\xi}) = \exp_{\mathbf{X}}(\boldsymbol{\xi}) + O(\|\boldsymbol{\xi}\|^3)$
- **局部收敛**：在非退化最小值附近保持二次收敛
- **收敛域**：收敛域的大小依赖于回缩的近似质量

*信赖域法*：更鲁棒的全局收敛性
- 一阶回缩即可保证收敛到一阶驻点
- 二阶回缩时可以保证收敛到二阶驻点
- 自适应信赖域半径自动处理回缩的近似误差

**加速方法的收敛性**：

*Nesterov加速*：需要仔细处理向量传输
- **凸函数**：使用等距向量传输可达到$O(1/k^2)$
- **强凸函数**：加速线性收敛率
- **关键技术**：动量项的正确传输和momentum restart策略

Recent结果（2023）表明，即使使用简单的投影传输，通过适当的算法设计仍能保持加速收敛率。

**共轭梯度的收敛性**：

线性CG的流形推广：
- **有限步终止**：在精确算术下，$n$步内收敛（$n$是切空间维数）
- **实际收敛**：依赖于Hessian的条件数$\kappa$，$O(\sqrt{\kappa}\log(1/\epsilon))$
- **向量传输的影响**：使用平行移动或其良好近似保持共轭性

**BFGS的收敛性**：

*超线性收敛*：在适当条件下
- Dennis-Moré条件的流形版本
- 需要向量传输保持割线条件
- L-BFGS：保持R-线性收敛，内存受限时的实用选择

**收敛性分析的关键工具**：

1. **Łojasiewicz不等式**：流形版本用于分析非凸情况
2. **测地凸性**：比欧氏凸性更弱但仍保证良好性质
3. **曲率界**：截面曲率的界影响收敛常数

**实用指导**：
- 一阶方法对回缩和传输的要求最低
- 二阶方法的superior收敛值得额外的计算投入
- 在实践中，简单的回缩+传输often sufficient
- 监控算法进展，必要时提高几何近似的精度

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