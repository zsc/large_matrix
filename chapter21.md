# 附录C：常用矩阵恒等式

在大规模矩阵计算中，巧妙运用矩阵恒等式往往能将原本不可行的计算转化为高效算法。本章汇总了实践中最重要的矩阵恒等式，强调它们在数值计算中的应用价值。每个恒等式不仅给出数学表达，更重要的是说明其计算复杂度的改进和数值稳定性的考量。

## 21.1 矩阵求逆恒等式

### 21.1.1 Woodbury矩阵恒等式

最重要的矩阵求逆公式之一：

$$(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}^T\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{A}^{-1}$$

其中 $\mathbf{A} \in \mathbb{R}^{n \times n}$，$\mathbf{U} \in \mathbb{R}^{n \times k}$，$\mathbf{C} \in \mathbb{R}^{k \times k}$，$\mathbf{V} \in \mathbb{R}^{n \times k}$。

**计算复杂度改进**：当 $k \ll n$ 时，将 $O(n^3)$ 的直接求逆降至 $O(n^2k)$。

**典型应用**：
- 增量最小二乘更新
- 低秩修正的快速求逆
- Schur补计算
- 在线学习算法的协方差更新

### 21.1.2 Sherman-Morrison公式

Woodbury恒等式的特例（$k=1$）：

$$(\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$$

**数值稳定性注意**：分母 $1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}$ 接近零时需特殊处理。

### 21.1.3 块矩阵求逆

对于分块矩阵：

$$\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{pmatrix}^{-1} = \begin{pmatrix} \mathbf{A}^{-1} + \mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1}\mathbf{C}\mathbf{A}^{-1} & -\mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1} \\ -\mathbf{S}^{-1}\mathbf{C}\mathbf{A}^{-1} & \mathbf{S}^{-1} \end{pmatrix}$$

其中 $\mathbf{S} = \mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}$ 是Schur补。

**替代形式**（当 $\mathbf{D}^{-1}$ 更易计算时）：

$$\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{pmatrix}^{-1} = \begin{pmatrix} \mathbf{T}^{-1} & -\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1} \\ -\mathbf{D}^{-1}\mathbf{C}\mathbf{T}^{-1} & \mathbf{D}^{-1} + \mathbf{D}^{-1}\mathbf{C}\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1} \end{pmatrix}$$

其中 $\mathbf{T} = \mathbf{A} - \mathbf{B}\mathbf{D}^{-1}\mathbf{C}$。

### 21.1.4 Neumann级数

当 $\|\mathbf{B}\| < 1$ 时：

$$(\mathbf{I} - \mathbf{B})^{-1} = \sum_{k=0}^{\infty} \mathbf{B}^k = \mathbf{I} + \mathbf{B} + \mathbf{B}^2 + \mathbf{B}^3 + \cdots$$

**实用截断形式**：

$$(\mathbf{I} - \mathbf{B})^{-1} \approx \sum_{k=0}^{K} \mathbf{B}^k$$

误差界：$\|\text{error}\| \leq \frac{\|\mathbf{B}\|^{K+1}}{1-\|\mathbf{B}\|}$

## 21.2 迹与行列式恒等式

### 21.2.1 迹的循环性质

$$\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$$

**计算优化应用**：选择计算量最小的顺序。例如，若 $\mathbf{A} \in \mathbb{R}^{n \times k}$，$\mathbf{B} \in \mathbb{R}^{k \times m}$，$\mathbf{C} \in \mathbb{R}^{m \times n}$，且 $k \ll n, m$，则计算 $\text{tr}(\mathbf{B}\mathbf{C}\mathbf{A})$ 最高效。

### 21.2.2 迹的线性性质

$$\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$$
$$\text{tr}(c\mathbf{A}) = c \cdot \text{tr}(\mathbf{A})$$

### 21.2.3 内积的迹表示

$$\mathbf{x}^T\mathbf{y} = \text{tr}(\mathbf{y}\mathbf{x}^T)$$
$$\langle \mathbf{A}, \mathbf{B} \rangle_F = \text{tr}(\mathbf{A}^T\mathbf{B}) = \text{tr}(\mathbf{B}^T\mathbf{A})$$

### 21.2.4 行列式恒等式

**矩阵行列式引理**：
$$\det(\mathbf{A} + \mathbf{u}\mathbf{v}^T) = \det(\mathbf{A})(1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u})$$

**Sylvester行列式定理**：
$$\det(\mathbf{I}_n + \mathbf{A}\mathbf{B}) = \det(\mathbf{I}_m + \mathbf{B}\mathbf{A})$$

其中 $\mathbf{A} \in \mathbb{R}^{n \times m}$，$\mathbf{B} \in \mathbb{R}^{m \times n}$。

### 21.2.5 对数行列式技巧

为避免数值溢出：
$$\log\det(\mathbf{A}) = \text{tr}(\log(\mathbf{A})) = \sum_{i=1}^n \log(\lambda_i)$$

其中 $\lambda_i$ 是 $\mathbf{A}$ 的特征值。实践中通过Cholesky分解计算：
$$\log\det(\mathbf{A}) = 2\sum_{i=1}^n \log(L_{ii})$$

其中 $\mathbf{L}$ 是 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ 的Cholesky因子。

## 21.3 特征值与奇异值恒等式

### 21.3.1 谱定理应用

对称矩阵 $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$：

$$\mathbf{A}^k = \mathbf{Q}\mathbf{\Lambda}^k\mathbf{Q}^T$$
$$f(\mathbf{A}) = \mathbf{Q}f(\mathbf{\Lambda})\mathbf{Q}^T$$

其中 $f(\mathbf{\Lambda}) = \text{diag}(f(\lambda_1), \ldots, f(\lambda_n))$。

### 21.3.2 特征值扰动界

**Weyl不等式**：设 $\lambda_i(\mathbf{A})$ 表示按降序排列的特征值，则：
$$|\lambda_i(\mathbf{A} + \mathbf{E}) - \lambda_i(\mathbf{A})| \leq \|\mathbf{E}\|_2$$

**Hoffman-Wielandt定理**：
$$\sum_{i=1}^n |\lambda_i(\mathbf{A} + \mathbf{E}) - \lambda_i(\mathbf{A})|^2 \leq \|\mathbf{E}\|_F^2$$

### 21.3.3 交织定理

对于 $n \times n$ 对称矩阵 $\mathbf{A}$ 及其 $(n-1) \times (n-1)$ 主子矩阵 $\mathbf{A}_{-i}$：

$$\lambda_1(\mathbf{A}) \geq \lambda_1(\mathbf{A}_{-i}) \geq \lambda_2(\mathbf{A}) \geq \cdots \geq \lambda_{n-1}(\mathbf{A}_{-i}) \geq \lambda_n(\mathbf{A})$$

### 21.3.4 奇异值恒等式

$$\sigma_i(\mathbf{A}) = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})} = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^T)}$$

**Eckart-Young定理**：最优秩-$k$ 近似
$$\min_{\text{rank}(\mathbf{B})=k} \|\mathbf{A} - \mathbf{B}\|_F = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}$$

### 21.3.5 迹与特征值关系

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i(\mathbf{A})$$
$$\text{tr}(\mathbf{A}^k) = \sum_{i=1}^n \lambda_i^k(\mathbf{A})$$

## 21.4 Kronecker积与Hadamard积恒等式

### 21.4.1 Kronecker积基本性质

$$(\mathbf{A} \otimes \mathbf{B})^T = \mathbf{A}^T \otimes \mathbf{B}^T$$
$$(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$$
$$(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})$$

### 21.4.2 混合积规则

$$(\mathbf{A} \otimes \mathbf{B}) + (\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A} + \mathbf{C}) \otimes \mathbf{B} \quad \text{（仅当} \mathbf{B} = \mathbf{D}\text{）}$$

### 21.4.3 向量化技巧

$$\text{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = (\mathbf{B}^T \otimes \mathbf{A})\text{vec}(\mathbf{X})$$

特别地：
$$\text{vec}(\mathbf{A}\mathbf{X}) = (\mathbf{I} \otimes \mathbf{A})\text{vec}(\mathbf{X})$$
$$\text{vec}(\mathbf{X}\mathbf{B}) = (\mathbf{B}^T \otimes \mathbf{I})\text{vec}(\mathbf{X})$$

### 21.4.4 Hadamard积性质

$$(A \odot B)^T = A^T \odot B^T$$
$$\text{rank}(A \odot B) \leq \text{rank}(A) \cdot \text{rank}(B)$$
$$A \odot (B + C) = (A \odot B) + (A \odot C)$$

**与Kronecker积的关系**：
$$\text{diag}(\mathbf{a}) \mathbf{B} \text{diag}(\mathbf{c}) = \mathbf{B} \odot (\mathbf{a}\mathbf{c}^T)$$

## 21.5 矩阵导数公式

### 21.5.1 基本导数规则

$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}) = \mathbf{A}^T$$
$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^T\mathbf{A}) = \mathbf{A}$$
$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}\mathbf{B}) = \mathbf{A}^T\mathbf{B}^T$$
$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^T\mathbf{A}\mathbf{X}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{X}$$

### 21.5.2 行列式导数

$$\frac{\partial}{\partial \mathbf{X}} \log\det(\mathbf{X}) = \mathbf{X}^{-T}$$
$$\frac{\partial}{\partial \mathbf{X}} \det(\mathbf{X}) = \det(\mathbf{X})\mathbf{X}^{-T}$$

### 21.5.3 逆矩阵导数

$$\frac{\partial}{\partial t} \mathbf{A}^{-1}(t) = -\mathbf{A}^{-1}(t) \frac{\partial \mathbf{A}(t)}{\partial t} \mathbf{A}^{-1}(t)$$

### 21.5.4 链式法则

对于 $f(\mathbf{A}(\mathbf{X}))$：
$$\frac{\partial f}{\partial X_{ij}} = \text{tr}\left(\left(\frac{\partial f}{\partial \mathbf{A}}\right)^T \frac{\partial \mathbf{A}}{\partial X_{ij}}\right)$$

## 21.6 特殊结构矩阵恒等式

### 21.6.1 对称矩阵性质

对于对称矩阵 $\mathbf{S}$：
- 所有特征值为实数
- 存在正交对角化 $\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$
- $\mathbf{S}^{1/2} = \mathbf{Q}\mathbf{\Lambda}^{1/2}\mathbf{Q}^T$（当 $\mathbf{S} \succeq 0$）

### 21.6.2 正交矩阵性质

对于正交矩阵 $\mathbf{Q}$（$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$）：
- $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$（保范性）
- $\det(\mathbf{Q}) = \pm 1$
- 所有特征值的模为1
- $\mathbf{Q}^{-1} = \mathbf{Q}^T$

### 21.6.3 正定矩阵性质

对于正定矩阵 $\mathbf{P} \succ 0$：
- Cholesky分解存在且唯一：$\mathbf{P} = \mathbf{L}\mathbf{L}^T$
- 所有主子矩阵也是正定的
- $\mathbf{A}^T\mathbf{P}\mathbf{A} \succeq 0$，等号成立当且仅当 $\mathbf{A}\mathbf{x} = 0$

### 21.6.4 投影矩阵性质

对于投影矩阵 $\mathbf{P}$（$\mathbf{P}^2 = \mathbf{P}$）：
- 特征值只能是0或1
- $\text{rank}(\mathbf{P}) = \text{tr}(\mathbf{P})$
- 正交投影时：$\mathbf{P} = \mathbf{P}^T$

## 本章小结

本章汇总的矩阵恒等式是大规模计算的基础工具。关键要点：

1. **Woodbury恒等式**是处理低秩更新的核心，将高维求逆转化为低维计算
2. **迹的循环性质**允许重排矩阵乘积以优化计算顺序
3. **Kronecker积的向量化技巧**将矩阵方程转化为向量形式，便于求解
4. **矩阵导数公式**是优化算法和自动微分的理论基础
5. **特殊结构**（对称、正交、正定）的性质可大幅简化计算

记住：选择合适的恒等式往往能将不可行的计算变为实际可行，特别是在处理大规模稀疏或低秩结构时。

## 练习题

### 练习21.1（基础）
证明Sherman-Morrison公式是Woodbury恒等式的特例。

<details>
<summary>提示</summary>
令 $\mathbf{U} = \mathbf{u}$，$\mathbf{V}^T = \mathbf{v}^T$，$\mathbf{C} = 1$。
</details>

<details>
<summary>答案</summary>
在Woodbury恒等式中设置 $\mathbf{U} = \mathbf{u}$（列向量），$\mathbf{V}^T = \mathbf{v}^T$（行向量），$\mathbf{C} = 1$（标量）。则：
$$(\mathbf{A} + \mathbf{u} \cdot 1 \cdot \mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{u}(1^{-1} + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u})^{-1}\mathbf{v}^T\mathbf{A}^{-1}$$

由于 $(1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u})^{-1} = \frac{1}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$，得到Sherman-Morrison公式。
</details>

### 练习21.2（基础）
利用迹的循环性质，找出计算 $\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D})$ 的最优顺序，其中 $\mathbf{A} \in \mathbb{R}^{1000 \times 10}$，$\mathbf{B} \in \mathbb{R}^{10 \times 1000}$，$\mathbf{C} \in \mathbb{R}^{1000 \times 10}$，$\mathbf{D} \in \mathbb{R}^{10 \times 1000}$。

<details>
<summary>提示</summary>
比较不同顺序的计算复杂度，考虑中间矩阵的大小。
</details>

<details>
<summary>答案</summary>
可能的计算顺序及复杂度：
1. $\text{tr}((\mathbf{A}\mathbf{B})(\mathbf{C}\mathbf{D}))$：$\mathbf{A}\mathbf{B}$ 是 $1000 \times 1000$，复杂度 $O(10^7)$
2. $\text{tr}(((\mathbf{A}\mathbf{B})\mathbf{C})\mathbf{D})$：同样需要计算 $1000 \times 1000$ 矩阵
3. $\text{tr}(\mathbf{B}(\mathbf{C}\mathbf{D})\mathbf{A})$：$\mathbf{C}\mathbf{D}$ 是 $1000 \times 1000$，复杂度高
4. $\text{tr}((\mathbf{B}\mathbf{C})(\mathbf{D}\mathbf{A}))$：$\mathbf{B}\mathbf{C}$ 和 $\mathbf{D}\mathbf{A}$ 都是 $10 \times 10$，复杂度 $O(10^5)$

最优顺序是第4种，先计算 $\mathbf{B}\mathbf{C}$ 和 $\mathbf{D}\mathbf{A}$。
</details>

### 练习21.3（基础）
推导正定矩阵 $\mathbf{A}$ 的条件数与其最大最小特征值的关系。

<details>
<summary>提示</summary>
回忆条件数定义 $\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2$。
</details>

<details>
<summary>答案</summary>
对于正定矩阵：
- $\|\mathbf{A}\|_2 = \lambda_{\max}(\mathbf{A})$
- $\|\mathbf{A}^{-1}\|_2 = \lambda_{\max}(\mathbf{A}^{-1}) = \frac{1}{\lambda_{\min}(\mathbf{A})}$

因此：
$$\kappa(\mathbf{A}) = \lambda_{\max}(\mathbf{A}) \cdot \frac{1}{\lambda_{\min}(\mathbf{A})} = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})}$$
</details>

### 练习21.4（挑战）
设 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 是对称正定矩阵，$\mathbf{B} \in \mathbb{R}^{n \times k}$ 且 $k \ll n$。推导一个高效算法计算 $(\mathbf{A} + \mathbf{B}\mathbf{B}^T)^{-1}\mathbf{B}$，避免显式计算逆矩阵。

<details>
<summary>提示</summary>
使用Woodbury恒等式，注意最终目标是计算 $(\mathbf{A} + \mathbf{B}\mathbf{B}^T)^{-1}\mathbf{B}$，而非单独的逆矩阵。
</details>

<details>
<summary>答案</summary>
应用Woodbury恒等式（$\mathbf{U} = \mathbf{V} = \mathbf{B}$，$\mathbf{C} = \mathbf{I}_k$）：
$$(\mathbf{A} + \mathbf{B}\mathbf{B}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{B}(\mathbf{I}_k + \mathbf{B}^T\mathbf{A}^{-1}\mathbf{B})^{-1}\mathbf{B}^T\mathbf{A}^{-1}$$

因此：
$$(\mathbf{A} + \mathbf{B}\mathbf{B}^T)^{-1}\mathbf{B} = \mathbf{A}^{-1}\mathbf{B} - \mathbf{A}^{-1}\mathbf{B}(\mathbf{I}_k + \mathbf{B}^T\mathbf{A}^{-1}\mathbf{B})^{-1}\mathbf{B}^T\mathbf{A}^{-1}\mathbf{B}$$

令 $\mathbf{Z} = \mathbf{A}^{-1}\mathbf{B}$，则：
$$(\mathbf{A} + \mathbf{B}\mathbf{B}^T)^{-1}\mathbf{B} = \mathbf{Z} - \mathbf{Z}(\mathbf{I}_k + \mathbf{B}^T\mathbf{Z})^{-1}\mathbf{B}^T\mathbf{Z} = \mathbf{Z}(\mathbf{I}_k + \mathbf{B}^T\mathbf{Z})^{-1}$$

计算步骤：
1. 解线性系统 $\mathbf{A}\mathbf{Z} = \mathbf{B}$ 得到 $\mathbf{Z}$
2. 计算 $k \times k$ 矩阵 $\mathbf{M} = \mathbf{I}_k + \mathbf{B}^T\mathbf{Z}$
3. 解 $k \times k$ 系统 $\mathbf{M}\mathbf{Y} = \mathbf{I}_k$
4. 计算 $\mathbf{Z}\mathbf{Y}$

总复杂度：$O(n^2k + k^3)$，远优于直接求逆的 $O(n^3)$。
</details>

### 练习21.5（挑战）
证明对于任意矩阵 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$，有：
$$\text{eig}(\mathbf{A} \otimes \mathbf{I} + \mathbf{I} \otimes \mathbf{B}) = \{\lambda_i + \mu_j : i,j = 1,\ldots,n\}$$
其中 $\lambda_i$ 和 $\mu_j$ 分别是 $\mathbf{A}$ 和 $\mathbf{B}$ 的特征值。

<details>
<summary>提示</summary>
考虑 $\mathbf{A}$ 和 $\mathbf{B}$ 的特征向量，利用Kronecker积的性质。
</details>

<details>
<summary>答案</summary>
设 $\mathbf{A}\mathbf{v}_i = \lambda_i\mathbf{v}_i$，$\mathbf{B}\mathbf{w}_j = \mu_j\mathbf{w}_j$。考虑向量 $\mathbf{v}_i \otimes \mathbf{w}_j$：

$$(\mathbf{A} \otimes \mathbf{I} + \mathbf{I} \otimes \mathbf{B})(\mathbf{v}_i \otimes \mathbf{w}_j)$$
$$= (\mathbf{A} \otimes \mathbf{I})(\mathbf{v}_i \otimes \mathbf{w}_j) + (\mathbf{I} \otimes \mathbf{B})(\mathbf{v}_i \otimes \mathbf{w}_j)$$
$$= (\mathbf{A}\mathbf{v}_i) \otimes \mathbf{w}_j + \mathbf{v}_i \otimes (\mathbf{B}\mathbf{w}_j)$$
$$= \lambda_i\mathbf{v}_i \otimes \mathbf{w}_j + \mathbf{v}_i \otimes \mu_j\mathbf{w}_j$$
$$= (\lambda_i + \mu_j)(\mathbf{v}_i \otimes \mathbf{w}_j)$$

因此 $\mathbf{v}_i \otimes \mathbf{w}_j$ 是特征向量，对应特征值 $\lambda_i + \mu_j$。由于这给出了 $n^2$ 个线性独立的特征向量，所以这是完整的特征值集合。
</details>

### 练习21.6（挑战）
给定 $\mathbf{X} \in \mathbb{R}^{n \times p}$ 和对称矩阵 $\mathbf{S} \in \mathbb{R}^{p \times p}$，推导 $\frac{\partial}{\partial \mathbf{S}} \log\det(\mathbf{X}^T\mathbf{X} + \mathbf{S})$，假设 $\mathbf{X}^T\mathbf{X} + \mathbf{S} \succ 0$。

<details>
<summary>提示</summary>
使用链式法则和 $\frac{\partial}{\partial \mathbf{A}} \log\det(\mathbf{A}) = \mathbf{A}^{-T}$。
</details>

<details>
<summary>答案</summary>
令 $\mathbf{M} = \mathbf{X}^T\mathbf{X} + \mathbf{S}$。使用链式法则：

$$\frac{\partial}{\partial \mathbf{S}} \log\det(\mathbf{M}) = \text{tr}\left(\frac{\partial \log\det(\mathbf{M})}{\partial \mathbf{M}} \frac{\partial \mathbf{M}}{\partial \mathbf{S}}\right)$$

已知 $\frac{\partial}{\partial \mathbf{M}} \log\det(\mathbf{M}) = \mathbf{M}^{-T} = \mathbf{M}^{-1}$（因为 $\mathbf{M}$ 对称）。

由于 $\mathbf{M} = \mathbf{X}^T\mathbf{X} + \mathbf{S}$，有 $\frac{\partial M_{ij}}{\partial S_{kl}} = \delta_{ik}\delta_{jl}$。

考虑到 $\mathbf{S}$ 是对称的，最终结果是：
$$\frac{\partial}{\partial \mathbf{S}} \log\det(\mathbf{X}^T\mathbf{X} + \mathbf{S}) = (\mathbf{X}^T\mathbf{X} + \mathbf{S})^{-1}$$

注意：如果不限制 $\mathbf{S}$ 对称，则需要对结果进行对称化。
</details>

### 练习21.7（开放性思考）
在深度学习中，经常需要计算形如 $(\lambda\mathbf{I} + \mathbf{H})^{-1}\mathbf{g}$ 的表达式，其中 $\mathbf{H}$ 是Hessian矩阵，$\mathbf{g}$ 是梯度。讨论当 $\mathbf{H}$ 具有不同结构（低秩、稀疏、Kronecker因子分解）时，如何设计高效算法。

<details>
<summary>提示</summary>
考虑不同结构下的特殊求解技巧，以及如何避免显式形成和存储 $\mathbf{H}$。
</details>

<details>
<summary>答案</summary>
不同结构的处理策略：

1. **低秩结构** $\mathbf{H} = \mathbf{U}\mathbf{U}^T$：
   - 使用Woodbury恒等式：$(\lambda\mathbf{I} + \mathbf{U}\mathbf{U}^T)^{-1} = \frac{1}{\lambda}\mathbf{I} - \frac{1}{\lambda^2}\mathbf{U}(\mathbf{I} + \frac{1}{\lambda}\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T$
   - 只需求解 $k \times k$ 系统（$k$ 是秩）

2. **稀疏结构**：
   - 使用共轭梯度法（CG）迭代求解
   - 利用稀疏矩阵向量乘法
   - 预条件子选择：不完全Cholesky分解

3. **Kronecker因子分解** $\mathbf{H} \approx \mathbf{A} \otimes \mathbf{B}$：
   - $(\lambda\mathbf{I} + \mathbf{A} \otimes \mathbf{B})^{-1}\text{vec}(\mathbf{G}) = \text{vec}(\mathbf{X})$
   - 等价于求解 Sylvester方程：$\lambda\mathbf{X} + \mathbf{A}\mathbf{X}\mathbf{B}^T = \mathbf{G}$
   - 复杂度从 $O(n^3)$ 降至 $O(n^{3/2})$

4. **混合结构**（低秩+对角）$\mathbf{H} = \mathbf{D} + \mathbf{U}\mathbf{U}^T$：
   - 再次使用Woodbury恒等式
   - 对角部分易于处理

5. **隐式表示**（只有Hessian-向量积）：
   - 使用CG法，只需要计算 $\mathbf{H}\mathbf{v}$
   - 通过自动微分实现，避免存储 $\mathbf{H}$

实践建议：
- 始终避免显式形成完整Hessian
- 利用问题的特殊结构
- 考虑精度-效率权衡
- 监控条件数，必要时调整 $\lambda$
</details>

## 常见陷阱与错误

1. **数值稳定性陷阱**
   - Sherman-Morrison公式中分母接近零
   - 解决方案：检查 $|1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}| > \epsilon$，否则回退到直接方法

2. **矩阵求逆的误用**
   - 错误：计算 $\mathbf{A}^{-1}\mathbf{b}$ 时显式求逆
   - 正确：解线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$

3. **Kronecker积的内存爆炸**
   - 错误：显式形成 $\mathbf{A} \otimes \mathbf{B}$
   - 正确：利用向量化技巧或保持因子形式

4. **特征值计算的条件数敏感性**
   - 病态矩阵的特征值计算不可靠
   - 使用SVD作为更稳定的替代

5. **迹运算的计算顺序**
   - 忽视循环性质导致不必要的大矩阵计算
   - 始终选择计算量最小的顺序

6. **对称性的丢失**
   - 数值误差可能破坏对称性
   - 定期进行对称化：$\mathbf{A} \leftarrow (\mathbf{A} + \mathbf{A}^T)/2$

## 最佳实践检查清单

- [ ] 识别矩阵的特殊结构（稀疏、低秩、对称等）
- [ ] 选择利用结构的恒等式
- [ ] 避免显式矩阵求逆，使用线性系统求解
- [ ] 考虑数值稳定性，特别是在接近奇异的情况
- [ ] 利用迹的循环性质优化计算顺序
- [ ] 保持隐式表示以节省内存（如Kronecker积）
- [ ] 在迭代算法中监控条件数
- [ ] 使用对数行列式避免溢出
- [ ] 定期验证对称性和正定性
- [ ] 基准测试不同方法的性能