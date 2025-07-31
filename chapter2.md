# 第2章：Hessian近似的艺术

在优化理论中，Hessian矩阵提供了目标函数的二阶信息，是理解函数局部几何结构的关键。然而，对于大规模问题，直接计算和存储完整的Hessian矩阵往往是不现实的——一个包含百万参数的模型需要存储10^12个元素的Hessian矩阵。本章深入探讨各种Hessian近似技术，这些方法在保持计算效率的同时，巧妙地利用了Hessian的结构特性。我们将学习如何在内存限制下构建有效的二阶优化算法，如何高效计算Hessian-向量乘积，以及如何处理非凸优化中的负曲率问题。

## 2.1 从BFGS到L-BFGS：有限内存方法的深入剖析

### 2.1.1 BFGS更新公式的推导与几何直觉

BFGS (Broyden-Fletcher-Goldfarb-Shanno) 算法是拟牛顿方法中最成功的代表。其核心思想是通过迭代更新来构建Hessian矩阵（或其逆）的近似，而无需显式计算二阶导数。

设 $\mathbf{B}_k$ 为第 $k$ 步的Hessian近似，BFGS更新满足拟牛顿条件（secant equation）：
$$\mathbf{B}_{k+1} \mathbf{s}_k = \mathbf{y}_k$$

其中 $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$ 是步长向量，$\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$ 是梯度差。

BFGS更新公式为：
$$\mathbf{B}_{k+1} = \mathbf{B}_k - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^T \mathbf{B}_k}{\mathbf{s}_k^T \mathbf{B}_k \mathbf{s}_k} + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k}$$

**几何直觉**：BFGS更新可以理解为对当前Hessian近似进行秩-2修正，使其在新的方向 $\mathbf{s}_k$ 上满足曲率信息。第一项去除了旧的信息，第二项加入了新的曲率信息。

**关键洞察**：BFGS保持了近似矩阵的正定性，这在 $\mathbf{y}_k^T \mathbf{s}_k > 0$ （Wolfe条件）下自动满足。

### 2.1.2 Sherman-Morrison-Woodbury公式的巧妙应用

实践中，我们通常需要Hessian逆矩阵的近似 $\mathbf{H}_k \approx \mathbf{B}_k^{-1}$。Sherman-Morrison-Woodbury (SMW) 公式允许我们直接更新逆矩阵：

$$(\mathbf{A} + \mathbf{U}\mathbf{V}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{I} + \mathbf{V}^T\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{A}^{-1}$$

应用SMW公式，我们得到BFGS的逆更新公式：
$$\mathbf{H}_{k+1} = \left(\mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^T\right) \mathbf{H}_k \left(\mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^T\right) + \rho_k \mathbf{s}_k \mathbf{s}_k^T$$

其中 $\rho_k = 1/(\mathbf{y}_k^T \mathbf{s}_k)$。

**计算技巧**：这个公式避免了矩阵求逆，只需要矩阵-向量乘法，计算复杂度从 $O(n^3)$ 降到 $O(n^2)$。

### 2.1.3 L-BFGS的双循环递归算法

L-BFGS (Limited-memory BFGS) 是BFGS的内存高效版本，它只存储最近 $m$ 个向量对 $\{(\mathbf{s}_i, \mathbf{y}_i)\}$，而不是完整的 $n \times n$ 矩阵。

**核心算法**：L-BFGS双循环

```
输入：梯度 g, 历史向量对 {(s_i, y_i)}_{i=k-m}^{k-1}, 初始Hessian逆近似 H_0
输出：搜索方向 d = H_k g

// 第一个循环：从最新到最旧
q = g
for i = k-1, k-2, ..., k-m:
    ρ_i = 1/(y_i^T s_i)
    α_i = ρ_i s_i^T q
    q = q - α_i y_i

// 应用初始Hessian逆
r = H_0 q

// 第二个循环：从最旧到最新
for i = k-m, k-m+1, ..., k-1:
    β = ρ_i y_i^T r
    r = r + s_i (α_i - β)

return d = r
```

**内存复杂度**：$O(mn)$，其中 $m$ 通常选择3-20。

**数学本质**：L-BFGS隐式地构建了 $\mathbf{H}_k$ 的紧凑表示：
$$\mathbf{H}_k = \mathbf{H}_0 + \mathbf{V}_k \mathbf{R}_k^{-1} \mathbf{V}_k^T$$
其中 $\mathbf{V}_k$ 和 $\mathbf{R}_k$ 由历史向量对构成。

### 2.1.4 内存使用与收敛速度的权衡

选择历史长度 $m$ 涉及内存与性能的权衡：

- **小 $m$ (3-5)**：内存占用少，但可能丢失重要的曲率信息
- **大 $m$ (10-20)**：更好地近似Hessian，但内存需求增加

**理论结果**：在强凸函数上，L-BFGS的收敛率为：
$$\|\mathbf{x}_k - \mathbf{x}^*\| \leq C \cdot r^k$$
其中 $r < 1$ 依赖于 $m$ 和问题的条件数。

**实践建议**：
- 对于条件数良好的问题，$m = 3-5$ 通常足够
- 对于病态问题，增加 $m$ 到 10-20
- 监控 $\mathbf{y}_k^T \mathbf{s}_k$ 的值，过小时可能需要跳过更新

### 2.1.5 在深度学习中的实践考量

深度学习带来了独特的挑战：

**1. 随机性处理**：
- 使用较大的批量计算梯度差 $\mathbf{y}_k$
- 应用方差缩减技术（如SVRG）
- 考虑重叠批次策略

**2. 非凸性适应**：
- 添加线搜索确保 $\mathbf{y}_k^T \mathbf{s}_k > 0$
- 使用damped BFGS更新处理负曲率
- 结合trust region方法

**3. 分布式实现**：
- 向量对的高效广播
- 使用all-reduce操作计算内积
- 考虑异步更新策略

**4. 自适应初始化**：
$$\mathbf{H}_0^{(k)} = \frac{\mathbf{s}_{k-1}^T \mathbf{y}_{k-1}}{\mathbf{y}_{k-1}^T \mathbf{y}_{k-1}} \mathbf{I}$$
这个选择基于最新的曲率信息，往往比固定的 $\mathbf{H}_0 = \mathbf{I}$ 更有效。

**研究前沿**：
- 将L-BFGS与Adam等一阶方法结合
- 使用神经网络学习更新规则
- 探索块对角L-BFGS变体

### 2.1.6 分块与结构化L-BFGS变体

大规模问题往往具有特殊结构，利用这些结构可以显著提升L-BFGS的效率：

**1. 分块L-BFGS (Block L-BFGS)**：
当参数具有自然分组时（如神经网络的不同层），可以对每组维护独立的L-BFGS近似：

$$\mathbf{H} = \begin{bmatrix}
\mathbf{H}_1 & & \\
& \mathbf{H}_2 & \\
& & \ddots
\end{bmatrix}$$

优势：
- 捕获不同参数组的尺度差异
- 降低内存需求（每块独立的小历史）
- 自然的并行化

实现考虑：
- 块划分策略（按层、按参数类型、按梯度统计）
- 块间耦合信息的处理
- 自适应块大小调整

**2. 结构化BFGS (Structured BFGS)**：
利用问题的特定结构设计更新：

**Kronecker结构**：
对于形如 $\mathbf{H} = \mathbf{A} \otimes \mathbf{B}$ 的问题：
- 分别更新 $\mathbf{A}$ 和 $\mathbf{B}$ 的L-BFGS近似
- 内存从 $O(n^2)$ 降到 $O(n)$
- 常见于多维优化和张量分解

**低秩加对角结构**：
$$\mathbf{H}^{-1} \approx \mathbf{D}^{-1} + \mathbf{U}\mathbf{V}^T$$
- $\mathbf{D}$：对角部分，捕获局部曲率
- $\mathbf{U}\mathbf{V}^T$：低秩部分，捕获全局相关性
- 适用于具有稀疏Hessian的问题

**3. 压缩L-BFGS (Compact L-BFGS)**：
将L-BFGS表示为紧凑形式：
$$\mathbf{H}_k = \mathbf{H}_0 + [\mathbf{S}_k \quad \mathbf{H}_0\mathbf{Y}_k] \begin{bmatrix} \mathbf{D}_k & \mathbf{L}_k^T \\ \mathbf{L}_k & -\mathbf{H}_0 \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{S}_k^T \\ \mathbf{Y}_k^T\mathbf{H}_0 \end{bmatrix}$$

其中：
- $\mathbf{S}_k = [\mathbf{s}_{k-m}, ..., \mathbf{s}_{k-1}]$
- $\mathbf{Y}_k = [\mathbf{y}_{k-m}, ..., \mathbf{y}_{k-1}]$
- $\mathbf{D}_k = \text{diag}(\mathbf{s}_i^T\mathbf{y}_i)$
- $\mathbf{L}_k$ 是严格下三角矩阵，$(L_k)_{ij} = \mathbf{s}_{k-m+i-1}^T\mathbf{y}_{k-m+j-1}$ for $i > j$

这种表示便于：
- 批量操作多个向量
- 更高效的并行实现
- 与其他矩阵运算的结合

### 2.1.7 收敛性分析与理论界限

理解L-BFGS的理论性质对于算法调优和问题诊断至关重要：

**1. 收敛速率分析**：

**强凸情况**：
对于 $\mu$-强凸、$L$-光滑的函数，L-BFGS的收敛率为：
$$f(\mathbf{x}_k) - f^* \leq \left(1 - \frac{2\mu\gamma}{L + \mu\gamma}\right)^k (f(\mathbf{x}_0) - f^*)$$
其中 $\gamma$ 依赖于L-BFGS近似质量。

**一般凸情况**：
$$f(\mathbf{x}_k) - f^* \leq \frac{C}{k}$$
其中常数 $C$ 依赖于初始点和Hessian近似质量。

**非凸情况**：
在适当的正则性条件下：
$$\min_{i \leq k} \|\nabla f(\mathbf{x}_i)\|^2 \leq \frac{2(f(\mathbf{x}_0) - f^*)}{k\eta}$$

**2. 近似质量的量化**：

**Dennis-Moré条件**：
超线性收敛的充要条件：
$$\lim_{k \to \infty} \frac{\|(\mathbf{H}_k - \mathbf{H}(\mathbf{x}^*))\mathbf{s}_k\|}{\|\mathbf{s}_k\|} = 0$$

**谱条件数界**：
理想情况下，希望：
$$\kappa(\mathbf{H}_k^{-1}\mathbf{H}) \approx 1$$

实践中，监控：
$$\rho_k = \frac{\|\mathbf{H}_k\mathbf{g}_k\|}{\|\mathbf{g}_k\|} \cdot \frac{\|\nabla^2 f(\mathbf{x}_k)^{-1}\mathbf{g}_k\|}{\|\nabla^2 f(\mathbf{x}_k)^{-1}\mathbf{g}_k\|}$$

**3. 内存限制的影响**：

**理论结果**：
- 当 $m \geq n$ 时，L-BFGS退化为完整BFGS
- 当 $m < n$ 时，可能丢失重要的曲率信息
- 存在问题使得 $m < n$ 的L-BFGS无法达到超线性收敛

**实用指导**：
- 对于二次函数，$m = n$ 保证有限步收敛
- 对于一般函数，$m = O(\log n)$ 通常足够
- 监控有效秩：$\text{rank}(\mathbf{S}_k^T\mathbf{Y}_k)$

### 2.1.8 高级实现技巧与优化

**1. 向量化与SIMD优化**：
利用现代CPU的向量指令集：

```
// 向量化的内积计算
float dot_product_simd(float* a, float* b, int n) {
    // 使用AVX/SSE指令集
    // 一次处理多个元素
}
```

关键优化点：
- 内积计算（L-BFGS的主要操作）
- 向量加法和标量乘法
- 数据对齐和预取

**2. 缓存优化策略**：

**循环展开**：
减少循环开销，提高指令级并行：
```
// 展开因子4的示例
for (i = 0; i < n-3; i += 4) {
    sum += a[i] * b[i];
    sum += a[i+1] * b[i+1];
    sum += a[i+2] * b[i+2];
    sum += a[i+3] * b[i+3];
}
```

**数据布局优化**：
- 将频繁访问的向量对连续存储
- 考虑cache line大小（通常64字节）
- 避免false sharing在多线程环境

**3. 数值稳定性增强**：

**Kahan求和在L-BFGS中的应用**：
```
// 高精度累加alpha值
kahan_sum = 0.0
compensation = 0.0
for i in range(m):
    y = alpha[i] - compensation
    t = kahan_sum + y
    compensation = (t - kahan_sum) - y
    kahan_sum = t
```

**缩放技术**：
- 动态缩放防止上溢/下溢
- 使用对数空间计算极小值
- 监控数值范围并自适应调整

**4. 自适应历史管理**：

**动态历史长度**：
根据问题特性动态调整 $m$：
- 初始阶段使用较小的 $m$
- 接近收敛时增加 $m$ 提高精度
- 基于可用内存自适应调整

**选择性更新**：
不是所有的 $(\mathbf{s}_k, \mathbf{y}_k)$ 对都有同等价值：
- 跳过数值不稳定的更新
- 优先保留信息量大的向量对
- 使用信息论准则评估更新质量

### 2.1.9 与现代优化器的融合

**1. L-BFGS与动量方法的结合**：

**L-BFGS-B with Momentum**：
结合L-BFGS的二阶信息和动量的加速效果：
$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + (1-\beta) \mathbf{H}_k \mathbf{g}_k$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{v}_{k+1}$$

优势：
- 更平滑的收敛轨迹
- 对噪声更鲁棒
- 保持二阶收敛性质

**2. 自适应学习率集成**：

**L-BFGS-Adam混合**：
- 使用Adam的自适应学习率
- 用L-BFGS提供搜索方向
- 在不同阶段切换策略

实现框架：
```
if phase == "exploration":
    use Adam with large learning rate
elif phase == "refinement":
    use L-BFGS for fast local convergence
else:  # phase == "hybrid"
    direction = L-BFGS.compute_direction()
    step_size = Adam.compute_step_size()
    update = step_size * direction
```

**3. 正则化与L-BFGS**：

**隐式正则化效应**：
L-BFGS的有限内存特性产生隐式正则化：
- 限制了可表示的曲率信息
- 倾向于低秩解
- 可能有助于泛化

**显式正则化的处理**：
对于 $f(\mathbf{x}) + \lambda R(\mathbf{x})$ 形式的问题：
- 弹性网：将 $\lambda \mathbf{I}$ 加入Hessian近似
- L1正则化：使用orthant-wise L-BFGS
- 结构化正则化：相应调整更新规则

## 2.2 Hessian-vector product的高效计算

在许多二阶优化算法中，我们并不需要完整的Hessian矩阵，而只需要计算Hessian与向量的乘积 $\mathbf{H}\mathbf{v}$。这个观察导致了一类"matrix-free"的方法，它们能够在 $O(n)$ 的内存和 $O(n)$ 的时间内完成计算。

### 2.2.1 自动微分的二阶扩展

自动微分(AD)不仅可以高效计算梯度，还可以扩展到二阶导数。关键洞察是：
$$\mathbf{H}\mathbf{v} = \nabla(\nabla f^T \mathbf{v}) = \nabla(g^T \mathbf{v})$$

其中 $g = \nabla f$ 是梯度。这将Hessian-向量乘积转化为标量函数 $g^T \mathbf{v}$ 的梯度计算。

**前向模式AD**：计算方向导数
$$\frac{d}{dt} f(\mathbf{x} + t\mathbf{v})\big|_{t=0} = \nabla f(\mathbf{x})^T \mathbf{v}$$

**反向模式AD**：计算梯度的梯度
$$\mathbf{H}\mathbf{v} = \frac{d}{dt} \nabla f(\mathbf{x} + t\mathbf{v})\big|_{t=0}$$

**计算复杂度**：两次反向传播的成本，约为梯度计算的2-3倍。

### 2.2.2 Pearlmutter技巧的数学原理

Pearlmutter提出了一种优雅的方法，通过修改反向传播来直接计算 $\mathbf{H}\mathbf{v}$：

**核心思想**：在计算图中同时传播两个量：
1. 标准梯度 $\bar{\mathbf{w}} = \frac{\partial L}{\partial \mathbf{w}}$
2. 方向导数 $\dot{\mathbf{w}} = \frac{\partial \mathbf{w}}{\partial t}\big|_{t=0}$ 当 $\mathbf{w}(t) = \mathbf{w}_0 + t\mathbf{v}$

**传播规则**：对于操作 $\mathbf{y} = f(\mathbf{x})$：
- 前向：$\dot{\mathbf{y}} = \frac{\partial f}{\partial \mathbf{x}} \dot{\mathbf{x}}$
- 反向：$\bar{\mathbf{x}} = \frac{\partial f}{\partial \mathbf{x}}^T \bar{\mathbf{y}}$
- Hessian：$\mathbf{H}\mathbf{v} = \frac{\partial f}{\partial \mathbf{x}}^T \mathbf{H}_f \dot{\mathbf{x}} + \frac{\partial^2 f}{\partial \mathbf{x}^2}[\dot{\mathbf{x}}, \bar{\mathbf{y}}]$

**实现细节**：
- 需要为每个操作定义额外的传播规则
- 可以与现有AD框架集成
- 内存开销仅增加一倍

### 2.2.3 R-operator与L-operator的实现

在自动微分框架中，有两种计算Hessian-向量乘积的算子：

**R-operator** (Right multiplication)：计算 $\mathbf{H}\mathbf{v}$
```
def R_op(f, x, v):
    # 创建dual number: x + ε*v
    # 计算 f(x + ε*v) 并提取 ε 的系数
    g = grad(f, x)
    return grad(lambda x: dot(g(x), v), x)
```

**L-operator** (Left multiplication)：计算 $\mathbf{v}^T\mathbf{H}$
```
def L_op(f, x, v):
    # 使用向量-Jacobian乘积
    g = grad(f, x)
    return vjp(g, x, v)
```

**关系**：由于Hessian的对称性，$\mathbf{v}^T\mathbf{H} = (\mathbf{H}\mathbf{v})^T$

**优化技巧**：
- 使用checkpointing减少内存使用
- 批量计算多个向量的Hessian乘积
- 利用稀疏性和结构

### 2.2.4 在共轭梯度法中的应用

共轭梯度(CG)法是求解线性系统 $\mathbf{H}\mathbf{x} = \mathbf{b}$ 的迭代方法，特别适合大规模问题：

**算法框架**：
```
初始化：r_0 = b - H*x_0, p_0 = r_0
for k = 0, 1, 2, ...:
    α_k = (r_k^T r_k) / (p_k^T H p_k)  # 需要计算 H*p_k
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k - α_k H p_k
    β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + β_k p_k
```

**Newton-CG方法**：
- 使用CG求解Newton方程：$\mathbf{H}_k \mathbf{d}_k = -\mathbf{g}_k$
- 只需要Hessian-向量乘积，不需要存储Hessian
- 可以提前终止（inexact Newton）

**预条件CG (PCG)**：
- 使用预条件矩阵 $\mathbf{M} \approx \mathbf{H}^{-1}$
- 常见选择：对角预条件、不完全Cholesky分解
- 显著加速收敛，特别是对病态问题

**收敛分析**：
CG的误差满足：
$$\|\mathbf{x}_k - \mathbf{x}^*\|_{\mathbf{H}} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|_{\mathbf{H}}$$

其中 $\kappa = \lambda_{max}/\lambda_{min}$ 是条件数。

### 2.2.5 并行化策略与内存优化

大规模Hessian-向量乘积的高效实现需要考虑现代硬件架构：

**1. 数据并行**：
- 将向量 $\mathbf{v}$ 分块：$\mathbf{v} = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_p]$
- 每个处理器计算部分乘积
- 使用all-reduce聚合结果

**2. 模型并行**：
- 将模型参数分布到不同设备
- 使用pipeline并行减少通信开销
- 注意处理跨设备的依赖关系

**3. 内存优化技术**：

**Checkpointing**：
- 不存储所有中间激活值
- 在反向传播时重新计算
- 内存使用从 $O(n)$ 降到 $O(\sqrt{n})$

**混合精度计算**：
- 使用FP16进行前向传播
- FP32累积梯度和Hessian乘积
- 注意数值稳定性

**4. 通信优化**：
- 重叠计算与通信
- 使用压缩技术减少带宽需求
- 探索异步更新策略

**5. 批量Hessian-向量乘积**：
计算 $\mathbf{H}[\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k]$ 比单独计算每个更高效：
- 更好的缓存利用率
- 减少kernel启动开销
- 适合GPU架构

**实践建议**：
- profile代码找出瓶颈
- 使用专门的线性代数库（如cuBLAS）
- 考虑近似技术权衡精度与速度

### 2.2.6 结构化Hessian的特殊处理

许多实际问题的Hessian具有特殊结构，利用这些结构可以大幅提升效率：

**1. 带状Hessian**：
当变量间的依赖关系局部化时，Hessian呈现带状结构：
$$\mathbf{H} = \begin{bmatrix}
* & * & * & 0 & \cdots \\
* & * & * & * & \ddots \\
* & * & * & * & * \\
0 & * & * & * & * \\
\vdots & \ddots & * & * & *
\end{bmatrix}$$

高效计算策略：
- 只存储带内元素
- 利用稀疏矩阵-向量乘法
- 带宽 $b$ 时，复杂度降至 $O(bn)$

**2. 分块Hessian**：
对于多任务学习或分层模型：
$$\mathbf{H} = \begin{bmatrix}
\mathbf{H}_{11} & \mathbf{H}_{12} & \cdots \\
\mathbf{H}_{21} & \mathbf{H}_{22} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

优化技术：
- 分块计算Hessian-向量乘积
- 利用块间稀疏性
- 并行处理不同块

**3. 低秩扰动结构**：
$$\mathbf{H} = \mathbf{D} + \mathbf{U}\mathbf{V}^T$$
其中 $\mathbf{D}$ 是对角矩阵，$\mathbf{U}, \mathbf{V} \in \mathbb{R}^{n \times r}$，$r \ll n$。

高效计算：
$$\mathbf{H}\mathbf{v} = \mathbf{D}\mathbf{v} + \mathbf{U}(\mathbf{V}^T\mathbf{v})$$
- 先计算 $\mathbf{V}^T\mathbf{v}$（$O(rn)$）
- 再计算 $\mathbf{U}$ 与结果的乘积
- 总复杂度：$O(rn)$ 而非 $O(n^2)$

**4. Kronecker积结构**：
对于张量化模型或多线性问题：
$$\mathbf{H} = \mathbf{A}_1 \otimes \mathbf{A}_2 \otimes \cdots \otimes \mathbf{A}_d$$

利用Kronecker积性质：
$$(\mathbf{A} \otimes \mathbf{B})\text{vec}(\mathbf{X}) = \text{vec}(\mathbf{B}\mathbf{X}\mathbf{A}^T)$$

这将 $O(n^2)$ 的操作降至 $O(dn^{2/d})$。

### 2.2.7 随机化Hessian-向量乘积

对于超大规模问题，即使 $O(n)$ 的计算也可能过于昂贵，随机化方法提供了可行的替代：

**1. 子采样方法**：
不使用完整数据计算Hessian-向量乘积：
$$\mathbf{H}\mathbf{v} \approx \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla^2 f_i(\mathbf{x}) \mathbf{v}$$

其中 $\mathcal{B}$ 是随机选择的小批量。

方差缩减技术：
- 使用控制变量：$\tilde{\mathbf{H}}\mathbf{v} = \mathbf{H}_0\mathbf{v} + \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (\nabla^2 f_i - \mathbf{H}_0)\mathbf{v}$
- 重要性采样：根据曲率变化选择样本
- 渐进方差缩减：随迭代增加批量大小

**2. 随机投影方法**：
使用随机矩阵 $\mathbf{S} \in \mathbb{R}^{n \times k}$ 近似：
$$\mathbf{H} \approx \mathbf{H}\mathbf{S}(\mathbf{S}^T\mathbf{H}\mathbf{S})^{-1}\mathbf{S}^T\mathbf{H}$$

常用随机矩阵：
- 高斯随机矩阵：$S_{ij} \sim \mathcal{N}(0, 1/k)$
- 稀疏随机矩阵：大部分元素为零
- Hadamard矩阵的随机子集

**3. 量子启发的采样**：
利用量子计算的思想设计经典采样策略：
- 基于振幅放大的重要性采样
- 利用纠缠结构减少采样复杂度
- 量子蒙特卡洛方法的经典模拟

### 2.2.8 自动微分框架的高级应用

现代自动微分框架提供了强大的工具来计算Hessian-向量乘积：

**1. 高阶自动微分**：
```python
# JAX示例
def hvp(f, x, v):
    return jax.grad(lambda x: jax.vmap(jax.grad(f))(x) @ v)(x)

# PyTorch示例  
def hvp(f, x, v):
    grad = torch.autograd.grad(f(x), x, create_graph=True)[0]
    return torch.autograd.grad(grad @ v, x)[0]
```

**2. 向量化计算多个HVP**：
同时计算多个方向的Hessian-向量乘积：
```python
# 批量HVP计算
def batch_hvp(f, x, V):  # V是多个向量组成的矩阵
    return jax.vmap(lambda v: hvp(f, x, v), in_axes=1, out_axes=1)(V)
```

**3. 混合模式微分**：
结合前向和反向模式获得最佳效率：
- 对于"高瘦"Jacobian（$m \ll n$），使用前向模式
- 对于"矮胖"Jacobian（$m \gg n$），使用反向模式
- 对于Hessian，可以混合使用

**4. 自定义VJP规则**：
为特殊操作定义高效的向量-Jacobian乘积：
```python
@jax.custom_vjp
def custom_op(x):
    # 前向计算
    return result

def custom_op_fwd(x):
    # 保存前向传播需要的中间结果
    return result, saved_values

def custom_op_bwd(saved_values, g):
    # 定义高效的反向传播
    return vjp_result

custom_op.defvjp(custom_op_fwd, custom_op_bwd)
```

### 2.2.9 应用案例：大规模科学计算

**1. 分子动力学中的Hessian计算**：
对于 $N$ 个原子的系统，Hessian是 $3N \times 3N$ 的矩阵：
- 利用力场的局部性（截断半径）
- 使用邻居列表加速
- 并行计算不同原子对的贡献

**2. 偏微分方程的离散化**：
有限元方法产生的Hessian通常稀疏且结构化：
- 利用网格的规则性
- 使用多重网格方法作为预条件子
- 域分解实现并行化

**3. 图神经网络的二阶优化**：
GNN的Hessian具有图结构诱导的稀疏性：
- 消息传递框架下的高效HVP
- 利用图的谱性质
- 局部化计算减少通信

**4. 变分推断中的自然梯度**：
Fisher信息矩阵的高效计算：
- 利用概率模型的因子分解
- Monte Carlo估计HVP
- 结构化变分族的特殊处理

### 2.2.10 未来研究方向

**1. 硬件-算法协同设计**：
- 针对新型加速器（如Graphcore IPU）优化HVP
- 利用近数据计算减少内存带宽压力
- 探索模拟计算器的可能性

**2. 理论界限的改进**：
- 随机HVP的最优采样复杂度
- 结构化问题的下界
- 通信复杂度的理论分析

**3. 与机器学习的深度结合**：
- 学习问题相关的HVP近似
- 元学习优化HVP计算策略
- 神经网络加速经典算法

**4. 量子-经典混合算法**：
- 量子计算机上的HVP原语
- 经典预处理+量子核心计算
- 容错量子计算的算法设计

## 2.3 负曲率方向的检测与利用

在非凸优化中，Hessian矩阵可能有负特征值，对应的特征向量指向负曲率方向。沿着这些方向移动可以帮助算法逃离鞍点，这对深度学习等应用至关重要。

### 2.3.1 鞍点问题的本质

**鞍点定义**：点 $\mathbf{x}^*$ 是鞍点如果 $\nabla f(\mathbf{x}^*) = 0$ 且Hessian $\mathbf{H}(\mathbf{x}^*)$ 有至少一个负特征值。

**在高维空间的普遍性**：
- 随机矩阵理论表明，高维空间中鞍点远多于局部极小值
- 对于随机高斯函数，鞍点与局部极小值的比例随维度指数增长
- 深度网络的损失景观主要由鞍点而非局部极小值主导

**鞍点的类型**：
1. **严格鞍点**：至少有一个特征值 $\lambda < -\epsilon < 0$
2. **退化鞍点**：最小特征值接近零
3. **高阶鞍点**：一阶和二阶导数都为零

**逃逸的必要性**：
- 一阶方法在鞍点附近收敛极慢
- 可能长时间停滞在次优解
- 负曲率方向提供了快速下降的路径

### 2.3.2 Lanczos算法与最小特征值估计

Lanczos算法是计算大规模对称矩阵极端特征值的有效方法：

**算法流程**：
```
输入：对称矩阵 H (通过 H*v 访问), 初始向量 v_0
输出：三对角矩阵 T_k, 正交基 V_k

v_1 = v_0 / ||v_0||
for j = 1 to k:
    w = H * v_j
    α_j = v_j^T * w
    w = w - α_j * v_j - β_{j-1} * v_{j-1}  (β_0 = 0)
    β_j = ||w||
    v_{j+1} = w / β_j
```

**三对角矩阵**：
$$\mathbf{T}_k = \begin{bmatrix}
\alpha_1 & \beta_1 & & \\
\beta_1 & \alpha_2 & \beta_2 & \\
& \beta_2 & \ddots & \beta_{k-1} \\
& & \beta_{k-1} & \alpha_k
\end{bmatrix}$$

**关键性质**：
- $\mathbf{T}_k$ 的特征值（Ritz值）近似 $\mathbf{H}$ 的极端特征值
- 通常 10-30 次迭代就能得到好的近似
- 可以同时估计最小和最大特征值

**用于负曲率检测**：
1. 运行Lanczos算法 k 步
2. 计算 $\mathbf{T}_k$ 的最小特征值 $\theta_{min}$ 和对应特征向量 $\mathbf{y}$
3. 如果 $\theta_{min} < -\epsilon$，负曲率方向为 $\mathbf{d} = \mathbf{V}_k \mathbf{y}$

**数值稳定性考虑**：
- 使用完全正交化避免精度损失
- 监控正交性：$|\mathbf{v}_i^T \mathbf{v}_j| < \epsilon$
- 考虑块Lanczos变体提高鲁棒性

### 2.3.3 随机化方法：加速逃逸鞍点

随机化技术可以显著加速鞍点逃逸：

**1. 随机扰动方法**：
- 当梯度范数小于阈值时，添加随机噪声
- 理论保证：以高概率逃离严格鞍点
- 噪声尺度：$\mathcal{N}(0, \sigma^2 \mathbf{I})$，其中 $\sigma \propto \sqrt{\epsilon/d}$

**2. 随机化Lanczos**：
- 使用多个随机起始向量
- 并行计算提高效率
- 更鲁棒地找到负曲率方向

**3. Oja's算法变体**：
在线估计最小特征向量：
$$\mathbf{v}_{t+1} = \mathbf{v}_t - \eta_t (\mathbf{H} \mathbf{v}_t + \lambda_t \mathbf{v}_t)$$
其中 $\lambda_t$ 是当前特征值估计。

**4. 加速方法**：
Accelerated Gradient Descent with Negative Curvature (AGD+NC)：
- 结合Nesterov加速和负曲率利用
- 理论收敛率：$O(\epsilon^{-7/4})$ vs 标准GD的 $O(\epsilon^{-2})$

**实践技巧**：
- 自适应选择检测频率
- 使用warm-start减少计算
- 结合一阶信息判断是否需要二阶信息

### 2.3.4 Trust Region框架下的负曲率利用

Trust Region方法提供了利用负曲率的原则性框架：

**子问题**：
$$\min_{\|\mathbf{d}\| \leq \Delta} m_k(\mathbf{d}) = f_k + \mathbf{g}_k^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H}_k \mathbf{d}$$

**Moré-Sorensen算法**：
精确求解trust region子问题：
1. 如果 $\mathbf{H}_k \succeq 0$ 且 $\|\mathbf{H}_k^{-1} \mathbf{g}_k\| \leq \Delta$：$\mathbf{d}^* = -\mathbf{H}_k^{-1} \mathbf{g}_k$
2. 否则，寻找 $\lambda \geq 0$ 使得：
   - $(\mathbf{H}_k + \lambda \mathbf{I}) \mathbf{d} = -\mathbf{g}_k$
   - $\|\mathbf{d}\| = \Delta$ （如果约束活跃）
   - $\mathbf{H}_k + \lambda \mathbf{I} \succeq 0$

**负曲率的利用**：
当 $\mathbf{H}_k$ 有负特征值时：
- 如果梯度在负曲率方向有分量，自然利用
- 否则，沿最小特征向量移动到边界
- 保证目标函数的充分下降

**两阶段方法**：
```
阶段1：计算Cauchy点
d_c = -α_c g_k, where α_c = argmin_{α} m_k(-α g_k)

阶段2：从Cauchy点出发，寻找更好的解
使用CG或Lanczos在子空间中优化
如果遇到负曲率，沿该方向到边界
```

**收敛性保证**：
- 全局收敛到二阶必要条件
- 局部二次收敛（在适当条件下）
- 对非凸问题的鲁棒性

### 2.3.5 在非凸优化中的理论保证

利用负曲率可以获得更强的理论保证：

**1. 二阶必要条件**：
局部极小值 $\mathbf{x}^*$ 必须满足：
- $\nabla f(\mathbf{x}^*) = 0$ （一阶条件）
- $\nabla^2 f(\mathbf{x}^*) \succeq 0$ （二阶条件）

**2. 逃逸鞍点的复杂度**：

**Cubic Regularization**：
$$\min_{\mathbf{d}} f_k + \mathbf{g}_k^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H}_k \mathbf{d} + \frac{\sigma}{3} \|\mathbf{d}\|^3$$
- 迭代复杂度：$O(\epsilon^{-3/2})$ 到二阶驻点
- 自适应选择 $\sigma$ 参数

**带负曲率的加速方法**：
- SPIDER-SFO+：$O(\epsilon^{-3})$ 样本复杂度
- SARAH-SOTA：改进的方差缩减技术

**3. 随机算法的保证**：

**Perturbed GD (PGD)**：
```
if ||∇f(x_t)|| ≤ ε:
    x_{t+1} = x_t + ξ_t,  ξ_t ~ N(0, σ²I)
else:
    x_{t+1} = x_t - η∇f(x_t)
```
- 以 $1-\delta$ 概率逃离鞍点
- 时间复杂度：$\tilde{O}(\log(d/\delta)/\epsilon^4)$

**4. 景观分析**：

**严格鞍点性质**：
许多机器学习问题满足严格鞍点性质：
- 所有鞍点都是严格的（有负曲率）
- 所有局部极小值都是全局最优
- 例如：矩阵补全、相位恢复、某些神经网络

**Polyak-Łojasiewicz (PL) 条件**：
$$\|\nabla f(\mathbf{x})\|^2 \geq 2\mu (f(\mathbf{x}) - f^*)$$
- 比强凸性弱，但保证线性收敛
- 在鞍点附近可能不满足，需要负曲率

**研究前沿**：
- 高阶方法（利用三阶信息）
- 分布式环境下的负曲率检测
- 与现代深度学习优化器的集成
- 隐式偏差与负曲率的关系

## 2.4 数值稳定性与条件数控制

Hessian近似的数值稳定性直接影响优化算法的可靠性和收敛性。本节探讨如何识别和缓解数值问题，设计鲁棒的算法。

### 2.4.1 条件数恶化的根源分析

**条件数定义**：
$$\kappa(\mathbf{H}) = \frac{\lambda_{max}(\mathbf{H})}{\lambda_{min}(\mathbf{H})} = \|\mathbf{H}\| \cdot \|\mathbf{H}^{-1}\|$$

**恶化来源**：

**1. 问题固有的病态性**：
- 参数尺度差异巨大（如神经网络不同层）
- 高度相关的特征
- 接近奇异的设计矩阵

**2. 算法引入的病态性**：
- BFGS更新可能累积数值误差
- 有限精度算术的舍入误差
- 不适当的步长选择

**3. 近似引起的问题**：
- 有限差分近似的步长选择
- 截断误差vs舍入误差的权衡
- 低秩近似的秩不足

**诊断工具**：
```
条件数估计（不需要完整矩阵）：
1. 使用Lanczos估计极端特征值
2. 条件数估计器：CONDEST算法
3. 监控 ||H*v||/||v|| 的变化范围
```

**预警信号**：
- L-BFGS中 $\mathbf{y}_k^T \mathbf{s}_k$ 接近零
- CG迭代数急剧增加
- 优化步长变得极小
- 目标函数值震荡

### 2.4.2 正则化技术：Levenberg-Marquardt阻尼

Levenberg-Marquardt (LM) 方法通过添加阻尼项改善条件数：

**基本形式**：
$$(\mathbf{H} + \lambda \mathbf{I}) \mathbf{d} = -\mathbf{g}$$

其中 $\lambda > 0$ 是阻尼参数。

**自适应策略**：
```
初始化：λ = λ_0
repeat:
    求解 (H + λI)d = -g
    计算实际下降 Δf_actual = f(x+d) - f(x)
    计算预测下降 Δf_pred = -g^T d - 0.5 d^T H d
    比率 ρ = Δf_actual / Δf_pred
    
    if ρ > 0.75:  # 很好的近似
        λ = λ / 2
    elif ρ < 0.25:  # 差的近似
        λ = λ * 2
```

**理论性质**：
- 条件数改善：$\kappa(\mathbf{H} + \lambda \mathbf{I}) \leq \frac{\lambda_{max} + \lambda}{\lambda_{min} + \lambda} < \kappa(\mathbf{H})$
- 当 $\lambda \to \infty$：退化为梯度下降
- 当 $\lambda \to 0$：接近牛顿法

**变体与扩展**：

**1. 自适应对角正则化**：
$$(\mathbf{H} + \mathbf{D}) \mathbf{d} = -\mathbf{g}$$
其中 $\mathbf{D} = \text{diag}(\lambda_1, ..., \lambda_n)$ 根据参数敏感度调整。

**2. Trust Region联系**：
LM更新等价于求解：
$$\min_{\mathbf{d}} \mathbf{g}^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H} \mathbf{d} \quad \text{s.t.} \quad \|\mathbf{d}\| \leq \Delta$$
其中 $\lambda$ 是Lagrange乘子。

**3. 概率解释**：
添加 $\lambda \mathbf{I}$ 相当于对参数施加高斯先验 $\mathcal{N}(0, \lambda^{-1} \mathbf{I})$。

### 2.4.3 预条件子设计的艺术

预条件子 $\mathbf{M} \approx \mathbf{H}^{-1}$ 可以显著改善迭代方法的收敛性：

**预条件系统**：
$$\mathbf{M}^{-1} \mathbf{H} \mathbf{d} = -\mathbf{M}^{-1} \mathbf{g}$$

**设计原则**：
1. **易于计算**：$\mathbf{M}\mathbf{v}$ 应该高效
2. **良好近似**：$\mathbf{M}^{-1}\mathbf{H} \approx \mathbf{I}$
3. **数值稳定**：保持正定性
4. **内存高效**：稀疏或结构化

**常用预条件子**：

**1. 对角预条件（Jacobi）**：
$$\mathbf{M} = \text{diag}(\mathbf{H})^{-1}$$
- 最简单，适合对角占优矩阵
- 可以使用running average更新

**2. 不完全Cholesky分解**：
$$\mathbf{H} \approx \mathbf{L}\mathbf{L}^T, \quad \mathbf{M} = (\mathbf{L}\mathbf{L}^T)^{-1}$$
- 控制填充水平权衡精度与稀疏性
- IC(0): 零填充，IC(k): k级填充

**3. BFGS作为预条件子**：
使用L-BFGS近似作为预条件：
$$\mathbf{M} = \mathbf{H}_k^{LBFGS}$$
- 自然地包含曲率信息
- 可以重用已有计算

**4. 多级预条件**：
```
粗网格校正 + 细网格平滑
M^{-1} = S^T (I - P(P^T H P)^{-1} P^T H) S + P(P^T H P)^{-1} P^T
```
其中 $\mathbf{P}$ 是限制算子，$\mathbf{S}$ 是平滑算子。

**自适应预条件**：
- 监控预条件效果：$\kappa(\mathbf{M}^{-1}\mathbf{H})$
- 动态更新策略
- 均衡预处理成本与迭代次数

### 2.4.4 数值误差的传播分析

理解误差如何在Hessian近似中传播对设计鲁棒算法至关重要：

**误差来源**：
1. **舍入误差**：$fl(x \pm y) = (x \pm y)(1 + \delta)$，$|\delta| \leq \epsilon_{machine}$
2. **截断误差**：有限精度表示
3. **方法误差**：近似算法固有误差

**误差传播分析**：

**1. 矩阵-向量乘积**：
$$\|fl(\mathbf{H}\mathbf{v}) - \mathbf{H}\mathbf{v}\| \leq n\epsilon_{machine}\|\mathbf{H}\|\|\mathbf{v}\| + O(\epsilon_{machine}^2)$$

**2. BFGS更新误差**：
设 $\tilde{\mathbf{H}}_k$ 是计算的近似，$\mathbf{H}_k$ 是精确值：
$$\|\tilde{\mathbf{H}}_{k+1} - \mathbf{H}_{k+1}\| \leq \|\tilde{\mathbf{H}}_k - \mathbf{H}_k\| + C\epsilon_{machine}$$
误差可能累积！

**3. 求解线性系统**：
相对误差界：
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{H}) \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|}$$
其中 $\mathbf{r} = \mathbf{b} - \mathbf{H}\tilde{\mathbf{x}}$ 是残差。

**稳定性增强技术**：

**1. 迭代精化**：
```
求解 H x = b:
1. 计算近似解 x_0
2. for k = 0, 1, ...
   r_k = b - H x_k
   求解 H δ_k = r_k
   x_{k+1} = x_k + δ_k
```

**2. 混合精度策略**：
- 低精度计算主体
- 高精度累积关键量
- 定期高精度校正

**3. Kahan求和**：
减少浮点数累加误差：
```
sum = 0, c = 0
for x in values:
    y = x - c
    t = sum + y
    c = (t - sum) - y
    sum = t
```

### 2.4.5 鲁棒性增强技巧

**1. 安全保护措施**：

**数值检查**：
```
BFGS更新前检查：
if y^T s < ε * ||y|| * ||s||:
    跳过更新或使用damped BFGS
    
if ||H|| > max_norm or cond(H) > max_cond:
    重置为 H = I 或其他安全值
```

**2. 修正技术**：

**Powell's修正**：
确保 $\mathbf{y}^T \mathbf{s} > 0$：
$$\tilde{\mathbf{y}} = \theta \mathbf{y} + (1-\theta) \mathbf{B}_k \mathbf{s}$$
其中 $\theta$ 选择使得 $\tilde{\mathbf{y}}^T \mathbf{s} \geq 0.2 \mathbf{s}^T \mathbf{B}_k \mathbf{s}$

**3. 自适应精度控制**：
- 根据问题规模调整容差
- 监控相对误差而非绝对误差
- 使用问题相关的停止准则

**4. 重启策略**：
- 定期重置Hessian近似
- 基于性能指标触发重启
- 保留部分历史信息的软重启

**实践建议总结**：
1. 始终监控条件数和数值稳定性指标
2. 使用相对误差准则而非绝对误差
3. 实现防御性编程，检查异常情况
4. 保持算法的可解释性和可调试性
5. 在精度和效率之间找到平衡点

**研究方向**：
- 自适应精度优化算法
- 硬件感知的数值稳定性
- 随机舍入的影响分析
- 量子计算中的数值稳定性

## 本章小结

本章深入探讨了Hessian近似的各种技术，这些方法使得二阶优化在大规模问题上成为可能：

**关键概念总结**：

1. **有限内存方法 (L-BFGS)**：
   - 通过存储有限的向量对 $\{(\mathbf{s}_i, \mathbf{y}_i)\}$ 隐式表示Hessian逆
   - 双循环递归算法实现 $O(mn)$ 内存和计算复杂度
   - Sherman-Morrison-Woodbury公式的巧妙应用

2. **Hessian-向量乘积**：
   - 无需显式构建Hessian矩阵，通过 $\mathbf{H}\mathbf{v} = \nabla(g^T\mathbf{v})$ 计算
   - Pearlmutter技巧和R/L-operator的实现
   - 在共轭梯度法和Newton-CG中的核心作用

3. **负曲率利用**：
   - Lanczos算法高效检测负特征值
   - 随机化方法加速鞍点逃逸
   - Trust Region框架下的原则性处理

4. **数值稳定性**：
   - 条件数控制通过Levenberg-Marquardt阻尼
   - 预条件子设计改善收敛性
   - 误差传播分析和鲁棒性增强技术

**重要公式回顾**：

- BFGS逆更新：$\mathbf{H}_{k+1} = (\mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^T) \mathbf{H}_k (\mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^T) + \rho_k \mathbf{s}_k \mathbf{s}_k^T$
- 条件数定义：$\kappa(\mathbf{H}) = \lambda_{max}/\lambda_{min}$
- Trust Region子问题：$\min_{\|\mathbf{d}\| \leq \Delta} f_k + \mathbf{g}_k^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H}_k \mathbf{d}$
- LM正则化：$(\mathbf{H} + \lambda \mathbf{I}) \mathbf{d} = -\mathbf{g}$

## 练习题

### 基础题

**练习 2.1**：推导BFGS更新公式
证明BFGS更新公式保持对称性和正定性。具体地，如果 $\mathbf{B}_k \succ 0$ 且 $\mathbf{y}_k^T \mathbf{s}_k > 0$，证明 $\mathbf{B}_{k+1} \succ 0$。

*提示*：使用谱分解和Cauchy-Schwarz不等式。

<details>
<summary>答案</summary>

设 $\mathbf{B}_k = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$ 为谱分解，其中 $\mathbf{\Lambda} = \text{diag}(\lambda_1, ..., \lambda_n) \succ 0$。

BFGS更新可写为：
$$\mathbf{B}_{k+1} = \mathbf{B}_k + \mathbf{u}\mathbf{u}^T - \mathbf{v}\mathbf{v}^T$$
其中 $\mathbf{u} = \mathbf{y}_k/\sqrt{\mathbf{y}_k^T\mathbf{s}_k}$，$\mathbf{v} = \mathbf{B}_k\mathbf{s}_k/\sqrt{\mathbf{s}_k^T\mathbf{B}_k\mathbf{s}_k}$。

对任意 $\mathbf{x} \neq \mathbf{0}$：
$$\mathbf{x}^T\mathbf{B}_{k+1}\mathbf{x} = \mathbf{x}^T\mathbf{B}_k\mathbf{x} + (\mathbf{x}^T\mathbf{u})^2 - (\mathbf{x}^T\mathbf{v})^2$$

需要证明这个表达式为正。考虑最坏情况，当 $\mathbf{x}$ 与 $\mathbf{v}$ 共线时...（完整证明涉及构造性论证）
</details>

**练习 2.2**：L-BFGS内存复杂度分析
设 $n$ 是问题维度，$m$ 是存储的向量对数。分析L-BFGS算法的时间和空间复杂度，并与完整BFGS比较。

*提示*：考虑双循环中的操作次数。

<details>
<summary>答案</summary>

空间复杂度：
- L-BFGS：$O(mn)$ 存储 $m$ 个 $n$ 维向量对
- 完整BFGS：$O(n^2)$ 存储 $n \times n$ 矩阵

时间复杂度（每次迭代）：
- L-BFGS：$O(mn)$ 双循环，每步 $O(n)$ 内积
- 完整BFGS：$O(n^2)$ 矩阵更新

当 $m \ll n$ 时，L-BFGS显著节省内存和计算。
</details>

**练习 2.3**：Hessian-向量乘积的有限差分近似
给定函数 $f: \mathbb{R}^n \to \mathbb{R}$ 和向量 $\mathbf{v}$，如何用有限差分近似 $\mathbf{H}\mathbf{v}$？分析截断误差和舍入误差。

*提示*：考虑 $\nabla f(\mathbf{x} + h\mathbf{v})$ 的Taylor展开。

<details>
<summary>答案</summary>

有限差分公式：
$$\mathbf{H}\mathbf{v} \approx \frac{\nabla f(\mathbf{x} + h\mathbf{v}) - \nabla f(\mathbf{x})}{h}$$

误差分析：
- 截断误差：$O(h)$ 来自Taylor展开的高阶项
- 舍入误差：$O(\epsilon_{machine}/h)$ 来自数值计算

最优步长：$h_{opt} = \sqrt{\epsilon_{machine}}$，总误差 $O(\sqrt{\epsilon_{machine}})$。
</details>

### 挑战题

**练习 2.4**：块L-BFGS设计
设计一个块L-BFGS算法，同时更新多个方向。分析其优势和实现挑战。

*提示*：考虑块向量 $\mathbf{S}_k = [\mathbf{s}_1, ..., \mathbf{s}_b]$ 和 $\mathbf{Y}_k = [\mathbf{y}_1, ..., \mathbf{y}_b]$。

<details>
<summary>答案</summary>

块更新公式：
$$\mathbf{H}_{k+1} = \mathbf{H}_k + (\mathbf{S}_k - \mathbf{H}_k\mathbf{Y}_k)(\mathbf{Y}_k^T\mathbf{S}_k)^{-1}\mathbf{Y}_k^T$$

优势：
- 更好的缓存利用率
- 并行化潜力
- 可能更好的收敛性

挑战：
- 需要求解 $b \times b$ 线性系统
- 存储需求增加到 $O(mbn)$
- 数值稳定性更复杂
</details>

**练习 2.5**：自适应负曲率检测
设计一个算法，自适应地决定何时进行负曲率检测，平衡计算成本和收敛速度。

*提示*：监控梯度范数下降率和步长变化。

<details>
<summary>答案</summary>

自适应策略：
1. 监控指标：
   - 梯度范数比率：$r_k = \|\mathbf{g}_k\|/\|\mathbf{g}_{k-1}\|$
   - 步长趋势：$\alpha_k$ 的移动平均
   
2. 触发条件：
   - 如果 $r_k > 0.9$ 连续 $T$ 步（停滞）
   - 或 $\bar{\alpha} < \epsilon$ （步长过小）
   - 则进行Lanczos检测

3. 动态调整：
   - 成功逃离后增加检测间隔
   - 多次未检测到负曲率则减少频率
</details>

**练习 2.6**：混合精度L-BFGS
设计一个混合精度L-BFGS实现，在FP16和FP32之间智能切换。分析精度损失和性能提升的权衡。

*提示*：关键量（如 $\mathbf{y}_k^T\mathbf{s}_k$）使用高精度。

<details>
<summary>答案</summary>

混合精度策略：
1. FP16存储：向量对 $\{(\mathbf{s}_i, \mathbf{y}_i)\}$
2. FP32计算：
   - 内积 $\mathbf{y}_i^T\mathbf{s}_i$ 和 $\rho_i$
   - 累积量 $\alpha_i$
   
3. 自适应切换：
   - 监控 $\|\mathbf{y}_k\|/\|\mathbf{s}_k\|$ 比率
   - 过大时升级到FP32避免溢出
   
性能分析：
- 内存减少50%
- 计算加速1.5-2x（GPU依赖）
- 精度损失通常可接受（监控收敛）
</details>

**练习 2.7**：分布式Trust Region求解
设计一个分布式算法求解大规模Trust Region子问题，考虑通信成本。

*提示*：使用Lanczos迭代的分布式版本。

<details>
<summary>答案</summary>

分布式策略：
1. 数据分区：参数空间分块
2. 分布式Lanczos：
   - 本地计算 $\mathbf{H}_i\mathbf{v}_i$
   - All-reduce聚合结果
   
3. 通信优化：
   - 重叠计算与通信
   - 使用梯度压缩技术
   - 批量处理多个向量

关键挑战：
- 保持数值正交性
- 负载均衡
- 容错性设计
</details>

**练习 2.8**：理论分析题
证明：对于强凸二次函数，使用精确线搜索的BFGS方法在 $n$ 步内收敛到精确解。

*提示*：利用共轭方向的性质。

<details>
<summary>答案</summary>

证明概要：
1. 对二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} - \mathbf{b}^T\mathbf{x}$
2. BFGS生成共轭方向：$\mathbf{s}_i^T\mathbf{A}\mathbf{s}_j = 0$ for $i \neq j$
3. 共轭方向张成整个空间需最多 $n$ 个
4. 因此 $n$ 步后 $\mathbf{H}_n = \mathbf{A}^{-1}$（精确）
5. 下一步即得到精确解

关键：BFGS保持了"遗传性质"。
</details>

## 常见陷阱与错误 (Gotchas)

### 1. L-BFGS更新的数值问题
- **陷阱**：当 $\mathbf{y}_k^T\mathbf{s}_k \approx 0$ 时直接更新
- **后果**：数值不稳定，甚至除零错误
- **解决**：检查 $\mathbf{y}_k^T\mathbf{s}_k > \epsilon\|\mathbf{y}_k\|\|\mathbf{s}_k\|$，否则跳过更新

### 2. Hessian-向量乘积的内存泄漏
- **陷阱**：在自动微分中保留计算图
- **后果**：内存快速耗尽
- **解决**：使用`detach()`或`stop_gradient`及时释放

### 3. 负曲率检测的时机
- **陷阱**：每步都进行Lanczos迭代
- **后果**：计算成本过高
- **解决**：自适应检测策略，基于收敛指标

### 4. Trust Region半径的初始化
- **陷阱**：固定初始半径
- **后果**：对不同规模问题适应性差
- **解决**：基于梯度范数自适应：$\Delta_0 = \|\mathbf{g}_0\|$

### 5. 预条件子的更新频率
- **陷阱**：每步重新计算预条件子
- **后果**：开销超过收益
- **解决**：定期更新或基于条件数变化

### 6. 混合精度的数值陷阱
- **陷阱**：关键量使用低精度
- **后果**：累积误差导致发散
- **解决**：识别数值敏感操作，选择性使用高精度

## 最佳实践检查清单

### 算法设计阶段
- [ ] 选择合适的Hessian近似方法（内存 vs 精度权衡）
- [ ] 确定是否需要负曲率信息（问题是否非凸）
- [ ] 设计数值稳定性保护措施
- [ ] 考虑并行化和分布式需求

### 实现阶段
- [ ] 实现数值检查（条件数、正定性）
- [ ] 添加异常处理（除零、溢出）
- [ ] 使用高效的线性代数库
- [ ] 实现诊断和日志功能

### 调参阶段
- [ ] L-BFGS历史长度 $m$：从5开始，根据内存调整
- [ ] Trust Region初始半径：$\Delta_0 = \min(1, \|\mathbf{g}_0\|)$
- [ ] 线搜索参数：$c_1 = 10^{-4}$, $c_2 = 0.9$（Wolfe条件）
- [ ] 数值容差：相对误差 $10^{-8}$，绝对误差 $10^{-12}$

### 性能优化
- [ ] Profile找出计算瓶颈
- [ ] 考虑向量化和批处理
- [ ] 优化内存访问模式
- [ ] 实现checkpoint策略

### 鲁棒性测试
- [ ] 测试病态问题（高条件数）
- [ ] 测试随机初始化
- [ ] 验证收敛性保证
- [ ] 压力测试（大规模、长时间运行）

## 深入研究方向

1. **随机二阶方法**：如何在随机梯度setting下有效利用二阶信息
2. **隐式正则化**：L-BFGS等方法的隐式偏好如何影响解的质量
3. **硬件协同设计**：针对特定硬件（GPU/TPU）优化的Hessian近似
4. **在线学习场景**：流数据下的增量Hessian更新策略
