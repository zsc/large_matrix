# 高级大规模矩阵计算教程

## 前言

本教程面向熟悉基础矩阵计算的研究者和AI科学家，深入探讨大规模矩阵计算中的高级主题。我们不仅关注经典算法，更强调在现代AI应用中尚未充分研究的数学问题。每章包含理论分析、实践技巧、常见陷阱以及大量习题。

## 目录

### 第一部分：二阶优化方法基础

#### [第1章：二阶优化的统一框架](chapter1.md)
- Newton法、Gauss-Newton法与Natural Gradient的数学联系
- Fisher信息矩阵与Hessian的关系
- Trust Region方法在深度学习中的复兴
- 鞍点逃逸的理论与实践

#### [第2章：Hessian近似的艺术](chapter2.md)
- 从BFGS到L-BFGS：有限内存方法的深入剖析
- Hessian-vector product的高效计算
- 负曲率方向的检测与利用
- 数值稳定性与条件数控制

#### [第3章：结构化二阶方法](chapter3.md)
- Kronecker因子分解：K-FAC及其变体
- Block对角近似：Shampoo算法解析
- 低秩加对角结构的利用
- 稀疏Hessian模式的自动发现

### 第二部分：增量与在线算法

#### [第4章：增量Hessian计算](chapter4.md)
- Woodbury矩阵恒等式的高级应用
- Block-wise更新策略
- Sliding window技术
- 与在线凸优化的深度联系

#### [第5章：Schur补的妙用](chapter5.md)
- 分块矩阵求逆的递归策略
- 在分布式优化中的应用
- 条件数改善技术
- 与域分解方法的联系

### 第三部分：随机化方法

#### [第6章：矩阵Sketching技术](chapter6.md)
- Johnson-Lindenstrauss引理的实用化
- CountSketch与随机投影
- Frequent Directions算法
- 在神经网络压缩中的应用

#### [第7章：随机化数值线性代数](chapter7.md)
- 随机SVD的误差分析
- Nyström方法的现代视角
- 随机化预条件子设计
- 量子启发的采样策略

### 第四部分：分布式与并行计算

#### [第8章：分布式矩阵运算](chapter8.md)
- 通信高效的矩阵分解
- Gossip算法的收敛性分析
- 异步更新的一致性保证
- 拜占庭鲁棒性设计

#### [第9章：异步优化的数学基础](chapter9.md)
- 延迟梯度的误差累积分析
- Lock-free算法设计
- 局部一致性与全局收敛
- 硬件感知的算法调优

### 第五部分：流形优化

#### [第10章：Riemannian优化基础](chapter10.md)
- 矩阵流形上的几何结构
- Riemannian梯度与Hessian
- 回缩与向量传输
- 在低秩矩阵补全中的应用

#### [第11章：流形预条件技术](chapter11.md)
- 流形上的Natural Gradient
- Riemannian BFGS方法
- 几何感知的Trust Region
- 与欧氏空间方法的性能对比

### 第六部分：特殊结构利用

#### [第12章：结构化矩阵的快速算法](chapter12.md)
- Toeplitz与循环矩阵的FFT技巧
- Kronecker积的高效运算
- 分层矩阵（H-matrices）
- 在卷积网络中的应用

#### [第13章：动态低秩近似](chapter13.md)
- 流式SVD更新
- 自适应秩选择
- 在线矩阵补全
- 与神经网络剪枝的联系

### 第七部分：推荐系统中的矩阵计算

#### [第14章：大规模协同过滤的矩阵技术](chapter14.md)
- 隐式反馈矩阵分解的加权策略
- ALS-WR算法的并行化与数值优化
- 负采样的数学原理与偏差校正
- 置信度加权的理论基础

#### [第15章：实时推荐的增量矩阵方法](chapter15.md)
- 在线矩阵分解的遗忘机制
- 用户/物品嵌入的快速更新
- 冷启动问题的矩阵补全视角
- 时序动态的矩阵建模

#### [第16章：多模态推荐的张量分解](chapter16.md)
- 高阶交互的张量建模
- CP分解与Tucker分解的可扩展实现
- 稀疏张量的高效存储与计算
- 跨域推荐的耦合矩阵分解

### 第八部分：前沿主题

#### [第17章：隐式微分与双层优化](chapter17.md)
- 大规模线性系统的隐式求解
- 自动微分的高级技巧
- 在元学习中的应用
- 数值稳定性挑战

#### [第18章：量子启发的矩阵算法](chapter18.md)
- 张量网络方法
- 量子奇异值变换
- 经典模拟的计算复杂度
- 在机器学习中的潜力

## 附录

### [附录A：数值稳定性速查表](appendixA.md)
### [附录B：性能调优检查清单](appendixB.md)
### [附录C：常用矩阵恒等式](appendixC.md)

## 使用说明

- 每章独立成文，可按需阅读
- 习题答案默认折叠，鼓励独立思考
- "Gotchas"部分总结实践中的常见陷阱
- "研究方向"指出值得深入探索的开放问题

## 符号约定

- $\mathbf{A}, \mathbf{B}$：矩阵
- $\mathbf{x}, \mathbf{y}$：向量
- $\lambda_i$：特征值
- $\kappa(\mathbf{A})$：条件数
- $\mathcal{O}(\cdot)$：计算复杂度
- $\mathbb{E}[\cdot]$：期望
- $\|\cdot\|_F$：Frobenius范数
- $\otimes$：Kronecker积
- $\odot$：Hadamard积
