（交流可以用英文，本文档中文，保留这句）

# 高级大规模矩阵计算教程项目说明

## 项目目标
编写一份大规模矩阵计算的高级教程markdown，要包含大量的习题和参考答案（答案默认折叠）。合适时提及相关函数名但不写代码。
覆盖话题包括preconditioning, optimization on manifold, matrix sketching, distributed matrix computation, online incremental algorithm 等：覆盖技术有incremental hessian computation, Schur complement, L-BFGS 深入解析等
项目特色是，包含大量的可继续研究的线索
文件组织是 index.md + chapter1.md + ...

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **本章小结**：总结关键概念和公式
3. **练习题**：
   - 每章包含6-8道练习题
   - 50%基础题（帮助熟悉材料）
   - 50%挑战题（包括开放性思考题）
   - 每题提供提示（Hint）
   - 答案默认折叠，不包含代码
4. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧
5. **最佳实践检查清单**：每章末尾提供设计审查要点

