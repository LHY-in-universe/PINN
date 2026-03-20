# Physics-Informed Neural Networks (PINNs) 极简教程与实战

本项目包含了几个从零开始实现的物理信息神经网络（PINN）经典示例，完全基于纯 PyTorch 实现，并在 Jupyter Notebook 中分步讲解。如果你想快速上手 PINN 求解偏微分方程 (PDE) 甚至发现物理规律，这些代码可以作为非常好的起步模板。

---

## 什么是 PINN？(核心深度解析)

在传统的数值分析（如有限元 FEM 或有限差分 FDM）中，求解偏微分方程 (PDE) 必须将物理空间离散化为复杂的网格（Mesh）。而在 **PINN (Physics-Informed Neural Networks)** 范式中，我们彻底抛弃了网格，将物理定律直接编码进神经网络的 DNA 中。

它的核心数学逻辑由以下三根支柱支撑：

### 1. 神经网络作为通用函数逼近器
神经网络 $u_\theta(\mathbf{x}, t)$ 本质上是一个参数化的连续函数。与那些依赖离散节点插值的网格方法不同，它在整个时空定义域内是**处处可导**且**连续**的。这意味着无论你在计算域的哪个坐标位置，神经网络都能给出一个平滑的物理场预测。

### 2. 自动微分 (Automatic Differentiation, AD) —— PINN 的秘密武器
这是 PINN 优于传统方法的最关键技术。
*   **传统方法**：依赖有限差分近似导数（如 $\Delta u / \Delta x$），这会受到网格步长 $h$ 的限制，且阶数越高（如二阶导、四阶导）误差累积就越严重。
*   **PINN (AD)**：利用深度学习框架内建的 `autograd` 机制。它不是求导数值的近似，而是根据链式法则对神经网络的计算树进行符号层面的精确操作。这使得我们能以**机器精度**计算出物理量相对于输入坐标的任意阶空间或时间导数，完全消除了网格截断误差。

### 3. 物理残差作为损失函数 (Loss Function)
我们将 PDE 转化为了一个**优化问题**。
1. **方程残差 (PDE Loss)**：在计算域内撒下“配点”（Collocation Points），要求这些点上的数学关系必须成立。如果网络预测的导数制衡不守恒，产生的残差平方和即为 $Loss_{PDE}$。
2. **约束补偿 (BC/IC Loss)**：在边界上撒点，强行要求网络输出等于真实物理条件（如初值或墙壁约束）。
3. **正则化 (Norm Loss)**：在量子力学等问题中，引入归一化约束，防止神经网络通过输出“全零”的平庸解来逃避物理惩罚。

当总损失下降到一定阈值时，神经网络的非线性函数叠加就自然“生长”成了符合该物理偏微分规律的最优数值解曲线。

---

## 仓库实战案例说明

仓库内包含了从浅入深、循序渐进的五个实战模板：

### 1. 简谐运动的常微分方程求解 [`simple_pinn.ipynb`]
基础入门。带您第一次了解如何把方程残差拼装成 PDE Loss，画出与理论解析解完美重构的纯物理图表。

### 2. 盒子里的量子 - 薛定谔系统求解 [`quantum_pinn.ipynb`]
进阶量子力学问题。详细探讨了 PINN **如何利用引入先验条件（Peak Loss 或归一化条件）打破 0 解平衡**，自动定位基态波函数的形态。

### 3. 流体力学之痛 - Burgers' 方程激波捕捉 [`no_exact_pinn.ipynb`]
经典难题。对比流体力学高精度差分法产生的真实结果（`burgers_shock.mat`），验证 PINN 在稀疏采样下捕获**非线性激波断层 (Shock Front)** 的非凡能力。

### 4. 固体物理基石 - Kronig-Penney (KP) 模型 [`kp_pinn.ipynb`]
通过 **周期性特征编码 (Periodic Feature Mapping)** 在架构层面上强制保证周期物理对称性。展示了如何通过数值解色散方程找到精确能级并实现像素级拟合。

### 5. 原子世界的推演 - 氢原子轨道求解 [`hydrogen_pinn.ipynb`] & [`hydrogen_multistate_pinn.ipynb`]
处理库仑势的 **$1/r$ 空间奇异性**。通过引入正交化 Loss 约束，成功让同一个神经网络自动分离出 1s, 2s, 3s 等不同的量子能级轨道。

---

## 快速上手

1. **环境准备**：
   ```bash
   pip install torch numpy matplotlib scipy
   ```
2. **运行实验**：
   使用 Jupyter Notebook 或 VS Code 打开对应的 `.ipynb` 文件运行即可。

## 参考资料
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*.
