# 出处与知识产权

本项目为 **GPL-3.0-or-later** 全量开源，`LICENSE` 是权威原文。

**功能形式与权重复现自公开发表的方程，代码为独立实现。** 下表列出每一项的来源
与许可；上游作者姓名同样保留在对应 Rust 模块的 `Provenance` 文档段里。

| 借鉴内容 | 来源 | 许可 |
|---|---|---|
| 打分函数形式与权重（`gauss1` / `gauss2` / `repulsion` / `hydrophobic` / `hydrogen-bond`） | AutoDock Vina 1.2 — Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010) | Apache-2.0 |
| Vinardo 权重 | Quiroga & Villarreal, *PLoS ONE* **11**, e0163579 (2016) | CC-BY |
| 预制网格定义、原子类型与可旋转键规则 | AutoDock 4.2 — Morris et al., *J. Comput. Chem.* **29**, 2789 (2008) | GPL-2.0 |
| PDBQT 列格式与准备规则 | Meeko — Morris et al., *PLoS ONE* **17**, e0163573 (2022) | LGPL-2.1 |
| 岛式 MFFGA 搜索策略 | AutoDock-GPU — Morris et al., *JCAMD* **25**, 10 (2011) | LGPL-2.1 |
| L-BFGS 拟牛顿优化 | Byrd, Nocedal & Schnabel, *SIAM J. Optim.* **16**, 1182 (1994) | 自由 |
| SO(3) 指数映射与右 Jacobian | Barfoot, *State Estimation for Robotics* (2017) | 自由 |
| 数值方法的一般性参考 | *Numerical Recipes*（三线性插值、模拟退火） | 版权保留 |

## clean-room 声明

**没有**阅读或借用 AutoDockTools / MGLTools 的专有代码。
PDBQT 的读写两侧均为 clean-room 实现，依据的是公开发布的列格式规范，
见 [`ARCHITECTURE.md`](ARCHITECTURE.md) 关于文件格式的说明。

## 本项目自行推导、未经交叉验证的部分

以下是**本项目自己的决定**，不是从任何上游抄来的：

- 逐元素的 XS 相互作用半径（C 1.9 / N 1.75 / O 1.6 / S 2.0 Å …），
  按晶体学接触距离独立推导。**未与 AutoDock Vina 交叉验证**，
  所以绝对 kcal/mol 不可与文献比较，见 [`LIMITATIONS.md`](LIMITATIONS.md)。
- 4-slot 网格（把 donor 与 acceptor 拆成两张图），
  用于消除单一 `e_hb` 图产生的 donor–donor 伪氢键。
- 碰撞分区在聚类**之前**执行，以及 `rejected_pose_count` 的显式上报。
- 解析一阶梯度的三处实现，及其相对“存储径向导数”这一常见做法的偏离，
  推导见 [`SCORING.md`](SCORING.md)。

## 分发义务

由于本项目分发 GPL-3.0 代码，**任何二进制分发都必须同样以 GPL-3.0 提供完整源码**，
包括被链接进你产品的部分。

如果你的贡献派生自其他作品，请在文件头写明来源与许可，
并在本文件补一行——本文件是唯一的总账。
