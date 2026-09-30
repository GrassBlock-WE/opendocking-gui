# 验证报告

本文如实记录**跑了什么、结果如何、以及什么没有验证**。
不要跳过第二、三节。在那些项目上，本工具只是一个能编译、能跑的程序，
不是一个被证明过能给出正确对接姿势的程序。

---

## 1. 已验证

### 1.1 Rust 单元测试

| 命令 | 结果 |
|---|---|
| `cargo test -p dock-core` | **106 passed; 0 failed** + 1 doctest |
| `cargo test -p dock-core --features gpu` | **116 passed; 0 failed** + 1 doctest |
| `cargo clippy --workspace --all-targets -- -D warnings` | **exit 0** |
| `cargo clippy -p dock-core --features gpu --all-targets -- -D warnings` | **exit 0** |
| `cargo build --workspace --all-targets`（含 bench） | 零警告零错误 |
| `cargo fmt --all --check` | **exit 0**（原先不通过，见 §3.12） |

### 1.2 GPU（实机，wheel 全链路）

不只是"能编译"，而是从 Python 调通的完整链路：

| 项 | 结果 |
|---|---|
| `maturin build --features gpu` | 成功产出 wheel |
| `odcli info` | `gpu: compiled in and an adapter is available` |
| `gpu_status()` | `{"compiled": True, "available": True}` |
| 实际适配器 | `NVIDIA GeForce RTX 3050 Laptop GPU` |
| 真实蛋白网格，5 / 1024 构体 GPU vs CPU 最大差 | **2.220e-06**（两种规模相同） |
| Rust 侧逐原子 GPU vs CPU | < 1e-3（`gpu_energies_match_the_cpu_interpolation`） |

GPU 后端**已经接入批量打分路径**（`evaluate_conformations(use_gpu=True)`），
不是只在测试里存在的死代码。回退原因会通过 `report_backend` 显式返回。

`use_gpu` **默认 False**：GPU 端用 `f32` 累加，与 CPU 差 ~1e-6，
让默认值随硬件变化会破坏跨机可复现性。

### 1.3 Python

| 项 | 结果 |
|---|---|
| `python -m pip install -r requirements.txt` | 依赖清单按运行时 / 构建 / GUI / 开发分组，附实测版本 |
| `python -m pytest --pyargs opendocking.tests -q` | **38 passed**（对**已安装的 wheel**运行） |
| CPU wheel | `dist/opendocking-0.1.0-cp38-abi3-win_amd64.whl` |
| GPU wheel | `dist-gpu/opendocking-0.1.0-cp38-abi3-win_amd64.whl` |

> 注意 `--pyargs` 跑的是**已安装的包**。改了 `dock-py/python/` 下的 `.py`
> 必须重新 `maturin build` + `pip install`，否则测试跑的是旧代码。踩过一次。

### 1.4 真实蛋白（crambin 1CRN，RCSB 实验结构）

| 项 | 结果 |
|---|---|
| 受体原子 | 327 重原子 + 58 极性氢 = 385 |
| 供体位点覆盖率 | **55 / 55**（`check_receptor_donors.py`，exit 0） |
| 重原子位移 | **0**（受体保持刚性） |
| 配体 | RCSB 配体库生物素，19 原子 / 5 扭转 / 11 自由度 |
| 生物素化学审计 | 逐原子核对通过（羧基 `OA`+HD、两个脲基氮 `N` 即酰胺氮不作受体、噻吩硫 `S`） |
| 网格 | 18 Å 盒 @ 0.375 Å |
| CPU/GPU 能量一致性（真实蛋白网格） | **2.220e-06** |
| 返回姿势的物理检查（`audit_poses.py`，exit 0） | 见 §1.4.1 |

#### 1.4.1 姿势的物理审计

`python examples/audit_poses.py` → **exit 0**，5 个姿势全部通过：

```
pose 1: E  -5.222  contacts 19/19  min-dist 2.30 A  clash 0  |grad| 3.13  downhill +0.00e+00  face-gap 0.0000
pose 2: E  -5.042  contacts 15/19  min-dist 2.32 A  clash 0  |grad| 5.91  downhill +0.00e+00  face-gap 0.0000
pose 3: E  -4.836  contacts 15/19  min-dist 2.19 A  clash 0  |grad| 4.91  downhill +0.00e+00  face-gap 0.0000
pose 4: E  -4.618  contacts 17/19  min-dist 2.38 A  clash 0  |grad| 2.35  downhill +0.00e+00  face-gap 0.0000
pose 5: E  -4.571  contacts 18/19  min-dist 2.32 A  clash 0  |grad| 4.86  downhill +0.00e+00  face-gap 0.0000
```

* **0 个碰撞姿势**，最小重原子间距 2.19–2.38 Å（正常非键接触）。
* **全部是真正的局部极小点**：`downhill = 0.00e+00`，即沿 −∇E 任何步长都走不动能量。

**`face-gap = 0.0000` 这一列是本轮最重要的测量。** 每个返回姿势都**恰好有一个配体
原子坐在网格 cell 面上**，于是该点存在两个大小相近、符号相反的单边梯度
（`tx` 解析 +0.76 / 数值 −0.50，`tau0` 解析 −3.13 / 数值 +2.17），
而函数在那里取到极小值。

**所以 `|grad|` 大不是没收敛，是方法的固有性质**（三线性插值只有 C⁰）。
判断极小性必须用线搜索，不能用梯度模长。

**解析梯度本身是验证过的**，不是"看起来可疑"：

| 检查 | 结果 |
|---|---|
| 10 组随机在盒内构象 × 11 个 DOF，解析 vs 中心差分 | **全部 `err = 0.0000`** |
| Rust `the_analytic_gradient_is_the_gradient_of_the_reported_energy`（butane 1 扭转 + hexane 4 扭转） | 通过 |

> ⚠️ 顺带更正本报告早期版本的一句话：曾写"返回姿势的解析梯度范数 ~1e-1，
> L-BFGS 确实收敛"。那个数字来自合成冠醚受体的一次运行，
> 在真实蛋白上是 2.3–5.9，且与中心差分符号相反。**当时的解读是错的。**

### 1.5 workbench（真实 Qt 窗口 + 真实 OpenGL）

`python scripts/workbench_smoke.py` → **exit 0**，两个层级都通过：

1. **真实 Qt 窗口**（默认平台，非 offscreen）：

   `grabFramebuffer()` 读回 **800×771，542 种颜色，154400 个非黑样本**，
   输出 `dist/workbench_qt_widget.png`。
   **该图已目视确认**：搜索盒线框（黄）、30 原子冠醚受体环、16 原子布洛芬姿势，
   CPK 配色与深度顺序正确。

2. **独立 moderngl 上下文**：两个 shader 编译通过，
   FBO 读回 524 色 / 2027 非背景像素，输出 `dist/workbench_render.png`。

`python scripts/qt_gl_probe.py` 另外测量了 6 条取 GL 上下文的路径，
结果记录在 [`LIMITATIONS.md` §2](LIMITATIONS.md)。

> **这些只验证了"渲染层能画出东西"。** 加载、交互、控件是否生效，
> 属于 §3.7 / §3.8——那里发现了一批上一轮完全没看见的 bug。

### 1.6 鲁棒性

`python examples/robustness_check.py` → **exit 0**，全部用例给出可捕获的异常，
**没有任何一个子进程异常死亡**。release profile 是 `panic = "abort"`，
所以所有 Rust panic 不会变成 Python 异常，而是直接杀掉宿主解释器——
脚本因此用子进程监视退出码（`0xC0000409` / `SIGABRT`）。

覆盖：空文件、只有 REMARK、截断的原子行、两原子重叠、未知原子类型、
倒置盒子、零尺寸盒子、NaN/inf 角、负间距、荒谬间距、未知打分函数、
构象维度错误、NaN 构象、索引越界、盒子装不下配体、1e6 量级数值不产生非有限值。

### 1.7 确定性与并行

`python examples/determinism_check.py` → **exit 0**：

- 两次制表**逐字节相同**（说明并行切片没有交叉写入）
- 固定 seed 的两次对接：能量**完全为 0 的差异**、坐标**完全相同**、pose 数相同
- 不同 seed 给出不同区域（质心相差 5.8 Å）
- exhaustiveness 提高时能量单调变好（−4.6463 → −5.0242）

### 1.8 CLI

全部 7 个子命令实跑通过，含目录批量与 `--json`。
碰撞计数已进 JSON（`rejected_pose_count`）与终端摘要（`WARNING:` 行）。

CLI 的命令名是 `opendocking`（`opendocking` 子命令），`odgui` 是另一个独立入口点。

注意 CLI 的 box 参数是**下划线**：`--center_x`，不是 `--center-x`。

---

## 2. 未验证 / 未完成

**这一节是本文的重点。**

### 2.1 与 AutoDock Vina 的数值比对 —— 未做

**没有**用 Vina 跑同一个配体/受体再比对 pose 与能量。
权重按公开值实现、梯度有有限差分锁定，但"实现是对的"和
"算出来的姿势与 Vina 一样"是两件事。

**这在本轮变得更重要**：本轮修掉的相互作用半径缺陷（§3.2.1）用的
是**按参考非键接触距离取的每元素 XS 半径**，不是从 Vina 抄来的值。
它们让每一项落在物理上正确的距离（这一点由晶体学接触距离独立验证），
但**具体常数和 Vina 用的是否逐位相同，无从确认**。

因此：**本工具输出的绝对 kcal/mol 数字不应与 Vina 输出或文献值直接比较。**
相对排序、姿势质量、几何合理性可以参考。

**未能完成的原因（已核实，不是没试）**：`pip install vina` 失败——
本机 pip 被指向一个已失效的本地代理 `127.0.0.1:7890`，
绕过代理后镜像同样不可达。Python 与 PowerShell 的出网都被拦截，
只有工具层 `web_fetch` 可用，而它无法安装软件包。

### 2.2 回对接（redocking）—— 未做

这是对接质量的金标准：把配体对进它在晶体中的口袋，看能否复现晶体姿势。

**未能完成的原因（已核实）**：拿到了 1STP（streptavidin + 生物素）的完整
配体（16 重原子 + CONECT 键表），但工具层 `web_fetch` 对超过约 16000 token
的响应会**截断中间内容**，而 1STP 的蛋白部分（901 原子）恰好落在被截断的区间。
实际取到的蛋白片段是残基 96–133，距离生物素最近 12 Å——
**口袋壁全在缺失的残基里**，用这个片段做回对接没有意义。

crambin 没有结合口袋，所以它也不能替代一个 holo 结构。

### 2.3 平台与硬件覆盖

只在 Windows 11 x86_64 + NVIDIA RTX 3050 Laptop 上验证过。
macOS（Metal）、Linux（Vulkan）、AMD 独显、Intel 核显、Apple Silicon **未测**。

### 2.4 GUI 交互 —— 本轮之前从未验证

上一轮只验证了渲染层。拖拽、缩放、平移、box 工具、pose browser 控件、
实时打分面板刷新，**一次都没有被操作过**。

**本轮已补上**，见 §3.7 / §3.8。仍然没做的是：多显示器 / 高 DPI 混合缩放、
触屏与三键鼠标以外的输入设备、窗口在后台时长时间运行的稳定性。

### 2.5 网格规模

只在 10–26 Å 的盒子上测过。`MAX_GRID_POINTS`（2^28 点，约 40 GiB）
以上的拒绝路径只测了参数错误，**没有**真的跑过那种规模的合法计算。
也没有大盒子的吞吐量数据。

**踩过的天花板**：18 Å 盒 @ 0.03125 Å 需要 30.7 GB，实测
`memory allocation of 30736005280 bytes failed` —— 一个干净的分配失败，
不是崩溃。要做精细的梯度收敛研究必须缩小盒子或接受 f32 量级的量化。

---

## 3. 验证过程中发现并修复的 bug

以下每一项都是**跑出来**的，不是读代码想出来的。
共 78 条，按性质分组。

### 3.1 正确性

| # | 现象 | 根因 |
|---|---|---|
| 1 | 扭转角不是从 0 累加的 | 初值写成 1.0 |
| 2 | 键感知用了全局距离阈值 | 改成逐原子共价半径判据 |
| 3 | PDBQT 的 `ROOT`/`ENDROOT`/`BRANCH` 记录被当成原子 | 解析器不过滤这些记录 |
| 4 | `TORSDOF` 与实际扭转数不一致 | 按 `BRANCH` 树数而不是按 `TORSDOF` |
| 5 | 芳香性没有从 PDBQT 环里恢复 | 环原子没跑芳香性判定 |
| 6 | `NA` 氮被标成疏水 | 邻居统计没考虑脲基/酰胺 |
| 7 | 羧基氧被标成疏水 | 羧基两个氧都算进邻居判定 |
| 8 | 极性氢缺失时所有羟基氧都不是供体 | 只看 `GetTotalNumHs()` |
| 9 | 原子顺序影响分类结果 | 用了顺序相关的邻居遍历 |
| 10 | PDBQT 类型 `NDA` / `ODA`（3 字符）无法识别 | 类型表只收 2 字符；改为 `NA` / `OA` |
| 11 | RDKit 2026 移除了 `Atom.GetDoubleBondType()` | 改用 `mol.GetBondBetweenAtoms(...).GetBondType()` |
| 12 | 带氢的酰胺氮永远走不到酰胺判断 | 判断顺序在氢判断之后 |
| 13 | 晶体结构（无氢）导致 **0 个供体** | 新增 `_RECEPTOR_DONOR_SITES` + 显式加极性氢 |
| 14 | 网格 stride 与 WGSL 不一致 | 两处各写各的；加一致性单测 |
| 15 | 原子梯度没包含权重导数项 | 三线性插值导数漏项 |
| 16 | workgroup size 在 Rust 与 WGSL 各写一份 | 统一到一处并加单测 |

### 3.2 能量函数层面

这一节的**诊断最后被推翻了**，根因在 §3.2.1。保留原过程是因为它记录了
"症状指向错误方向"这件事本身。

| # | 现象 | 当时的结论 |
|---|---|---|
| 17 | 把配体对进 crambin 疏水核心，所有 pose 每个原子都在受体 2 Å 内，最近 0.17–0.71 Å | 见下 |
| 18 | 同样的大盒子、配体可待在表面时，**仍然**全部 clash | 不是"盒子选错"这么简单 |
| 19 | 500 次随机采样：最佳**无冲突**放置 E = −0.010，最佳**有冲突**放置 E = −1.001 | "打分函数本身偏好埋藏" |

**当时的修复**：在聚类**之前**把物理上不可能的姿势分离出去，
优先报告可成立的姿势；一个都找不到时才回退，并**通过
`DockingResult.rejected_pose_count` 与 CLI 的 `WARNING:` 显式告知**。
硬拒绝会把"搜索偏弱"变成"对接失败"，所以没有采用。

**这个修复的副作用，必须说清楚**：碰撞过滤**破坏**了
"提高 exhaustiveness 不会变差"的单调性——当弱搜索只找到碰撞姿势时，
它会回退并报告一个更差的（但诚实的）数字。
`test_higher_exhaustiveness_does_not_worsen` 因此改为在前提成立时断言该性质。

### 3.2.1 根因：所有重原子的相互作用半径是 0.4 Å

**§3.2 的"没有有效排斥"是一个症状，不是原因。** 真因是
`Element::interaction_radius()` 对**所有**重原子返回同一个 `0.4 Å`。
0.4 是共价键长，不是相互作用半径。表面距离是 `d = r − (Rᵢ + Rⱼ)`，
所以 `R` 直接决定每一项落在间距轴的哪个位置：

| 项 | 需要的 `d` | 换算成真实间距 | 合理？ |
|---|---|---|---|
| `hb` | `≤ −0.5` | **0.3 Å**（O···O） | 否 |
| `g1` | `= 0.5` | 1.3 Å（C···C） | 否 |
| `g2` | `= 0` | 0.8 Å（C···C） | 否 |
| `hyd` | `≤ 0.5` | 1.3 Å（C···C） | 否 |

两个氧原子相距 0.3 Å 不是氢键，是一个原子和它自己的镜像。
**权重 −0.587439（Vina 函数里最大的单项）在任何真实氢键能形成的距离上恰好为零。**

**怎么发现的**：冠醚 + 酚的氢键几何明明摆对了（2.0 Å 距离，有单调信号），
但所有配体的能量都是 **−0.000**。把酚羟基从 `OA` 改成 `A` 做对照，
氢键项的贡献只有 **−0.0032 kcal/mol**——应该约 −0.18，差了 35 倍。

**修法**：改用每元素的 XS 半径（C 1.9 / N 1.75 / O 1.6 / S 2.0 Å，H 为 0）。
同时修掉 `hbond_term` 的窗口端点（曾整体偏近 0.5 Å，最大值落在 2.5 Å）。

| 对照 | 修前 | 修后 |
|---|---|---|
| O···O 供受体氢键最小值 | 2.0 Å 处 −0.005（=没有） | **2.50 Å 处 −0.177** |
| C···C 疏水最小值 | 2.0 Å 处 −0.005 | **3.00 Å 处 −0.050** |
| 冠醚 + 苯 | E = −0.00 | **E = −2.05**，最小间距 2.88 Å |
| 冠醚 + 布洛芬 | E = −1.83，最小间距 **0.13 Å**（全 16 原子重叠） | **E = −5.03**，最小间距 **2.64 Å** |
| crambin + 生物素 | E = −2.20，最小间距 **0.54 Å** | **E = −5.04**，最小间距 **2.24 Å** |
| 500 次随机采样的最佳放置 | 无冲突 −0.010 vs 有冲突 −1.001（差 0.99） | **两者都是 −0.815，无冲突占优（差 0.000）** |
| 实心碳晶格 + 苯 | 埋进去是**最便宜**的 | **+289.9 kcal/mol** |

**连带修掉的一个潜伏 bug**：预制表内循环拿**真实距离** `r2` 去做**表面距离**阈值
`cutoff` 的比较，等于把真实截断设成了 `cutoff` 而不是 `cutoff + 2·R`，
高斯项的尾巴一直被削掉一截（0.8 Å → 3.8 Å）。这类错误不会让任何测试变红。

**连带发现并重写的一个坏测试**：`grid::tests::interpolation_is_linear_along_one_axis`
在一个**非线性**的场（高斯 + 排斥）上断言三线性插值是线性的。
它过去只是因为场太平而碰巧通过——修好半径后场变陡，它立刻变红。
现在改成对比**存储的节点值**，测的是插值算子本身，与场无关。

### 3.2.2 一个看起来像梯度 bug、其实不是的现象

生物素对进 crambin 后，返回姿势的 `|grad|` 是 2.3–5.9，
且**逐分量与中心差分符号相反**：

```
dof     analytic    numeric
tx       0.7632    -0.4956
ry       2.2074    -1.4849
tau0    -3.1266    +2.1669
```

第一反应是"扭转梯度算错了"。**不是。** 测姿势到最近 cell 面的距离：

```
pose 1: closest approach to a cell face = 0.0000 cells
```

**每个返回姿势都恰好有一个配体原子坐在网格面上**，于是该点存在两个大小相近、
符号相反的单边梯度，而函数在那里取到极小值（沿两个方向都走不动能量：
起点 −5.222，解析方向最好 −5.209，数值方向 −5.215）。

**解析梯度本身是对的**，两条独立证据：

| 检查 | 结果 |
|---|---|
| 10 组随机在盒内构象 × 11 DOF，解析 vs 中心差分 | 全部 `err = 0.0000` |
| 新增 Rust 测试（butane 1 扭转 + hexane 4 扭转，全 DOF） | 通过 |

**真正缺的是测试覆盖**：`kinematics.rs` 的 `conf_gradient` 有限差分测试喂的是
坐标的线性泛函 `E = Σ gᵢ·rᵢ`，只能证明回代代数对，
证明不了打分管线产出的**原子梯度**对。现已补上
`search::tests::the_analytic_gradient_is_the_gradient_of_the_reported_energy`。

**方法论教训**：沿 −∇E 的线搜索必须**先归一化方向**。
用原始 `∇` 走 `k·∇`，在 `|∇| = 5.9` 时步长 0.05 就把原子挪 0.3 Å、步长 3.2 挪 19 弧度，
中间那段有用的范围根本没采到，于是会误判成"已经是极小点"。

### 3.3 会杀死宿主进程的 bug

| # | 现象 | 根因 |
|---|---|---|
| 20 | `precalculate(spacing=1e-9)` **直接杀掉 Python 解释器**（退出码 3221226505） | 间距检查只挡 (0,1]，1e-9 通过；维度乘积溢出，`vec!` capacity overflow abort；release 是 `panic = "abort"` |
| 21 | 负间距被静默替换成默认间距 | `spacing <= 0.0` 一律回退，打错字会悄悄换成另一张网格 |
| 22 | `inf` 角点的盒子被接受 | `!(max > min)` 对 ±inf 成立 |
| 23 | 构象里的 NaN 静默返回惩罚值 | 没有非有限值校验 |
| 24 | 未知原子类型静默降级 | 没有可见的计数 |

现在 `MAX_GRID_POINTS`（2^28 点）在**分配之前**拒绝，并报出所需 MB 数；
负间距与 NaN 报错；`inf` 角点报错；构象非有限值报错。
`examples/robustness_check.py` 用子进程监视退出码，防止这类问题回归。

### 3.4 GUI 的严重 bug

| # | 现象 | 根因 |
|---|---|---|
| 25 | **workbench 视口在真实桌面上是空白的** | `QOpenGLWidget` 从 **FBO** 合成，而 `paintGL` 画到**默认帧缓冲**（`ctx.screen`）上，画的地方根本不会被显示 |
| 26 | 视口尺寸用 `self.width()` | 设备像素比 1.25，`target.size` 才是要画的区域 |
| 27 | 视口尺寸用的是控件尺寸而非 framebuffer 尺寸 | 同上 |

这个 bug 之前完全没被发现，原因是冒烟脚本**强制设置了
`QT_QPA_PLATFORM=offscreen`**，而 offscreen 恰恰是唯一不支持 OpenGL 的模式——
"拿不到上下文"看起来像渲染器坏了，其实是环境问题。
去掉强制设置后，默认平台能正常拿到上下文（`scripts/qt_gl_probe.py` 量化了 6 条路径），
随即暴露了上面这个真正的 bug。

修复：`target = ctx.detect_framebuffer()`，按 `target.size` 设 viewport 并清屏。
修复后 `grabFramebuffer()` 读回 542 色 / 154400 非黑样本，图像目视确认正确。

### 3.5 一致性

| # | 现象 | 根因 |
|---|---|---|
| 28 | `GridMaps` 文档声称每点存两份 | 早前移除径向导数后文档没跟上，索引公式还有优先级错误 |
| 29 | `workbench_smoke.py` 把源码树插到 `sys.path` 最前 | 删掉陈旧 `.pyd` 后会导入失败；改为优先用已安装的 |
| 30 | wheel 打包 `__pycache__` | 加 `exclude` |
| 31 | GPU 后端从未被调用 | `GpuContext` 只在 `gpu/mod.rs` 内部被引用，搜索与打分主路径完全没用它 |

### 3.6 测试夹具本身的问题

| # | 现象 | 根因 |
|---|---|---|
| 32 | `end_to_end_dock_finds_a_low_energy_pose` 在加入碰撞过滤后失败 | 该夹具把丁烷**同时**当受体和配体（"把分子对进它自己"），以前靠允许碰撞才通过 |
| 33 | 我为"大配体超 workgroup"写的夹具越界 | 追加原子后没有重建键图，邻接表比原子表短——是**越界索引**而非优雅报错 |
| 34 | 我的 crambin 检查脚本对 6 个 Cys 误报 | 二硫键的 SG 已被占用，**本来就不该有氢**，是化学正确的，脚本错了 |
| 35 | `interpolation_is_linear_along_one_axis` 在修好半径后变红 | 夹具在**非线性**的场上断言线性，过去只是碰巧通过。测的是场，不是算子 |
| 36 | `audit_poses.py` 报 5 个姿势里 3 个"1.79 Å clash" | 脚本把受体**极性氢**也算进去了。1.79 Å 正是正常 H···O 氢键距离；引擎的 `receptor_atoms()` 只返回重原子，脚本没跟上 |
| 37 | `audit_poses.py` 报"gradient not converged" | 用 `|grad| > 1.0` 判断收敛。正确姿势恰好坐在 cell 面上，两个单边梯度都大——见 §3.2.2 |
| 38 | 随机随机构象下"解析梯度 = 10⁴ 量级" | 那些构象大半在盒外，`OUT_OF_BOX_PENALTY = 1000` 主导了能量，而罚函数是**阶跃**不是斜坡，中心差分天然给 0。测试夹具错误，不是引擎错误 |

第 33 条值得单独说明：它复现的正是**引擎自己曾经踩过的坑**——
`Molecule::perceive_bonds` 会同时重建原子表和邻接表，只追加原子不重建就是越界。

第 38 条值得记一笔，因为**它一度看起来像"梯度放大了一万倍"的灾难性 bug**。
区分它只需要先问一句"这些构象在网格里吗"。

### 3.7 GUI —— 上一轮"已验证"其实只验证了渲染

这一节全部来自"用 `QTest` 投递真实鼠标/滚轮事件、读回控件状态、逐状态截图"
的重新验证（`scripts/workbench_interaction_check.py`，44 项检查，exit 0）。

**上一轮为什么一个都没发现**：冒烟脚本自己构造 `MoleculeView` 并手工解析 pose，
所以 `MainWindow.load_structure(kind="poses")` 这条路径**一次都没有被执行过**。
渲染验证通过的同时，`odgui -p poses.pdbqt` 是**启动即崩**的。

| # | 现象 | 根因 |
|---|---|---|
| 39 | `odgui -p poses.pdbqt` **启动即崩溃** | `read_pdbqt_models` 返回**行的列表**，而 `_energy_of` 对每个元素调 `.splitlines()`。写入方与读取方的接口契约不一致 |
| 40 | 姿势浏览器里 9 个 pose **全部显示 "0.00 kcal/mol"**，且"最佳 pose"其实只是第一行 | 写入器把 `REMARK VINA RESULT` 放在 `MODEL` **之前**，按 `MODEL..ENDMDL` 切分的读取器看不到它；`argmin` 在全等值上返回 0 |
| 41 | RMSD-to-best **永远是 "—"** | `setCurrentRow(best)` 发生在 `self._pose_view` 赋值**之前**，`_on_pose_selected` 直接 early-return |
| 42 | 连续两次 Dock 后**上一轮姿势仍留在画面上**，"ligand"勾选框再也藏不掉 | `_on_visibility` 只认当前 pose view；身份靠 `name` 匹配，而每次运行的 pose 来自**不同名**的临时文件 |
| 43 | 受体中心为**负坐标**时 box centre 被静默钳到 0，盒子偏离分子 40 Å | spin 的 `range(0.0, 999.0)`。负坐标在真实 PDB 里很普通 |
| 44 | 对接运行期间**窗口完全无响应** | 网格预制表在 GUI 线程上跑：`setText` 之后事件循环无法重绘，用户看不到任何提示 |
| 45 | 每次点 Dock **在 %TEMP% 留一个 `tmp*.pdbqt`** | `NamedTemporaryFile(delete=False)` 且从不删除 |
| 46 | 关闭窗口时若正在对接 → **进程 abort** | `QThread` 仍在运行时被析构。已加 `closeEvent` 停线程 |
| 47 | **苯环被画成绿色/蓝色**，与同一分子的脂肪碳颜色不同 | PDBQT 原子类型 `A`（芳香碳）被"取首个字母"规则映射成虚构元素 `"A"`，颜色表查不到 → 回退到 view 颜色 |
| 48 | 极性氢**和碳一样大** | 绘制时所有原子共用一个标量半径 |
| 49 | 文档写了"右键平移"，**代码里根本没有平移** | `mouseMoveEvent` 只处理 Left/Middle；Right 只设了 `_dragging` 却没有分支 |

第 40 条同时是**写入端的 bug**，已修 Rust `write_pose` 与 Python `write_pose`：
`REMARK` 现在落在 `MODEL..ENDMDL` 内（AutoDock 惯例），并新增
`pdbqt::tests::the_energy_remark_is_inside_the_model_block` 锁住这个布局。

第 47 条的教训通用：**"元素 = 类型去掉极性后缀"对 `A` 不成立**，
因为 `A` 本身就是元素。已改为显式的 `PDBQT_TYPE_ELEMENT` 表 + 未知类型回退。

修好后，独立视觉模型对渲染图的复核结论：CPK 配色正确
（碳灰 / 氧红 / 氢小而白）、**没有绿或蓝色的碳组**、氢明显更小、
无黑斑与缺键、黄色搜索盒线框可见。

> 一个**有意保留**的不一致：键按 view 颜色画（配体浅蓝、姿势绿），
> 而原子按元素上色。因为 pose 和 ligand 的球体现在看起来完全一样，
> 键的颜色是唯一能区分二者的视觉线索。改成逐原子配色会让两者无法分辨。

### 3.8 GUI 交互验证结果（exit 0）

| 分组 | 结果 |
|---|---|
| 布局 | 16 个控件全部有可用尺寸、无溢出面板、无互相重叠；dock 491 px，视口 785×787 |
| 元素解释 | 三个 view 的元素全是真实元素；pose 与其配体元素组成完全一致；H 半径 0.119 Å vs C 0.340 Å |
| 相机 | 左键旋转、pitch 钳位 ±1.5、滚轮缩放、Frame all 取景、**右键平移**、中键变焦——全部生效 |
| 可见性 | 三个勾选框各自真实减少像素（71916 → 42856 / 32475 / 69876） |
| box 控件 | centre 与 size 的 spin 确实移动/缩放线框；负坐标可输入（±9999） |
| **File 菜单** | 提供 `Load receptor…` / `Load ligand…` / `Open poses…` / `Dock now`；不再有笼统的 "Open structure" |
| 姿势浏览 | 9/9 行都产生读数；RMSD 0.00 Å → 7.78 Å；换行确实改变画面；列表按能量排序 |
| 真实对接 | 按钮禁用→恢复；事件循环 40/40 轮保持响应；状态栏报告结果；两次运行后场景恰好是受体+配体+姿势；0 个临时文件泄漏 |

### 3.9 `odgui` 入口点

`odgui` 与 `odcli` 是两个**互相独立**的 console script
（`opendocking.workbench.launcher:main` 与 `opendocking.cli:main`）：
装了工作台不影响 CLI，只装 CLI 也不影响工作台。
启动器不在模块顶层 import Qt / modernGL，
所以 `odgui --help` 与 `odgui --check` 在没有图形栈的机器上也能给出可读的结果，
而不是抛 traceback。

| 检查 | 结果 |
|---|---|
| `odgui --check`（无窗口） | `GUI stack OK`，exit 0 |
| `odgui --help` | 经 PATH 解析正常 |
| **真实启动**（`scripts/odgui_launch_check.py`） | 窗口 1294×858、951 种颜色；`WM_CLOSE` 干净退出，**exit 0** |

该脚本用 `WM_CLOSE` 而不是 `terminate()` 关闭，因为只有前者会走 `closeEvent`——
而"对接进行中关窗导致 `QThread` abort"正是 §3.7 第 46 条要防的情况。

> 一度以为存在"结构没有居中"的取景 bug。**那是测量错误，不是渲染错误**：
> `grabFramebuffer()` 返回**设备像素**（本机 DPR 1.25，981×984），
> 而我拿控件的逻辑尺寸 785 去比，于是凭空得到 25% 的偏移。
> `scripts/viewport_framing_check.py` 现在按图像尺寸判定：
> 实测内容中心偏移 **−0.1%**，`frame_all` 前后一致。
> "内容触到边缘"是正常的——22 Å 搜索盒比分子大，本来就该铺满视野。

### 3.10 改名过程中发现并修复的 bug

本项目把命令名收敛成两个——`odgui` 与 `odcli`——并把包名定为 `opendocking`。
改名不是纯文本替换：**改名漏掉了一整类引用，而现成的验证脚本正好覆盖了它们**。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 50 | `examples/` 下 4 个脚本、`scripts/workbench_interaction_check.py` 全部 `NameError` | 改名脚本重写了 `import X` → `import opendocking`，但**没有**重写 `X.Receptor` 这类**用法**。import 成功、执行时才炸 | 逐个改为 `opendocking.`；`examples/*.py` 与 `scripts/*.py` 现在全仓库无残留用法 |
| 51 | `robustness_check.py` 的三个 "panic 不杀进程" 用例**一直在假通过** | `run_in_subprocess` 只看 `returncode not in (0, 1)`。改名后子进程因 `ModuleNotFoundError` 退出 1，于是被当成"引擎干净地抛了可捕获异常" | 增加守卫：子进程 stderr 出现 `ModuleNotFoundError` / `NameError` 直接判 FAIL，并说明"什么都没测到" |
| 58 | CI 的 `gpu` 配置 clippy 失败，而本机通过 | 本机 rustc 1.87 **没有** `mismatched_lifetime_syntaxes` 这条新 lint；CI 用 `@stable`（1.98）有。只有开 `--features gpu` 才编译到那段代码，所以 default 配置的 job 是绿的 | `binding_entry` 显式命名输入与输出生命周期；本机工具链升到 1.98.1 与 CI 对齐后重跑 |

第 51 条比第 50 条更值得记：**一个无法区分"引擎抛了干净异常"和"脚本自己坏了"的测试，
等于没有测试**。守卫本身也做了反向验证——故意让子进程 import 一个不存在的模块，
确认它被判 FAIL 而不是 PASS。

> 发现 50、51 的过程值得记下来：`opendocking.tests` 37 项全绿，
> 但它们测的是**包内**代码；`examples/` 与 `scripts/` 在包外，
> 不在任何测试的收集范围内。改名改的恰好是包外的编排层。
>
> 58 条则说明另一件事：**"本地绿"不等于"CI 绿"**。本地工具链比 CI 旧，
> 新增的 lint 根本不会触发。工具链版本应当与 CI 对齐，否则验证是有缺口的。

### 3.11 命名与安装状态（实机核查）

四处名称是**分开的**，这是有意的：

| 名称 | 值 |
|---|---|
| distribution | `opendocking` |
| import 包 | `opendocking` |
| GUI 命令 | `odgui`（工作台） |
| CLI 命令 | `odcli`（7 子命令） |
| PyO3 扩展 | `opendocking._dockpy` |

只安装两个命令，没有第三个别名——多一个别名只会让用户多记一个名字。

| 检查 | 结果 |
|---|---|
| `pip list` | 只有 `opendocking 0.1.0`，无 invalid-distribution 警告 |
| `site-packages` | 只有 `opendocking/` 与 `opendocking-0.1.0.dist-info/`，无 `~dock*` 残骸 |
| Python 的 `Scripts` 目录 | 恰好 `odgui.exe` 与 `odcli.exe` |
| `dist/`、`dist-gpu/` | wheel 文件名跟随分发名，为 `opendocking-0.1.0-*.whl` |
| `[project.scripts]` | 只有 `odgui` 与 `odcli` 两条，重新安装也不会变出第三个 |
| CI 门禁 | 有一步专门断言"装出来的 console script 恰好是这两个" |

`odcli workbench` 子命令仍保留，等价于 `odgui`，只为兼容既有脚本。

### 3.12 发布前的仓库体检

准备开源提交时逐项检查，发现并修掉 6 个问题。前 4 个不是"bug"，
但每一个都会让拿到仓库的人踩坑。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 52 | `examples/maps/` 有 **40 个 `.map`、44.6 MB**，会被提交进 git | `odcli rec-grid` 的输出目录，之前既没进 `.gitignore`，也不在任何文档里 | 加入 `.gitignore`；`examples/multi/`、`examples/split_dir/` 同理（`dock` / `split` 的输出）。`examples/ligands/` **是**源码，保留 |
| 53 | 仓库**不是 rustfmt-clean** 的 | 手写代码未经 `cargo fmt` | `cargo fmt --all`；随后 104/114 测试与双配置 clippy 全部重跑，exit 0 |
| 54 | 4 个文件行尾是 **CRLF / 混用** | 在 Windows 上手写，`dock-core/Cargo.toml`、`dock-py/README.md` 等混了 LF 与 CRLF | 新增 `.gitattributes`（`* text=auto eol=lf`），Windows 脚本单独 `eol=crlf` |
| 55 | `dist-gpu/` 没被忽略 | `.gitignore` 只写了 `/dist/` | 补上，2.4 MB 的 GPU wheel 不再进仓库 |
| 56 | 验证报告里漏了本机绝对路径 | 上一轮核查时直接写了本机 Python 的 `Scripts` 绝对路径 | 改成"Python 的 `Scripts` 目录"；新增 `scripts/check_repo_docs.py` 专门守这条。这条后来又抓到一次——本文早先引用这个 bug 时把原路径抄了回来 |
| 57 | 元数据里的仓库地址是手写的占位值 | 写元数据时还没定仓库 | 7 个文件 11 处先改为 `<owner>/<repo>` 标 `TODO(RELEASE)`，确定账号后再统一替换为真实地址 |

另外补了 12 个开源社区文件此前完全缺失：
`CONTRIBUTING.md`、`CHANGELOG.md`、`SECURITY.md`、`CODE_OF_CONDUCT.md`、
`CITATION.cff`、`.gitattributes`、`requirements.txt`、`README.en.md`，
以及 `.github/` 下的 2 个 issue 模板、PR 模板和 CI 工作流。

**`odck.md` 没有进仓库。** 它是最初给 AI 的需求稿（开头是"你是一名资深科学计算
全栈架构师…"的提示词），里面的模块名（`*.chem.receptor` 等）从未实现，
不是用户文档。

#### 发布目录的独立性验证

`opendocking-gui/` 只含源码与文档，**94 个文件、972 KiB**。不是"看起来干净"，
而是逐项验过：

- 与工作区**逐字节比对** 94/94 一致，没有漏拷也没有多拷
- 22 个必备文件齐备，7 类禁入项（`odck.md` / 构建产物 / 缓存）一个都没有
- 文档卫生检查**在该目录内部**重跑：0 编码损坏、0 断链、0 本机路径
- 可执行源码里旧包名残留 **0** 处
  （`docs/VERIFICATION.md` 里有 1 处，是**举例说明 §3.10 那个 bug**，是有意保留的）
- **在该目录内部从零构建并测试**：rustfmt → 106/116 测试 → 双配置 clippy →
  wheel 构建安装 → 38 pytest → `odgui --check` → 5 个 examples →
  GUI 交互 75 项 → 真实启动关窗

最后一条是关键：它证明这个目录不依赖任何没被复制的文件。

### 3.13 CPU/GPU 一致性单测的阈值本身就是错的

推上 GitHub 后 CI 第一次真正跑完，`engine / bindings (gpu)` 在 Python 测试这一步变红——
**不是 clippy，也不是引擎算错了**：

```
AssertionError: GPU and CPU disagree by 0.000114503
assert 0.00011450314968897146 < 0.0001
```

本机 RTX 3050 上同一个断言是绿的。量了实际数字才看清根因：

| 量 | 值 |
|---|---|
| 该批构象的得分量级 `max\|cpu\|` | **1042** |
| 本机（NVIDIA）最差绝对误差 | 3.81e-05 |
| CI（软件光栅化）最差绝对误差 | 1.15e-04 |
| `f32_eps × 量级` | **1.24e-04** |
| CI 相对误差 | **1.10e-07** ≈ 0.92 个 f32 ulp |

原来的断言写的是 `worst < 1e-4`，一个**固定绝对阈值**。而这些构象是随机摆的，
相当一部分伸出盒子，得分到几百，单精度下一个 ulp 就有 `1e-4` 量级——
**阈值比它要检查的算术本身还小**。所以它不是"CI 更严所以暴露了问题"，
而是这个阈值在任何机器上都不可能可靠：NVIDIA 的编译器碰巧落在下面就绿，
换一个就红。误差本身完全在单精度舍入范围内，引擎没有问题。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 58 | gpu job 的 `test_the_gpu_batch_path_agrees_and_reports_itself` 在 CI 挂、本机绿 | 断言用固定绝对阈值 `1e-4`，低于该批得分量级下一个 f32 ulp（1.24e-4）。机器换了、shader 编译器换了就必红 | 改为**相对**阈值：`worst <= 1e-5 * max(1, max\|cpu\|)`，并在注释里写明为什么绝对阈值在这里没有意义 |
| 59 | `examples/audit_poses.py` 的 `gpu_gap > 1e-3` 是同一个错 | 同一类：绝对阈值套在可能带 clash 的位姿上（clash 位姿能量可达几百） | 一并改为相对阈值，flag 里带上相对数值 |
| 60 | `docs/ARCHITECTURE.md` 声称 GPU 与 CPU 结果"逐位相同" | **不实**。拆分只保证**分子内项**逐位相同，分子间插值是单精度的，测得相对误差 ~1e-7 | 改正说法，并补上两个平台上的实测相对误差 |
| 61 | `core.evaluate_conformations` 的 docstring 写"agrees to about 1e-8" | 同上，且 1e-8 是在**得分很小**的位姿上量的绝对值，被当成了普遍结论 | 改为"相对误差约 1e-7，量级 10 的得分上约合 1 微卡/摩尔" |
| 62 | workbench job 在 "Install system libraries" 6 秒失败：`E: Unable to locate package libxkbcommon-x11` | 包名漏了 `-0` 后缀。core job 里写的是对的，gui job 是从别处复制时漏掉的 | 改为 `libxkbcommon-x11-0` |

这一条与 §3.10 的教训是同一类：**断言写死的数字，必须问它"这个数字和什么量纲比"**。
"GPU 和 CPU 相差小于 1e-4" 读起来像一个物理精度声明，实际上它是一个
**关于测试机器 shader 编译器的断言**。

#### 相对阈值的取舍，以及它的反向验证

把阈值改成相对的不是"放宽"，所以必须证明它**仍然能抓住真 bug**。
对 10 个点逐个核对了新阈值的判定：

| 情形 | 最差绝对误差 | 得分量级 | 判定 |
|---|---|---|---|
| 本机 NVIDIA（实测） | 3.81e-05 | 1042 | pass（余量 **91×**） |
| CI 软件光栅化（实测） | 1.15e-04 | 1042 | pass |
| 无 GPU，CPU 回退 | 0 | 1042 | pass |
| 注入 1e-3 相对误差的真 bug | 1.04 | 1042 | **FAIL** |
| 注入 1e-2 相对误差的真 bug | 10.4 | 1042 | **FAIL** |
| 注入 0.01 绝对误差 @ 得分 −10.4 | 0.01 | 10.4 | **FAIL** |
| 注入 0.01 绝对误差 @ 得分 −9.8 | 0.01 | 9.8 | **FAIL** |
| 得分 0.5、f32 噪声 | 6e-08 | 0.5 | pass（`1.0` 下限生效） |
| 得分 0.5、注入 0.01 | 0.01 | 0.5 | **FAIL** |

**纯相对阈值有一个代价，这里明说**：在得分 1042 上，`0.01` 的绝对误差是
`9.6e-6` 相对误差，会**通过**。之所以接受，是因为 `+1042 kcal/mol` 意味着
构象被摆到了网格之外很远，那是一个没有人会当作结论的数字。
而在**真正会被解读的得分区间**（对接位姿约 −10 kcal/mol），
同样的 `0.01` 绝对误差比阈值大 100 倍，照样被抓住——上表后两行就是这个。

没有再加一道绝对上限：任何在量级 1042 上不误伤的绝对上限都 ≥ 0.01，
而在得分 −10 上它会比相对判据更松，反而**削弱**测试在关键区间的牙齿。
单精度舍入本身就是相对量，加绝对上限会把刚修掉的那个 bug 请回来。

### 3.14 重新跑 examples 时挖出的三个 bug

CI 的两个 job 都绿之后，回到工作区把 `examples/` 5 个脚本在**仓库根目录**重跑
（CI 根本不跑它们），结果挖出三个问题。其中一个是**公开 API 彻底失效**。

#### 3.14.1 `Ligand.from_pdbqt_str` 从来没有工作过

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 63 | `Ligand.from_pdbqt_str(合法 PDBQT 文本)` 在**任何输入**上都失败 | `dock-py/src/lib.rs` 里这个函数调的是 `read_pdbqt(text)`，而 `read_pdbqt` 收的是**文件路径**并会去 `File::open`。等于把整段 PDBQT 当文件名去打开。Windows 报 `os error 123`（"文件名语法不正确"，因为 PDBQT 满是换行），Linux 去找一个名字等于整篇文档的文件 | 改用 `parse_pdbqt(text)`。同文件的 `Receptor::from_pdbqt_str` 本来就是对的（走 `dock_core::receptor` 里的 `parse_pdbqt`），所以只有配体这一侧坏了 | 新增 `test_loading_from_text_matches_loading_from_a_file`：文本路径与文件路径必须给出相同的原子数、扭转数、自由度数，且**在同一构象上给出相同能量** |

这个 bug 能活下来，靠的是**它只被用在失败路径上**。唯一调用它的是
`robustness_check.py` 里的三个子进程测试，而那三个测试传的是**故意写坏的
PDBQT**——不管解析器行为如何，它们都会得到一个可捕获的 `ValueError`，
退出码 1，被判为"没有 abort，通过"。**一个正确的解析器和一个完全坏的解析器，
在这三个测试眼里一模一样。**

修完之后再跑，那三个子进程报出来的才是真正的解析错误，而不是 `os error 123`。
这和 §3.10 是同一个形状：**只断言"没有崩"，就分不清"做对了"和"根本没做"。**

#### 3.14.2 防假通过的守卫自己会崩

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 64 | `robustness_check.py` 在中文 Windows 上直接 `TypeError: argument of type 'NoneType' is not iterable` | `subprocess.run(..., text=True)` 不指定编码时用**系统 locale 码页**解码子进程输出，本机是 GBK。子进程混了两种编码：Python 自己的 traceback 跟 locale，Rust 引擎写 UTF-8（`...Ångström`）。解码异常发生在 `subprocess` 的读取线程里，线程死掉后 `buffer.append()` 从没执行，`proc.stderr` 保持 `None`——而守卫写的是 `"..." in proc.stderr` | 显式 `encoding="utf-8", errors="replace"`：解码不匹配退化成可见的替换字符，而不是杀掉读取线程。再加 `stderr = proc.stderr or ""` 兜底。注释里写明这个函数存在的意义是**报结论**，不能自己抛异常 |

#### 3.14.3 三个 example 在仓库根目录跑不起来

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 65 | `python examples/audit_poses.py` 等报 `FileNotFoundError: '1crn_prep.pdbqt'` | 默认参数是裸文件名，相对**调用者的当前目录**解析，只有先 `cd examples` 才行。而没人会 `cd examples` | 三个脚本的默认路径改为相对 `Path(__file__).parent` 解析，任意目录可跑 |

这三个的共同点：**它们此前"验证通过"是因为我一直在正确的目录下调用。**
CI 不跑 `examples/`，所以这条路径在 CI 上从来没有被覆盖过。

### 3.15 修好 3.14.1 之后，暴露出的更深一层问题

`from_pdbqt_str` 修好之后，`robustness_check.py` 里"两个原子重叠"这一项**从绿变红**。
不是它本来是错的，是**它此前是被那个 bug 满足的**：`from_pdbqt_str` 无论传什么都报
路径错误，所以这一项和"故意写坏的输入"长得一模一样。

把 bug 修掉，才第一次真正测到解析器——于是测出一个新问题。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 66 | 两个原子落在同一坐标的配体被**静默接受**，能量、梯度、批量打分、完整对接全部返回有限值，甚至给出一个 −0.61 kcal/mol 的"最优构象" | 引擎没有原子间距校验。这类结构不会崩：重叠对要么是成键邻居（本来就被排除在分子内项外），要么被短程钳位挡住，所以每一个数字都是有限的 | `Ligand::from_molecule_with` 增加 `MIN_ATOM_SEPARATION = 0.5 Å` 下限并拒绝，错误信息给出两个原子的名字、序号和实际距离。新增两个 Rust 测试：重叠必被拒；真实几何不得被误拒 |

**为什么阈值取 0.5 Å**：任何真实分子中最短的接触是 H–H 键约 **0.74 Å**，
所以 0.5 Å 不可能误伤真实结构，只可能抓住"同一坐标"这种截断或准备失败的文件。
配一个测试把这个余量钉住：构造一个最短接触为正常 C–H 键长（1.09 Å）的分子，
断言它通过——**如果这个下限哪天被调高，测试会先红**。

这个 bug 比崩溃更危险。崩溃你看得见；它给出的是**有限、合理、但建立在不存在的
几何上的**数字。

#### 测试夹具里藏着一个物理上不可能的分子

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 67 | 加上间距校验后，`a_large_ligand_falls_back_to_the_cpu` 失败：原子 0 与原子 4 距离 0.000 Å | 夹具把丁烷复制 40 份、每份只沿 x 平移 **0.1 Å**，而丁烷本身跨 3.65 Å，必然出现完全重合的原子（副本 k 的 C1 与副本 k+15 的 C2 坐标相同）。这个夹具只是想造一个 >64 原子的分子来测 CPU 回退，重叠是意外的 | 改为 4×4×3 晶格、步距 3.0 Å，副本间最小距离 ≥1.7 Å；全部原子都从晶格生成（保留原模板会与晶格原点的副本重复） |

这条值得单独记一笔：**这个夹具一直在用一个物理上不可能的分子做测试**，
而测试是绿的。它检验的是"大配体能否回退到 CPU"，不是"分子是否有效"，
所以重叠对它无所谓——引擎也不在乎，于是没人发现。

#### 门禁自己抓到的一个错误

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 68 | 新写的 `closest_atom_pair` 用了 `Option::is_none_or`，clippy 报 `incompatible_msrv` | 该 API 从 **Rust 1.82** 才稳定，而本 crate 声明的 MSRV 是 **1.75**。本机工具链是 1.98，编译得过 | 改用 `map_or(true, ..)`，并把原因写进注释 |

以及一处**我自己写的测试自己抓到的错**：新测试里我把一个 H 放在 C–C 键轴上，
结果它距另一端碳 0.45 Å——正好低于我刚设的下限，被自己的断言抓住。
另外 `closest_atom_pair` 在"没有近距离对"时返回 `None`，我第一版写成
`.expect("no close pair in real geometry")`，断言方向写反了。两条都是当场改掉的。

### 3.16 把 CI 的 GUI job 真正跑起来，以及它能证明什么

前面写的"GitHub runner 上没有可用的 OpenGL 3.3 环境"是**没有验证过的假设**。
补上 `libxcb-cursor0` 之后实测如下：

| 步骤 | 结果 |
|---|---|
| `odgui --check`（无需显示） | ✅ 通过 |
| xvfb + `LIBGL_ALWAYS_SOFTWARE=1` 下的 OpenGL 上下文探测 | ✅ **通过** |
| 布局 / 交互 / 行为 50 项 | ⚠️ **45 通过，5 失败** |
| 真实启动再关闭 | 被上一步挡住未执行 |

也就是说 runner 上**是有可用 OpenGL 上下文的**。5 个失败全是同一类——
以渲染像素为判据的检查：

```
[FAIL] viewport renders geometry              — 0 non-background px
[FAIL] unchecking 'receptor' removes pixels   — 0 -> 0 px
[FAIL] unchecking 'ligand / pose' removes px  — 0 -> 0 px
[FAIL] unchecking 'search box' removes wire   — 0 -> 0 px
[FAIL] selecting a different pose changes it  — identical
```

`grabFramebuffer()` 在无头软件光栅化下返回**整片单一颜色**（0 个非背景像素）。
其余 45 项——相机旋转、平移、缩放、微调框、位姿列表、对接、场景记账——全部通过，
它们不依赖像素回读。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 69 | 无头 runner 上 5 项像素检查报 FAIL，**看起来像产品坏了** | `grabFramebuffer()` 在无合成器的软件光栅化下回读为空白。`win.show()` + `qWait(600)` 并不能保证窗口真的被暴露 | 改为：先探测帧缓冲是否可读（并显式调用 `qWaitForWindowExposed`），读不到就把这 5 项记为 **SKIP 并写明原因**，而不是 FAIL。汇总里 SKIP 单独计数、单独打印、**不计入通过**，且不决定退出码 |

**这一条和 §3.14.1 是同一个形状。** 一个测不出东西的检查报 FAIL，等于宣称
"产品有问题"；而实际的事实是"这个环境回答不了这个问题"。两句话只有一句为真，
所以必须分开写。

改这条的时候我自己也犯了一次同类错误，而且**只有反向验证才能发现**：
`main()` 里写 `PIXELS_OK = False` 创建的是**局部变量**，模块全局仍是 `True`，
skip 分支根本不会触发——脚本看起来改好了，实际行为一点没变。
用一个驱动脚本把 `shot()` 换成返回单色图，复现 CI 的条件，验证：

| 断言 | 结果 |
|---|---|
| 空白帧缓冲 → 退出码 0 | ✅ |
| 0 个 FAIL、恰好 5 个 SKIP | ✅ |
| SKIP 的正是那 5 项 | ✅ |
| 其余 45 项**仍然运行且仍然通过** | ✅ |
| 本机正常路径仍是 50/50、0 SKIP | ✅ |

最后一条是关键的对照：**如果 skip 机制在真机上把通过项也吞掉了，这条会红。**

### 3.17 3D 显示方式：键的正确性，以及为此挖出的四个缺陷

新增 `display` 选择器：空间填充 / 球棍 / 键线式 / 条带 / 卡通。
真正的工作量不在画，而在**键必须连对原子**——这正是球棍模型最容易骗人的地方。

原来的规则只有一句："距离小于 1.95 Å 就连"。对二十个原子的药成立，
对蛋白则是灾难：crambin 上这条规则会造出 **48 条跨残基键**。

改成三条路径，**距离只用来审计，不用来决定连不连**：

| 结构类型 | 键的来源 | 本轮实测 |
|---|---|---|
| 配体 / 姿势（有 `ROOT`/`BRANCH`） | 文件**声明**的树 | 16 键，全部物理可能 |
| 蛋白质 | **残基模板** + 肽键 | 389 键，**跨残基键 0 条**（距离法是 48 条） |
| 扁平小分子（两者皆无） | 共价半径距离 | 20 键，**与 RDKit 独立感知完全一致** |

#### 挖出的四个缺陷

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 70 | `prep-receptor` 把残基信息**全部丢弃**：46 残基的蛋白被压成一个 `REC`、编号 1、无链、无原子名 | `format_atom_line` 签名里**根本没有 chain 参数**，`resname` 写死，`resseq` 写死为 1，`name` 用的是元素符号。RDKit 的 `PDBResidueInfo` 一直都有这些信息，只是没写出去 | 四个字段都从每个原子自己的 PDB 记录取；新增 chain 列 |
| 71 | 加的极性氢**沿用重原子的名字**，一个残基里出现三个 `N` | `AddHs` 新建的氢没有 PDB 记录，回退时返回了重原子的名字 | 氢一律写 `H` |
| 72 | 3 个加的氢被写在 **(0, 0, 0)** ——原点上的原子，离自己的父原子 20 Å，却照样参与对接 | `AddHs` 偶尔给不出三维位置，原样写出 | 离父原子不在 0.7–1.35 Å 的氢**丢弃并告警**，而不是写进文件 |
| 73 | 我自己写的 ILE 模板把 **CG2 连到了 CG1** | 记成了分支点在 CG1。异亮氨酸的分支点是 CG1（连 CB、CD1、CG2），**CG2 挂在 CB 上**；亮氨酸分支点才是 CG，缬氨酸是 CB 直接连两个甲基 | 改为 `CB`–`CG2`。**是键长审计抓到的**：crambin 里这两个碳相距 2.61 Å |

第 73 条值得单独说：它是我在**新写的代码**里的错误，由**同一批代码里的审计器**抓出来。
crambin 的 ILE7/25/33/34/35 五处都报了 "2.6 Å 的 C–C 键，比 C–C 键能有的长度还长"。
没有审计器，这就是五根画错的键，而且看起来完全正常。

#### 姿势文件自己声明了一根不存在的键

姿势的 PDBQT 里有一个分支：

```
BRANCH  11  13
    ATOM 13 C13 / ATOM 14 O14 / ATOM 15 O15 / ATOM 16 H16
```

PDBQT 规定块内原子**依次成链**。但羧基的化学是 C13(=O14)(−O15−H16)——一个**分叉**，
AutoDock 用**嵌套 BRANCH** 表达，而本项目的写入器把它写成了平链。
于是文件声明 O14–O15 成键，而它们相距 **2.23 Å**。

修 bug 前，这个文件给出 **22 键**，C7 有 6 根键（碳最多 4）。修之后 **16 键**，
全部物理可能，无过价原子。处理原则是**声明决定拓扑，几何可否决**：
一根 2.2 Å 的氧–氧键不管谁声明的都不画，并且报告出来。

**写入器本身还没修**（需要嵌套 BRANCH），已写进 CHANGELOG 的 Known limitations。

#### 验证

新增两个无头脚本（不依赖 Qt 与显示）：

| 脚本 | 检查数 | 关键断言 |
|---|---|---|
| `scripts/structure_bond_check.py` | **36 / 36** | 46 个残基各自成组；45 根肽键；**没有一根键跨残基**（除肽键）；每根侧链键都在；脯氨酸侧链闭合回 N；所有键长 0.99–1.82 Å；无过价原子；**与 RDKit 对同一文件的独立感知一致**（20 = 20） |
| `scripts/representation_geometry_check.py` | **21 / 21** | 圆柱确实跨越它该跨的那根键；双色键两半各自着色；条带**没有一点远离所有 CA 原子**（最差 1.52 Å）；条带尺寸不比它跟随的迹线大 |

`workbench_interaction_check.py` 从 50 项扩到 **75 项**，新增两组：
五种显示方式的选择与绘制，以及**真蛋白（crambin）上的条带**——
示例受体是合成团块，没有它条带就从未被真正画过一次。

**必须看图**：条带第一次画出来是"串珠状"的，数字全对而画面是错的。
原因是逐残基用 CA→CB 定向，而相邻残基之间这个向量摆动几十度，
截面在每一步都转一次。改成**平行传输**（沿样条传递朝向）后正常。
这一条是看截图才发现的，任何断言都发现不了——"每个顶点都靠近 CA"
和"画面是一根连续的带"是两回事。

### 3.18 推送后的 CI 抓到的缺陷：一个从未通过的检查

上面这一版推上去之后，CI 的 `Real launch, then close` 是红的。
查日志发现**上一版也是红的**——这个检查从来没有通过过。

它之所以看起来和通过的检查一样，是因为所在 job 是 best-effort。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 74 | `Real launch, then close` 在 CI 上必然超时失败（60 秒后 `exit code: -15`、`closed via: none`） | CI 用 `xvfb-run`，**只开显示、不起窗口管理器**。而 `wmctrl` 是向窗口管理器要 EWMH 的 `_NET_CLIENT_LIST`——没有 WM，它连窗口都列不出来，循环只能空转到超时 | **找**窗口改用 `xdotool search`：它走 `XQueryTree` 直接遍历 X 树，不需要 WM |
| 75 | 即使有 WM，`wmctrl -l` 分支也永远匹配不上 | 输出是 `<id> <desktop> <host> <title>` 四个字段，标题里带空格。原代码用 `split(None, 4)` 后要求 `len(parts) == 5`——**四字段的行永远凑不出五个元素**，这条分支是死代码 | `split(None, 3)`，标题作为不切分余数留在 `parts[3]` |
| 76 | 关窗机制不可用时，判定近乎空断言 | 判定写的是 `ok = rc is not None`，但那时 `odgui` 通常是被脚本自己的 `finally` 里的 `terminate()` 杀掉的——**那个退出码是脚本自己造的**，与 odgui 的健康无关 | 改为看 `exited_early`：找到窗口时进程还活着，才算 PARTIAL 通过 |
| 77 | 换用 xdotool 后**仍然红**，但这次窗口第一次被找到了：0.6 秒找到 `2097159`，20 秒后进程仍在 | 我以为 `xdotool windowclose` 等价于 `PostMessage(WM_CLOSE)`。**不是**：它直接 `XDestroyWindow` 销毁 X 窗口，Qt 收不到关闭事件，`closeEvent` 根本没跑——窗口没了，进程还活着。这是一个**不同的事件**，不是更慢的关闭 | 关窗改回 `wmctrl -c`（经窗口管理器投递真正的关闭请求），CI 里装并启动 `openbox`；`xdotool windowclose` 降级为「放弃前清理窗口」，**不再计入已验证的关闭** |

第 77 条值得单独说：**它是我照着「看起来等价」的直觉写的，而且第一次修的时候还把它写进了文档。**
真正定案的是日志——窗口在 0.6 秒被找到、20.6 秒进程仍存活，
这个组合只能是「窗口被销毁了」而不是「关闭很慢」。
如果当时只是把超时从 20 秒加到 60 秒，看起来也许能"修好"，但验证的仍然是另一个事件。

第 76 条是顺手发现的同类问题：只要退出码"有值"就算过，而它必定有值。
断言写在那里是为了让检查不要失败，但一个恒真的断言不提供任何信息。

#### 给它加了守卫

X11 那段逻辑在 Windows 上跑不了，但在 CI 上是 best-effort 的——
**一个从未成功的检查和一个成功的检查，在 Actions 列表里长得一模一样**。
所以把解析部分单独拆出来验证：把三个工具的真实输出格式喂进去，
断言该认出窗口 id、且**不认错近似标题**。

`scripts/x11_window_parse_check.py`：**11 / 11**，不需要 X server，
Windows 和 Linux 都能跑，已接入 workbench job。其中三项分别钉住 74、75、77，
最后一项断言**只装了 xdotool 不算作「有办法关窗」**。

> **诚实说明**：真实 X server 上的行为本地无法验证，只有 CI 的 runner 能证明。
> 本地已验证的是解析逻辑（11/11）与 Windows 路径
> （PASS，`1294x858`，955 种颜色，`WM_CLOSE` 后退出码 0）。
> 这一条在 CI 上跑过两轮才收敛：第一轮证明了 xdotool **找得到**窗口，
> 第二轮才暴露出它**关不掉**。

### 3.19 一项检查会静默消失，而不是被记为跳过

给步骤改名成「75 checks」之后，CI 报的是 `74 checks`。

差的这一项不是某台机器上少跑了什么，而是**这条检查在条件不满足时压根没有被登记**：

```python
if len(drawn) == 3:
    check("the protein ribbon is a different picture from its atoms", ...)
```

无头 runner 的 framebuffer 读回空白，`drawn` 是空的，于是 `if` 不成立，
这一项既不是 PASS 也不是 SKIP，而是**不存在**。
总数从 75 变成 74，而 74 这个数字没有任何人会去质疑——
它看起来就像「这个版本少了一项功能」。

同一份脚本里还有第二处（`if hydrogens and carbons:`），本地姿势文件里两者都有，
所以本地永远走不到 `else`，一直没暴露。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 78 | CI 报 74 项而本地 75 项；差异无处可查 | 两处检查用 `if` 守卫，前置条件不满足时**不登记结果**。少一项和「这项没跑」在输出里长得一样 | 两处都补 `else: skip(..., 原因)`，总数在任何环境下都恒为 75 |

这和 §3.18 的第 76 条是同一类：**恒真或可消失的断言不提供任何信息**。
区别只在于一个恒真、一个恒缺，而后者更难发现，因为它看起来像范围变小了。

**反向验证**：把 `PIXELS_OK` 强制为 `False`（复现 runner 条件）跑一遍——
改之前 `74 checks`，改之后 `61 passed, 0 failed, 14 skipped, 75 checks`，
通过数与 CI 实测的 61 完全一致，多出的那一项明确显示为 SKIP 并写明原因。
真实显示环境下仍是 `75 passed, 0 failed, 0 skipped`。

---

## 4. 复现本报告

```powershell
# Rust
cargo test -p dock-core                        # 106
cargo test -p dock-core --features gpu         # 116
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p dock-core --features gpu --all-targets -- -D warnings

# wheel（CPU 与 GPU 各一次）
python -m maturin build --release -m dock-py\Cargo.toml --out dist
python -m maturin build --release -m dock-py\Cargo.toml --out dist-gpu --features gpu
python -m pip install --force-reinstall --no-deps dist\opendocking-0.1.0-cp38-abi3-win_amd64.whl

# Python（对已安装的 wheel；改过 .py 必须先重建 wheel）
python -m pytest --pyargs opendocking.tests -q        # 38

# 验证脚本
python scripts\workbench_smoke.py               # workbench 两级渲染验证
python scripts\qt_gl_probe.py                   # 6 条 GL 上下文路径
python scripts\workbench_interaction_check.py   # GUI 布局 + 交互 + 行为 + 显示方式，75 项
python scripts\viewport_framing_check.py        # 内容是否真的居中
python scripts\structure_bond_check.py          # 键感知：残基分组、肽键、无跨残基键、RDKit 交叉验证，36 项
python scripts\representation_geometry_check.py # 圆柱 / 双色键 / 条带几何，21 项
python scripts\x11_window_parse_check.py         # 启动检查的 X11 窗口查找与关窗判定，11 项，不需要 X server
python scripts\odgui_launch_check.py            # 真实启动 odgui 子进程并关窗
python scripts\check_doc_encoding.py            # 8 个中文文档的 UTF-8 完整性
python scripts\check_repo_docs.py               # 链接完整性 + 本机路径泄露
cd examples
python check_receptor_donors.py 1crn_receptor.pdb   # 受体供体位点覆盖
python robustness_check.py                      # 畸形输入
python determinism_check.py                     # 确定性与并行
python audit_poses.py                           # pose 物理审计
python diagnose_clash.py                        # 打分函数偏干净还是偏重叠

# 装出来的 console script 必须恰好是两个，且名字正确
odcli --version
odcli info
odgui --check
python -c "import importlib.metadata as m; print(sorted(e.name for e in m.entry_points(group='console_scripts') if 'opendocking' in (e.value or '')))"
# 期望：['odcli', 'odgui']
```

发布目录 `opendocking-gui/` 是上面这套命令的**独立副本**。
验证它自洽（§3.12）的方式是在该目录内部把整套流程重跑一遍：

```powershell
cd opendocking-gui
cargo fmt --all --check
cargo test -p dock-core
cargo test --workspace --features gpu
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features gpu -- -D warnings
python -m maturin build --release -m dock-py\Cargo.toml --out dist
python -m pip install --force-reinstall --no-deps dist\opendocking-0.1.0-cp38-abi3-win_amd64.whl
python -m pytest --pyargs opendocking.tests -q
python scripts\workbench_interaction_check.py
python scripts\structure_bond_check.py
python scripts\representation_geometry_check.py
python scripts\x11_window_parse_check.py
python scripts\odgui_launch_check.py
python scripts\check_doc_encoding.py
python scripts\check_repo_docs.py
```

> `examples/` 与 `scripts/` 在**包外**，不在 `pytest` 的收集范围内。
> 改名或重构时最容易漏掉的就是它们（§3.10 的两条就是这么漏的）——
> 改完务必把上面列出的**全部 10 个 `scripts/` 与 5 个 `examples/` 脚本**重跑一遍。

---

## 5. 结论

**可以据以使用的部分**：Rust 引擎的数值正确性（106/116 测试、两配置零警告）、
CPU/GPU 数值一致性（相对误差 ~1e-7，约一个 f32 ulp；两个不同平台实测，
见 §3.13）、Python 绑定与零拷贝契约、
CLI 全部子命令、workbench 的真实渲染**与全部交互控件**、
畸形输入的安全处理、确定性与并行正确性、真实蛋白的完整预处理链路、
并写的输出能被自己的读取器完整读回。

**尚不能据以做出科学结论的部分**：任何"这个配体会结合到这个位点"的断言——
那需要 §2.1（与 Vina 交叉验证）和 §2.2（回对接），两者都还没做。
本轮修掉的打分函数缺陷（§3.2.1）让能量落在物理上正确的距离，
但**没有**让本工具变成经过验证的亲和力预测器。
