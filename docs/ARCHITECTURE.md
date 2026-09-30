# 架构

一句话：**Rust 负责所有数值，Python 负责所有化学判断。**

这不是随意的分工。芳香性判定、质子化状态、Gasteiger 电荷、显式极性氢的增删——
这些问题的正确答案取决于化学，猜错的后果是"能跑但结果是错的"，而这类 bug 极难从
对接输出里看出来。RDKit 已经把这些做对了，所以本工具让它做，并让 Rust 只接收
一张确定的表。反过来，网格、梯度、搜索这些是纯数值问题，Python 慢且难并行，
所以放在 Rust 里。

---

## 1. 分层

```
┌────────────────────────────────────────────────────────────────┐
│  opendocking.workbench    PyQt6 + ModernGL 3D 工作台            │
├────────────────────────────────────────────────────────────────┤
│  opendocking.cli          7 个子命令                            │
│  opendocking.prep         RDKit：芳香性 / 质子化 / 电荷 / 极性氢│
│  opendocking.pdbqt_writer 79 列 PDBQT 格式化（唯一实现）        │
├────────────────────────────────────────────────────────────────┤
│  opendocking.core         类型化包装层，零拷贝传 NumPy          │
├────────────────────────────────────────────────────────────────┤
│  opendocking._dockpy      PyO3 扩展（cdylib）                   │
├────────────────────────────────────────────────────────────────┤
│  dock-core               docking grid lbfgs lga                │
│                          gpu (wgpu / WGSL)                     │
│                          pdbqt scoring types                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 一次对接的数据流

```
输入文件
  │
  ├─ receptor.pdb ──RDKit──> 原子表 ──> Rust: Receptor
  │                                              │
  │                            GridBox.precalculate(0.375 Å)
  │                                              ↓
  │                                        GridMaps (只做一次)
  │                                          ≈ stride·nx·ny·nz 个 f32
  │
  └─ ligand.sdf ──RDKit──> (元素, 电荷, 坐标, 键) ──> Rust: Ligand
                                                       │
                                     KinematicTree::from_molecule
                                     （切断可旋转键 → 刚体簇 + 扭转树）
                                                       ↓
   ┌─────────────────────────────────────────────────────────┐
   │  迭代局部搜索（rayon，exhaustiveness 条并行轨迹）           │
   │    每步：ScoringContext::evaluate → (E, ∇E)               │
   │           运动学前向传播 → 原子坐标                        │
   │           网格三线性插值 → 原子梯度                        │
   │           conf_gradient  → DOF 梯度                      │
   │           L-BFGS + Armijo 回退                             │
   └─────────────────────────────────────────────────────────┘
                                                       ↓
                                          原始构象（数百个）
                                                       ↓
                              partition_by_clash（物理可能性）
                                                       ↓
                                    cluster_poses（per-atom RMSD）
                                                       ↓
                                       排序后的 DockingResult
                                                       ↓
                                     PDBQT / XYZ / JSON
```

**三个阶段刻意可分离**：制表昂贵且与配体无关，搜索昂贵由 `exhaustiveness` 控制，
聚类便宜。分开是为了让调用方只重跑变了的部分——20 Å 盒子制表 0.01 s，
但 40 Å 盒子制表会到分钟级，不该因为换个配体就重做。

碰撞分区夹在搜索和聚类**之间**，顺序是有意的：先分区再聚类，
一个不可能的姿势就占不掉一个聚类名额。反过来做的话，
一个 −9 kcal/mol 的重叠姿势会顶掉一个 −1 kcal/mol 的干净姿势。

### 2.1 批量打分路径（GPU / CPU 共用一条）

```
evaluate_population(ligand, maps, scoring, conformations, prefer_gpu)
  │
  ├─ 运动学前向传播（永远在 CPU）
  │    扭转树是递归的，没法搬到 GPU，所以这一步不管走哪条路都要做
  │
  ├─ 分子间能量
  │    ├─ GPU：wgpu compute kernel 批量做 (原子 × 构象) 的三线性插值
  │    └─ CPU：ScoringContext::energy 逐个构象
  │
  └─ 分子内能量（永远在 CPU，精确对邻居表求和）
       └── 两侧相加：inter + intra · scale
```

分子内项拆出来单独在 CPU 上算，是为了让 GPU 路径返回的数和 CPU 路径**逐位相同**
而不只是"差不多"。单测 `the_intramolecular_split_is_the_sum_of_its_parts` 钉住这一点。

回退条件（全部静默，因为回退后的结果必须和 CPU 一致）：

| 条件 | 原因 |
|---|---|
| 配体原子数 > `WORKGROUP_SIZE`（64） | 一个 workgroup 装不下 |
| 种群为空 | 没东西可算 |
| 拿不到适配器 / 设备创建失败 | 没有 GPU |
| 打包失败 / compute pass 失败 | 驱动层错误 |

`PopulationBackend` 会如实报告走了哪条路，回退时附带 `GpuSkip.reason`。

---

## 3. 并行模型

本工具全部并行都走 **rayon 的安全 API**，crate 顶部 `#![forbid(unsafe_code)]`，
所以不存在数据竞争或别名 UB 的可能。

| 位置 | 粒度 | 机制 |
|---|---|---|
| 网格预制表 | 按 **z 切片** | `par_chunks_mut(slab_len)`。一个切片是扁平数组里连续的一段，因此各 worker 拿到互不重叠的可变切片，编译器保证无别名 |
| 迭代局部搜索 | 按 **搜索轨迹** | `exhaustiveness` 条轨迹彼此独立，用 `map` 并行；每条轨迹的随机种子从主种子派生，因此给固定 seed 结果可复现 |
| LGA | 按 **世代内个体** | 岛屿之间独立演化 |
| L-BFGS | **不并行** | 一次下降是严格串行的 |

线程数用 `n_threads == 0` 表示"用满所有核心"。建池失败时（`rayon::ThreadPoolBuilder::build`
返回 `Err`）会**静默回退**到默认池，而不是 panic——预制表失败应该报错，建池失败不该。

---

## 4. 内存布局

### 4.1 网格

```
index = ((ix + nx·(iy + ny·iz)) · STRIDE) + type·4 + slot
STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE = 10 · 4 = 40
```

每点 40 个 `f32`（160 B），其中对该网格实际相关的只有该元素类型的 4 个。
这个布局是**故意**做成平坦且不压缩的：WGSL 的取址可以直接是
`((ix + nx*(iy + ny*iz)) * STRIDE + type*SLOTS + slot)`，
`raw_slice()` 把整个 `Vec<f32>` 原样交给 GPU，不需要重打包。

> 注意：**没有**存径向导数。见 [`SCORING.md` §6.3](SCORING.md)。
> 代价是 GPU 上传的数据量少了一半，收益是梯度是正确的。

### 4.2 GPU 的 per-atom 数据

为适配 downlevel 设备的 4 个 storage buffer 上限，原子类型与 4 个权重打包在一起：

```
ATOM_STRIDE = 8   // [type, w0, w1, w2, w3, pad, pad, pad]
```

权重用 `f32::to_bits()` 存成 `u32`，WGSL 端 `bitcast<f32>` 取回。
绑定布局：

| binding | 类型 | 内容 |
|---|---|---|
| 0 | storage | energies (输出) |
| 1 | storage | coords |
| 2 | storage | atom_data |
| 3 | storage | grid |
| 4 | uniform | `GridParams`（64 字节，含 3 个 `_pad` 以满足 16 字节对齐） |

`STRIDE` 与 `ATOM_STRIDE` 在 WGSL 里是 `const`，与 Rust 侧有单测绑定。
这类常量不一致不会让任何地方报错，只会让每次查表整体偏移——所以值得用测试钉死。

### 4.3 搜索热路径

`ScoringContext` 持有全部可复用缓冲（当前构象的全局坐标、逐原子梯度），
因此**长搜索的内层循环不做任何堆分配**。这在对接这种"同一组数据被调用几十万次"
的场景下，比微优化更有意义。

### 4.4 预制表的截断

`SpatialKernels::cutoff` 限制的是**表面距离**，但内循环手里只有**真实距离**，
所以真实距离的截断是 `cutoff + 2·R`：

```rust
let reach = kernels.cutoff + 2.0 * a.radius;
if r2 > reach * reach { continue; }
```

`GridMaps` 还额外存了一份 `receptor_heavy`（剔除 H 的重原子坐标），
给 `partition_by_clash` 用。**必须用重原子**：极性氢按构造就坐在自己的供体重原子上，
拿它去比会把每个正常的氢键报成一次碰撞。

---

## 5. PyO3 零拷贝契约

| 方向 | 拷贝次数 | 说明 |
|---|---|---|
| Python NumPy → Rust | **0** | `PyArray` 借用，C-contiguous `f64`；视图在调用期间保持引用计数 |
| Rust → Python NumPy | 1（move） | 所有权转移，无额外拷贝 |

因此 `evaluate_conformations` 处理一个 `(100_000, 14)` 的种群时，
内存占用只由**输出**分配决定，输入数组不占额外空间。

调用方责任：传入的数组必须是 **C-contiguous float64**。包装层用
`np.ascontiguousarray` 兜底，形状不合法时抛 `ValueError` 而不是静默重解释。

`dock-py` 里的一个曾经的真实 bug：`from_arrays` 把 RDKit 给出的键**追加**到了
距离感知键表之上，于是每条键有两条记录 → 扭转翻倍 → 簇图成环 → 运动学树
构建时 `expect` panic。release profile 是 `panic = "abort"`，所以这不只是抛异常，
而是**直接杀死宿主 Python 进程**（退出码 `0xC0000409`）。现在 `from_arrays`
会先 `clear()` 再用调用者给的图，并跳过重复键。

---

## 6. 错误处理

Rust 侧统一 `Result<T, DockError = DockError>`，变体：
`Io` `Parse` `InvalidMolecule` `Grid` `Gpu` `Convergence` `Parameter`。
每个变体都带**定位信息**（行号、参数名 + 取值 + 原因），
所以 Python 侧拿到的错误消息能直接告诉用户改什么。

参数校验刻意写成"否定式比较"：

```rust
if !(max[k] > min[k]) { /* 拒绝 */ }
if !(spacing > 0.0) || spacing > 1.0 { /* 拒绝 */ }
```

因为对 `f64`，只有 `!(a > b)` 这种形式会连 `NaN` 一起拒掉；
`a <= b` 会让 `NaN` 顺利通过，然后在后面除以它、分配出一个荒谬的网格。
这两处都加了 `#[allow(clippy::neg_cmp_op_on_partial_ord)]` 并附注释说明是有意的。

Python 侧：CLI 捕获所有异常并转成退出码，**不做栈回溯**——
用户要的是"box 太小"而不是 Python traceback。

### 6.1 能量与梯度必须来自同一次求值

`ScoringContext::evaluate` 一次返回 `(总能量, DOF 梯度)`，两者出自同一次遍历。
把"算能量"和"算梯度"拆成两条路径几乎必然漂移：网格权重项漏了、
分子内项的 `scale` 只加在一边、或者分子间/分子内用了不同的半径。
一旦漂移，**能量和梯度对不上**，L-BFGS 会朝错的方向走，
而症状是"能跑完、能量还挺负、姿势不收敛"——从输出里看不出来。

这条链以前没有测试覆盖：`kinematics.rs` 的 `conf_gradient` 有限差分测试喂的是
坐标的**线性泛函** `E = Σ gᵢ·rᵢ`，它的梯度按定义就等于 `conf_gradient` 的链式法则结果，
所以它只能证明**回代代数**对，证明不了**打分管线产出的原子梯度**对。

现在补上了 `search::tests::the_analytic_gradient_is_the_gradient_of_the_reported_energy`：
在 butane 和 hexane（1 和 4 个扭转）上，对每个 DOF 做中心差分并与解析梯度比对。
这把「三线性权重导数 → 原子梯度 → `conf_gradient`」整条链锁住了。

### 6.2 workbench 的三条硬约束

都是被真实 bug 教出来的，每一条都有回归检查。

**1. 身份必须显式，不能靠文件名。** `MoleculeView` 带一个 `role`
（`receptor` / `ligand` / `pose`）。对接结果来自**每次都不同名的临时文件**，
所以按 `name` 匹配无法区分"当前的姿势"和"上一轮的姿势"——结果是连续两次 Dock
之后旧结果一直留在画面上，而且"隐藏配体"勾选框再也藏不掉它们。

**2. 耗时的活不能跑在 GUI 线程上。** 网格预制表一度在 `start_docking` 里
同步执行：状态文字设了，但事件循环没法转去重绘它，用户在制表期间**看不到任何反馈**。
现在制表与搜索都在 `QThread` 上；`closeEvent` 在关窗前停线程，否则
"QThread: Destroyed while thread is still running" 会 abort 整个进程。

**3. 渲染的输入必须先规范化。** PDBQT 的原子类型**不是**"元素 + 极性后缀"：
`A` 本身就是芳香碳。一条"取首个字母"的规则会把它变成虚构元素，
颜色表查不到就回退到 view 颜色，于是苯环被画成绿色而同一分子的脂肪碳是灰色。
`PDBQT_TYPE_ELEMENT` 是一张显式表，未知类型才回退到那条（此时已无更好的选择）。

> 写回侧的对应约束：`REMARK VINA RESULT` 必须落在 `MODEL..ENDMDL` **内部**。
> 写在 `MODEL` 之前的内容不属于任何 pose，按 model 切分的读取器看不到它，
> 结果是所有 pose 的能量都变成 0.00，而 `argmin` 在全等值上返回第一行——
> 界面看起来正常，"最佳 pose"却是随机的。

---

## 7. 化学在 Python，数值在 Rust

| 问题 | 归属 | 理由 |
|---|---|---|
| 芳香性 | RDKit | 猜错会直接改变键级、氢键、原子类型 |
| 质子化状态 | RDKit | 同上，且影响氢键供体判定 |
| Gasteiger 电荷 | RDKit | 成熟的公开算法 |
| 显式极性氢的增删 | RDKit | 需要 SMARTS 匹配 |
| 原子类型指派 | `types.rs` | 以 PDBQT 类型为权威，读进来的就是定下来的 |
| 网格 / 插值 / 梯度 | Rust | 纯数值 |
| 搜索 | Rust | 纯数值 + 需要并行 |

`dock-core` 里**没有**任何环芳香性感知、没有 SMARTS、没有 pKa 模型。
它相信调用方给的表是化学上正确的，并据此做数值计算。
这个边界是刻意划的：一旦 Rust 开始"顺手猜一下"化学，两层就会各自猜一遍，
而且猜得不一样。

---

## 8. 构建与分发

```
Cargo.toml                 workspace: dock-core, dock-py
dock-core/Cargo.toml       [features] gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]
dock-py/Cargo.toml         cdylib + rlib
dock-py/pyproject.toml     maturin; python-source = "python"
```

**标准布局是 `dock-py/python/`**，不是顶层 `python/`。
maturin 1.15 对带 `..` 的 `python-source` 只做存在性校验、实际收集为空，
会**静默**产出一个只含 `.pyd` 而没有任何 `.py` 的 wheel。踩过这个坑。

`pyproject.toml` 里排除了 `__pycache__`——字节码缓存是构建产物，
打进 wheel 会撑大体积，还可能让改过的模块被过期 `.pyc` 遮蔽。

开发循环里 `target/release/dockpy.dll` **不能**直接改名当扩展用：
它的导出符号集不完整。正确做法是 `maturin develop` 或构建 wheel 后从
wheel 里取 `_dockpy`。
