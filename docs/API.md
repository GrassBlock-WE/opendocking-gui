# API 参考

两层 API：Rust crate `dock-core`（引擎）和 Python 包 `opendocking`（绑定 + 化学 + GUI）。

- [Rust: dock-core](#rust-dock-core)
- [Python: opendocking](#python-opendocking)
- [Python: opendocking.prep](#python-opendockingprep)
- [Python: opendocking.workbench](#python-opendockingworkbench)
- [CLI](#cli)

---

## Rust: `dock-core`

`#![forbid(unsafe_code)]`，Edition 2021。

```toml
[dependencies]
dock-core = { path = "../dock-core" }              # CPU
dock-core = { path = "../dock-core", features = ["gpu"] }   # + wgpu 计算着色器
```

### 顶层 re-export（`dock_core::prelude`）

```rust
use dock_core::prelude::*;
```

导出：`dock` `DockingConfig` `DockingResult` `Pose` `SearchMode`
`DockError` `Result` `GridBox` `GridMaps`
`Conformation` `KinematicTree`
`Ligand` `Receptor`
`ScoringFunction` `VinaScoring` `VinaWeights`
`LbfgsConfig` `LbfgsOutcome` `LgaConfig` `MonteCarloConfig`
`Atom` `AtomKind` `AtomType` `Bond` `Element` `Molecule` `Vec3`

### 最小可运行示例

```rust
use dock_core::prelude::*;

let receptor = Receptor::from_pdbqt("rec.pdbqt")?;
let box_ = GridBox::centered([0.0, 0.0, 0.0], [22.0, 22.0, 22.0])?;
let scoring = VinaScoring::new();

let maps = receptor.precalculate(&box_, &scoring, 0.375, 0)?;
let ligand = Ligand::from_pdbqt("lig.pdbqt")?;

let config = DockingConfig { num_modes: 9, ..Default::default() };
let result = dock(&ligand, &maps, &scoring, &config)?;

for pose in &result.poses {
    println!("{:.2}", pose.energy);
}
# Ok::<(), DockError>(())
```

### `types`

| 类型 | 说明 |
|---|---|
| `Element` | 12 种元素枚举（`COUNT = 12`），`from_symbol`、`symbol`、`interaction_radius`、`covalent_radius`、`is_metal`、`is_polar_hydrogen` |
| `AtomType` | 18 种 PDBQT 原子类型（`COUNT = 18`），`from_token`、`token`、`default_atom_type(Element)` |
| `AtomKind` | 5 种相互作用类（`COUNT = 5`）：`Hydrophobic` / `Donor` / `Acceptor` / `DonorAcceptor` / `Other` |
| `Atom` | `serial` `coord` `element` `atom_type` `kind` `charge` `ring` |
| `Bond` | `i` `j` `rotatable`；别名 `MoleculeBond` |
| `Molecule` | `atoms` `bonds`；`from_atoms`、`perceive_bonds`、`assign_ring_membership`、`assign_vina_atom_kinds`、`is_empty`、`total_charge`、`bounding_box`、`centroid` |
| `Vec3` | `[f64; 3]` 别名 |

辅助：`grid_type_index(Element) -> usize`、`grid_type_name(usize) -> &'static str`、
`dist3`、`dist2_3`。`GRID_TYPE_COUNT = 10`。

`Element::interaction_radius()` 是**每元素 XS 半径**（C 1.9 / N 1.75 / O 1.6 /
S 2.0 / … Å，H 为 0）。这个常数决定每一项打分落在间距轴的哪个位置，
**不是可调参数**——把它改小 1.5 Å 会让全部吸引项落进碰撞区。
推导与踩过的坑见 [`SCORING.md` §1.1](SCORING.md)。

### `pdbqt`

| 函数 | 说明 |
|---|---|
| `parse_pdbqt(text) -> Result<ParsedStructure>` | 从字符串解析 |
| `read_pdbqt(path) -> Result<ParsedStructure>` | 从文件读取 |
| `read_docked_poses(path) -> Result<Vec<DockedPose>>` | 读多 MODEL 文件 |
| `write_pose<W: Write>(...)` | 写单个构象 |
| `write_poses(...)` | 写多个构象 |
| `write_single_pose(...)` | 便捷封装 |
| `write_result(...)` | 写整个 `DockingResult` |

`ParsedStructure` 含 `atoms` / `bonds` / `torsions`。
PDBQT 的 **BRANCH 只写本簇原子，不做嵌套**（`collect_cluster` 负责收集）。

### `kinematics`

| 符号 | 说明 |
|---|---|
| `KinematicTree::from_molecule(&Molecule, &[(usize, usize)]) -> Result<Self>` | 切可旋转键建刚体簇；**收缩不动点**保证扭转集合是割集 |
| `KinematicTree::coordinates(&Conformation) -> Vec<Vec3>` | 前向运动学 |
| `body_frames` / `global_frames` | 两级 FK 坐标系 |
| `torsion_axis_in_world(...) -> Option<...>` | 无扭转时返回 `None`（不连通片段） |
| `conf_gradient(&Conformation, &[Vector3<f64>]) -> Vec<f64>` | 原子梯度 → DOF 梯度 |
| `subtree_atoms(cluster)` / `subtree_table()` | 运动学树的子表 |
| `Conformation { position, orientation, torsions }` | `identity(n)` `from_slice` `ndof` `randomize` |
| `Quaternion` | `from_rotvec` `to_rotvec` `to_matrix` `normalized`；`IDENTITY` 是单位四元数 `w = 1.0`、`x = y = z = 0.0` |
| `rotvec_to_matrix` / `so3_right_jacobian` / `skew` / `wrap_torsion` | SO(3) 工具 |

`MIN_INTRA_BOND_DISTANCE = 4`——分子内项的最小图距。

### `scoring`

```rust
pub trait ScoringFunction: Send + Sync {
    fn name(&self) -> &'static str;
    fn atom_weights(&self, atom: &Atom) -> [f32; MAPS_PER_TYPE];
    fn spatial_kernels(&self) -> SpatialKernels;
    fn pair_energy(&self, a: &Atom, b: &Atom, d: f64) -> f64;
    fn pair_gradient(&self, a: &Atom, b: &Atom, d: f64) -> f64;
    fn intramolecular_scale(&self) -> f64;
    fn description(&self) -> String { /* 有默认实现 */ }
}
```

该 trait 是 **object-safe**，`dock()` 接受 `&dyn ScoringFunction`。

空间函数（都是 `pub`，返回 `(值, 导数)`）：
`gaussian_term` `repulsion_term` `hbond_term` `hydrophobic_term`。
其他：`weights_for_kind` `receptor_slots` `surface_distance` `interaction_radius`。
常量：`STERIC_LIMIT = 0.0` `VINA_CUTOFF = 8.0`。

### `grid`

| 符号 | 说明 |
|---|---|
| `GridBox::new(min, max) -> Result<Self>` | **会拒绝 NaN 与 inf**（用 `!(max > min)` 与 `is_finite()`） |
| `GridBox::centered(center, size)` | |
| `center()` `size()` `contains(p)` | |
| `estimate_dims(&GridBox, spacing) -> [usize; 3]` | 不分配就能估算 |
| `GridMaps::precalculate(&Molecule, &GridBox, &dyn ScoringFunction, spacing, n_threads)` | `n_threads == 0` 表示用满所有核心 |
| `interpolate(type_index, &weights, p) -> Option<f64>` | 越界返回 `None` |
| `interpolate_with_gradient(...) -> Option<(f64, Vec3)>` | 精确解析梯度 |
| `raw(type_index, slot, ix, iy, iz)` / `raw_slice()` | 可视化 / GPU 上传 |
| `receptor_atoms() -> &[[f64; 3]]` | 制表所用的**重原子**坐标（已剔除 H）。碰撞过滤用这个：拿极性氢去比会把每个正常氢键报成碰撞 |
| `write_autodock_map_files(dir)` | 写 `.map` |
| `map_stride()` | `GRID_TYPE_COUNT * MAPS_PER_TYPE`（`const fn`） |

常量：`DEFAULT_SPACING = 0.375`、`MAPS_PER_TYPE = 4`、`MAX_GRID_POINTS = 107,374,182`。
枚举 `MapSlot { Shape, HbFromDonor, HbFromAcceptor, Hydrophobic }`。

**`MAX_GRID_POINTS` 不是内存上限，而是 `energy.wgsl` 里 `idx * STRIDE` 的 `u32` 索引上限。**
最大可寻址点数 = `u32::MAX / (GRID_TYPE_COUNT · MAPS_PER_TYPE)` = 4,294,967,295 / 40。
旧值 `1 << 28` 是按内存预算定的，比内核能寻址的范围大 2.5 倍——**旧文档承诺了一个
引擎兑现不了的上限**：GPU 后端算不出来的规模，`backend` 字段会答「跑了 CPU」，而
调用方看到的是一次成功的运行，无从知道自己在用一个 GPU 够不着的规模。
新值在当前步长下是 16.0 GiB，`grid.rs` 里有两条 `const` 断言把它钉在这个界上，
**而且界跟着 `GRID_TYPE_COUNT` 或 `MAPS_PER_TYPE` 一起动**，不会留下过期数字。

`MAX_GRID_POINTS` 在**分配之前**检查并报出所需 MB。**这个收窄是行为变更，不是一次
整理**：0.375 Å 下 170 Å 立方（94,196,375 点）仍然可建，**180 Å（111,284,641 点）
现在被拒绝**，而在一台 64 GiB 的机器上那些网格过去是能跑的。只在 CPU 上打分、且
机器内存足够的调用方，原来能建的 16–40 GiB 网格现在会被拒绝——这是刻意的取舍。
`spacing = 1e-9` 落在
`(0, 1]` 区间内但会要求 2×10¹⁰ 点的轴，维度乘积溢出后 `vec!` 触发 capacity
overflow，而 release profile 是 `panic = "abort"`——那会**杀掉宿主进程**
（Python 调用方的解释器直接消失，退出码 `0xC0000409`），不是抛异常。

### `search`

```rust
// 局部
minimize(start, &LbfgsConfig, &mut energy_fn) -> (Conformation, LbfgsOutcome)

// 全局
monte_carlo::search(&Ligand, &GridMaps, &dyn ScoringFunction, &MonteCarloConfig)
    -> Result<Vec<Pose>>
lga::evolve(&Ligand, &GridMaps, &dyn ScoringFunction, &LgaConfig) -> Result<Vec<Pose>>
```

`ScoringContext` 是搜索热路径的可复用求值器：

| 方法 | 说明 |
|---|---|
| `evaluate(&conf) -> (f64, Vec<f64>)` | **总能量 + 关于 DOF 的解析梯度**。两者必须来自同一次求值，否则 L-BFGS 会朝错的方向走 |
| `energy(&conf) -> f64` | 只有能量 |
| `apply(&conf)` | 只跑运动学，把全局坐标留在 `current_coords()` |
| `intramolecular_energy_at_current_coords()` | 分子内项；拆出来是为了让批量 GPU 路径能取分子间那一半、分子内这一半在 CPU 上算，再精确重组 |
| `last_outside()` | 上一次求值是否有原子跑出网格 |

批量入口 `evaluate_population(ligand, maps, scoring, conformations, prefer_gpu)`
返回 `(energies, PopulationBackend, Option<GpuSkip>)`，`Cpu` / `Gpu { adapter }`。

**上报契约**（`dock-core/src/search/mod.rs`，由 `a_decline_is_always_self_explaining`
钉进 `cargo test`，`scripts/gpu_cpu_parity_check.py:513-539` 在运行时钉同一件事）：
**GPU 跑了 XOR 给出了一个非空理由说明它没跑，从不两者兼具、从不两者皆无。**
配体大于一个 workgroup、配体为空、没有可用适配器、以及**构建本身没有 `gpu` feature**
这四种情况都**回退到 CPU 并带回一个理由**——所以它们**不是静默回退**。另一半方向：
`prefer_gpu = false` 的调用**必须不报任何回退**，否则那是一次没发生的回退。

**这个契约在本文件上一版里被写成了「静默回退到 CPU」，那是一份给缺陷发的许可证**：
它把「不给理由」写成了规格，于是实现与规格一致，而调用方无法区分「按要求用 CPU」
和「你要的 GPU 被丢掉了」。一个给出别的数字的后端比没有后端更糟，所以回退后的结果
必须和 CPU 完全一致（单测 `a_large_ligand_falls_back_to_the_cpu` 断言这一点）——
**但「结果一致」从来不等于「可以不说为什么」**。见 `VERIFICATION.md` 缺陷 286。

`LbfgsConfig` 默认：`memory = 10`、`max_iterations = 200`、
`gradient_tolerance = 1e-4`、`energy_tolerance = 1e-6`、
`initial_step = 0.5`、`armijo_c = 1e-4`、`max_step_norm = 4.0`。

`MonteCarloConfig` 默认：`exhaustiveness = 8`、`steps = 70`、
`initial_temperature = 1.2`、`final_temperature = 0.0`、
`mutation_amplitude = 2.0`、`temperature_decay = 0.5`。

`LgaConfig` 默认：`islands = 5`、`population_per_island = 24`、
`generations = 40`、`migration_interval = 5`、`crossover_rate = 0.9`、
`mutation_rate = 0.3`、`variance_weight = 0.15`、局部优化 `max_iterations = 60`。

`OUT_OF_BOX_PENALTY = 1000.0`。**这个数字本身不是罚项** —— 罚项是
`OUT_OF_BOX_PENALTY × out_of_box_violation_per_axis(p, box)`：按**每超出的一埃**、
按**轴**算，再对三个面求和；角落同时越界按两面各收一次。所以一个原子在 x 方向
超出 2 Å，收 2000，不是 1000。梯度就是这条斜坡自己的导数，因此
`dE/dx = +1000.000000`（每一个悬挑量都一样），`−∇E` 指向盒内。
定义与实测见 `docs/SCORING.md` §4.2。

### `docking`

`DockingConfig { mode, monte_carlo, lga, num_modes, rmsd_cutoff, min_contact_distance, num_threads }`
— 注意**没有顶层 `exhaustiveness`**，它在 `monte_carlo` 里。
`DockingConfig::fast()` 用于测试和交互式快速对接。

`min_contact_distance` 默认 `MIN_CONTACT_DISTANCE = 2.0` Å，设为 `0.0` 关闭。
它是一道**安全网**：排斥项自己能工作了，它只是拒绝报告任何物理上不成立的结构。
`0` 表示"关掉它，去看真正的能量极小值在哪"。
同一个字段在 Python 侧的 `dock()` 里叫同名关键字 `min_contact_distance`，
默认 `None` 表示不动引擎默认值。见下面 `## Python` 一节。

`dock(ligand, maps, scoring, config) -> Result<DockingResult>`。
`DockingResult { poses, elapsed_seconds, raw_pose_count, rejected_pose_count, unknown_atom_types, mode, scoring_function }`，
另有 `best()` `energies()` `pose_conformation()` `grid_box()` `ligand()` `maps()`。
`unknown_atom_types` 由 `GridMaps` 携带而来，而不是每次从 `Receptor` 现算——原因见下面 `Receptor` 一节那一行的说明。
`partition_by_clash(poses, receptor_heavy, min_distance) -> (clean, clashing)` 单独暴露。
`cluster_poses(...)` 也单独暴露，可单独复用。

聚类**之前**先做碰撞分区，这样一个不可能的姿势不会占掉一个聚类名额，
而一个被拒绝的姿势也永远不会静默地被报告出来。

### `error`

`DockError::{Io, Parse, InvalidMolecule, Grid, Gpu, Convergence, Parameter}`，
每种都带上下文字段。构造器：`molecule(msg)` `parse(line, msg)` `param(name, value, reason)`。

### `gpu`（feature `gpu`）

`GpuScorer`（按需增长的输出 buffer）、`gpu_status() -> GpuStatus`。
WGSL 在 [`energy.wgsl`](../dock-core/src/gpu/energy.wgsl)。

---

## Python: `opendocking`

安装：`pip install opendocking`（本地 `pip install dist/opendocking-*.whl`）。

### 顶层导出

```python
from opendocking import (
    Receptor, GridBox, GridMaps, Ligand, DockingResult,
    dock, score_conformation, evaluate_conformations,
    load_receptor, load_ligand, auto_box,
    gpu_status, available_backends, scoring_functions, engine_version,
    SCORING_FUNCTIONS,
)
```

`SCORING_FUNCTIONS == ("vina", "vinardo")`。

### `Receptor`

| 成员 | 说明 |
|---|---|
| `Receptor.from_pdbqt(path)` / `.from_pdbqt_str(text)` | 构造 |
| `num_atoms` | 原子数 |
| `num_polar_hydrogens` | 显式极性氢数；**为 0 通常意味着受体准备时把极性氢合并掉了**，会在每个 Ser/Thr/Tyr 上损失氢键 |
| `unknown_atom_types` | 无法识别的原子类型**计数**。不报错（否则本工具无法读别的工具产出的 PDBQT），但让它可见。**这是制表之前的视角**：手里只有一个 `Receptor`、还没 `precalculate` 的时候，这是唯一能问的地方。一旦制表完成，受体就被丢掉了（`Receptor` 不再参与后续任何计算），所以同一个计数被搬到了 `GridMaps` 上，并出现在 `DockingResult.unknown_atom_types` 里——想判断"这次跑出来的能量是不是低估了受体"，要读的是结果上的那个，不是这一个 |
| `center` / `bounds` | 几何信息 |
| `estimate_memory_mb(box_, spacing=0.375)` | 预估内存 |
| `precalculate(box_, scoring="vina", spacing=0.375) -> GridMaps` | 制表 |

`spacing` 的校验：`0.0` 表示"用默认值"，**负数与 NaN 直接报错**。
把非法值静默替换成默认值会让一个打错的参数看起来像生效了。

### `GridBox`

`GridBox(min_corner, max_corner)` 或 `GridBox.from_center_size(center, size)`。
属性：`min_corner` `max_corner` `center` `size`；方法 `contains(point)`。
`min` 含、`max` 不含。

### `GridMaps`

`dims` `spacing` `num_points` `memory_mb` `box` `raw_data` `write_map_files(dir)`。

`raw_data` 是一个只读的 `(num_points, 40)` float32 视图，可直接交给
numpy / 现代 GL 上传，不需要自己解析 `.map` 文件。列布局见
[`SCORING.md` §5.2](SCORING.md)。

> 网格与打分函数绑定。用 `vina` 制的表不能配 `vinardo` 读，会给出错误答案。

### `Ligand`

| 成员 | 说明 |
|---|---|
| `from_pdbqt` / `from_pdbqt_str` | 从 PDBQT 读 |
| `from_arrays(elements, charges, coords, bonds=None, atom_names=None)` | RDKit 前端走的路径 |
| `num_atoms` `num_torsions` `num_dof` `radius` | 基本性质；`num_dof = 6 + num_torsions` |
| `atom_kinds` | 每个原子的交互类别（字符串列表） |
| `reference_coords` | `(n, 3)` float64 |

### 独立命令 `odgui`

`odgui` 是 workbench 的独立入口点，**不依赖 `odcli`**——
只装了工作台也能单独启动，反之亦然。

```powershell
odgui                                        # 空窗口
odgui -r receptor.pdbqt -l ligand.pdbqt -p poses.pdbqt
odgui --check                                # 只检查能否启动，不开窗口
```

`--check` 输出引擎版本与 GPU 状态，缺 GUI 依赖时返回 3 并打印
缺哪个包、该敲什么命令——不会抛 traceback。这条路径不在模块顶层 import
Qt / modernGL，因此在没有图形栈的机器上也能运行。

File 菜单提供 `Load receptor…` `Load ligand…` `Open poses…` `Dock now`
（Ctrl+R / Ctrl+L / Ctrl+P / Ctrl+D），以及 `Quit`（Ctrl+Q）。
**没有笼统的 "Open structure"**：角色不同，显示方式与勾选框归属都不同，
必须在打开前就选定。

| 成员 | 说明 |
|---|---|
| `num_poses` `energies` `best_energy` `intermolecular_energies` `rmsds` | |
| `elapsed_seconds` `raw_pose_count` `scoring_function` | |
| `rejected_pose_count` | **非 0 意味着搜索一个物理上成立的姿势都没找到**，引擎回退报告了最好的那几个。几乎总是"盒子里是实心蛋白而不是口袋"或 `exhaustiveness` 太低 |
| `unknown_atom_types` | 受体里无法识别的原子类型数，由 `GridMaps` 带过来。**非 0 意味着跑成功了、但能量低估了受体**：这类原子仍贡献形状项，却丢掉了氢键与疏水特征，而姿势数量看不出这件事。数值取自 `GridMaps`：`#[serde(default)]`，所以**计数出现之前写出的 map 文件反序列化为 0**，而 0 不等于"这个受体的原子全都被认出来了"——它只意味着"至少这么多没被认出" |
| `pose_coords(i=0)` `all_pose_coords()` | 坐标 |
| `pose_conformation(i=0)` | 该姿势的 DOF 向量。可以拿去 `score_conformation` 重新打分、查梯度，或继续优化 |
| `write_pdbqt(path)` `write_xyz(path)` `summary()` | 输出 |

> **不要用 `|grad|` 判断一个返回姿势收敛了没有。** 网格是三线性插值，只有 C⁰，
> 而实测**每个返回姿势都有原子正好坐在 cell 面上**，所以 `|grad|` 通常是 2–6，
> 且与中心差分符号相反——但沿任一方向都走不动能量，它们是真极小点。
> 要判断极小性请沿线搜索（见 [`LIMITATIONS.md` §4](LIMITATIONS.md)）。

### 函数

```python
dock(ligand, maps, *, exhaustiveness=8, num_modes=9, rmsd_cutoff=1.0,
     seed=None, mode="mc", scoring=None, steps=None,
     min_contact_distance=None) -> DockingResult

score_conformation(ligand, maps, conformation, scoring="vina")
    -> (float, np.ndarray)          # 总能量 + 关于 DOF 的解析梯度

conformation_coordinates(ligand, conformation) -> np.ndarray
    -> (n_atoms, 3)                 # 走运动学，不查表、不需要 maps

evaluate_conformations(ligand, maps, conformations, scoring="vina",
                       use_gpu=False, report_backend=None) -> np.ndarray

load_receptor(path)     # .pdbqt 直接读，其他走 RDKit
load_ligand(path)       # .pdbqt 直接读，.sdf/.mol2/.mol/.smi 走 RDKit
auto_box(receptor, ligand, padding=4.0) -> GridBox

gpu_status() -> {"compiled": bool, "available": bool}
available_backends() -> list[str]
scoring_functions() -> list[str]
engine_version() -> str
```

`min_contact_distance` 与上面 Rust 段的 `DockingConfig::min_contact_distance`
是**同一个字段**，`None` 表示不动引擎默认的 2.0 Å，`0.0` 关闭过滤、报告原始能量极小值。
这一段以前根本没提它，而 `dock()` 也没有这个关键字——引擎读这个字段，Python 却改不了，
所以每一次 Python `dock()` 都被静默固定在 2.0 Å。现在两边都写在这里，
`scripts/core_check.py` 会核对本段与 `dock()` 的签名是否一致。

`report_backend` 是一个**会被填写的 dict**，不是 bool：

```python
info = {}
energies = evaluate_conformations(lig, maps, pop, use_gpu=True, report_backend=info)
info["backend"]          # "cpu" 或 "gpu"
info["adapter"]          # GPU 名；CPU 时为 None
info["gpu_skip_reason"]  # 回退原因；成功时为 None
info["num_conformations"]
```

`use_gpu` **默认 False**。GPU 端用 `f32` 累加，与 CPU 差约 1e-6，
让默认值随硬件变化会破坏跨机可复现性，所以要显式开启。

`conformation` 的布局是 `[tx, ty, tz, θx, θy, θz, τ₀, …]`，长度必须等于
`ligand.num_dof`，否则抛 `ValueError`。含 NaN / Inf 也会抛错——
静默返回一个惩罚值会让 NaN 传播到整条搜索里而看不出来。

---

## Python: `opendocking.prep`

RDKit 化学感知预处理。**这是把化学判断留在 Python 侧的理由**：芳香性、质子化、
电荷、Gasteiger 系数都由 RDKit 算，Rust 引擎只接收一张表。

| 函数 | 说明 |
|---|---|
| `prepare_ligand(path, keep_hydrogens=False)` | 返回 `(elements, charges, coords, bonds, names)` |
| `prepare_receptor(path, keep_waters=False, keep_heterogens=True)` | 返回 PDBQT 文本 |
| `pdbqt_atom_type(atom)` | 元素 + 邻居 → PDBQT 类型 |

关键行为：

- 极性氢通过 `mol.GetSubstructMatches(donor)` 取**索引**后逐个添加。
- 删除非极性氢后必须 `UpdatePropertyCache(strict=False)`，否则隐式氢计数丢失，
  所有羟基氧的 `GetTotalNumHs() == 0`，于是**没有任何配体有 donor**。
- 原子行统一走 `pdbqt_writer.format_atom_line`（79 列）。不要另写格式化器。

---

## Python: `opendocking.workbench`

```python
from opendocking.workbench import launch, MoleculeView, Camera
```

`Workbench` used to appear on this line and **does not exist**. The window class
is `app.MainWindow` and the entry point is `launch`; `__all__` promised a name
that no module defines, so `import *` raised `AttributeError` and the direct
import raised `ImportError`. It is removed rather than created, because the only
real binding would be an eager `from .app import MainWindow as Workbench`, and
that would make PyQt6 and moderngl hard requirements of `odcli sites` / `info`
/ `workbench` -- which breaks the invariant recorded at `cli.py:885-886`, that
the workbench package imports only the standard library plus numpy at module
scope and defers the GUI import to a call inside `launch`. Every other
occurrence of the string "Workbench" in this repository is the window *title*.

- `launch(receptor=None, ligand=None, poses=None) -> int`：CLI 入口。
- `Camera`：轨道相机。
- `MoleculeView`：球 + 键线的数据容器。
- `app.SphereMesh` / `build_programs(ctx)` / `draw_spheres(...)` / `draw_lines(...)`：
  **与 Qt 解耦**的渲染层，可以脱离 Qt 用独立 moderngl 上下文测试
  （`scripts/workbench_smoke.py` 就是这么做的）。

缺 PyQt6 / moderngl 时 `import opendocking.workbench` 给出带安装建议的错误。

---

## CLI

```
odcli [--version] {prep-receptor,prep-ligand,rec-grid,dock,split,info,workbench}
```

选项名用**下划线**（`--center_x`），不是连字符。

```
info            [--json]

prep-receptor   -r/--receptor PATH  -o/--output PATH
                [--keep-waters] [--drop-heterogens]

prep-ligand     -l/--ligand PATH  [-o/--output PATH] [--keep-hydrogens]

rec-grid        -r/--receptor PATH  -o/--output DIR
                [--scoring {vina,vinardo}] [--spacing F]
                --center_x F --center_y F --center_z F
                --size_x F   --size_y F   --size_z F

dock            -r/--receptor PATH  -l/--ligand PATH|DIR  [-o/--output PATH]
                [-e/--exhaustiveness N]  [-m/--num_modes N]
                [--rmsd-cutoff F] [--scoring {vina,vinardo}] [--spacing F]
                [--mode {mc,lga,both}] [--seed N] [--steps N] [--json]
                --center_x F --center_y F --center_z F
                --size_x F   --size_y F   --size_z F

split           -i/--input PATH  -o/--output-dir DIR

workbench       [-r/--receptor PATH] [-l/--ligand PATH] [-p/--poses PATH]
```

`dock` / `rec-grid` 的 6 个 box 参数**全部必填**——不做自动默认，
因为从受体自动推出来的 box 要么巨大到撑爆内存，要么根本没有搜索意义。

退出码：`0` 成功，`1` 出错，`2` 参数错误，`3` 缺 GUI 依赖，`130` 被中断。
