# API 参考

两层 API：Rust crate `dock-core`（引擎）和 Python 包 `odockmcode`（绑定 + 化学 + GUI）。

- [Rust: dock-core](#rust-dock-core)
- [Python: odockmcode](#python-odockmcode)
- [Python: odockmcode.prep](#python-odockmcodeprep)
- [Python: odockmcode.workbench](#python-odockmcodeworkbench)
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
| `AtomKind` | `Hydrophobic` / `Donor` / `Acceptor` / `DonorAcceptor` / `Other` |
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
| `Quaternion` | `from_rotvec` `to_rotvec` `to_matrix` `normalized` `IDENTITY` |
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

常量：`DEFAULT_SPACING = 0.375`、`MAPS_PER_TYPE = 4`、`MAX_GRID_POINTS = 1 << 28`。
枚举 `MapSlot { Shape, HbFromDonor, HbFromAcceptor, Hydrophobic }`。

`MAX_GRID_POINTS` 在**分配之前**检查并报出所需 MB。`spacing = 1e-9` 落在
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
配体大于一个 workgroup、配体为空、没有可用适配器都会**静默回退到 CPU**——
一个给出别的数字的后端比没有后端更糟，所以回退后的结果必须和 CPU 完全一致
（单测 `a_large_ligand_falls_back_to_the_cpu` 断言这一点）。

`LbfgsConfig` 默认：`memory = 10`、`max_iterations = 200`、
`gradient_tolerance = 1e-4`、`energy_tolerance = 1e-6`、
`initial_step = 0.5`、`armijo_c = 1e-4`、`max_step_norm = 4.0`。

`MonteCarloConfig` 默认：`exhaustiveness = 8`、`steps = 70`、
`initial_temperature = 1.2`、`final_temperature = 0.0`、
`mutation_amplitude = 2.0`、`temperature_decay = 0.5`。

`LgaConfig` 默认：`islands = 5`、`population_per_island = 24`、
`generations = 40`、`migration_interval = 5`、`crossover_rate = 0.9`、
`mutation_rate = 0.3`、`variance_weight = 0.15`、局部优化 `max_iterations = 60`。

`OUT_OF_BOX_PENALTY = 1000.0`。

### `docking`

`DockingConfig { mode, monte_carlo, lga, num_modes, rmsd_cutoff, min_contact_distance, num_threads }`
— 注意**没有顶层 `exhaustiveness`**，它在 `monte_carlo` 里。
`DockingConfig::fast()` 用于测试和交互式快速对接。

`min_contact_distance` 默认 `MIN_CONTACT_DISTANCE = 2.0` Å，设为 `0.0` 关闭。
它是一道**安全网**：排斥项自己能工作了，它只是拒绝报告任何物理上不成立的结构。
`0` 表示"关掉它，去看真正的能量极小值在哪"。

`dock(ligand, maps, scoring, config) -> Result<DockingResult>`。
`DockingResult { poses, elapsed_seconds, raw_pose_count, rejected_pose_count, mode, scoring_function }`，
另有 `best()` `energies()` `pose_conformation()` `grid_box()` `ligand()` `maps()`。
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

## Python: `odockmcode`

安装：`pip install odockmcode`（本地 `pip install dist/odockmcode-*.whl`）。

### 顶层导出

```python
from odockmcode import (
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
| `unknown_atom_types` | 无法识别的原子类型**计数**。不报错（否则本工具无法读别的工具产出的 PDBQT），但让它可见 |
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

`odgui` 是 workbench 的独立入口点，**不依赖 `odock` 这个命令名**——
如果本机另一个项目已经占用了 `odock`，工作台照常可用。

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
     seed=None, mode="mc", scoring=None, steps=None) -> DockingResult

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

## Python: `odockmcode.prep`

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

## Python: `odockmcode.workbench`

```python
from odockmcode.workbench import launch, Workbench, MoleculeView, Camera
```

- `launch(receptor=None, ligand=None, poses=None) -> int`：CLI 入口。
- `Camera`：轨道相机。
- `MoleculeView`：球 + 键线的数据容器。
- `app.SphereMesh` / `build_programs(ctx)` / `draw_spheres(...)` / `draw_lines(...)`：
  **与 Qt 解耦**的渲染层，可以脱离 Qt 用独立 moderngl 上下文测试
  （`scripts/workbench_smoke.py` 就是这么做的）。

缺 PyQt6 / moderngl 时 `import odockmcode.workbench` 给出带安装建议的错误。

---

## CLI

```
odockmcode [--version] {prep-receptor,prep-ligand,rec-grid,dock,split,info,workbench}
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
