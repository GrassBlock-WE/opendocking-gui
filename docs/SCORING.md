# 打分函数：推导 ↔ 实现对照

本文把本工具用的经验打分函数从公式推到代码行，标出每一处**有意偏离 AutoDock Vina**
的地方以及偏离的理由。所有偏离都是有意的，不是疏漏。

对应实现：

| 内容 | 文件 |
|---|---|
| 空间形状函数与权重 | [`dock-core/src/scoring.rs`](../dock-core/src/scoring.rs) |
| 网格预制表与插值 | [`dock-core/src/grid.rs`](../dock-core/src/grid.rs) |
| 原子分类 | [`dock-core/src/types.rs`](../dock-core/src/types.rs) |
| GPU 求值路径 | [`dock-core/src/gpu/energy.wgsl`](../dock-core/src/gpu/energy.wgsl) |

---

## 1. 原子对的几何量

对两个原子 `i`、`j`，打分只依赖**表面距离**：

```
d(rᵢ, rⱼ) = ‖rᵢ − rⱼ‖ − (R_i + R_j)
```

其中 `R` 是元素的有效半径（`types.rs::Element::interaction_radius`）：

| 元素 | `R` (Å) | | 元素 | `R` (Å) |
|---|---|---|---|---|
| C | 1.90 | | P | 2.10 |
| N | 1.75 | | S | 2.00 |
| O | 1.60 | | Cl | 1.948 |
| F | 1.545 | | Br | 2.220 |
| I | 2.350 | | Met | 1.20 |
| 无法识别的 token | 1.90 | | H（PDBQT 只保留极性氢） | 0.0 |

这些是**每元素的 XS 半径**，即参考非键接触距离的一半。C 取 1.9 Å，于是 C···C
落在 3.8 Å 的范德华接触上；O 取 1.6 Å，于是 O···O 落在 3.2 Å，而氢键项的最大值
在 `d ≤ −0.5`，对应 2.7 Å——正是晶体学上强氢键的重原子间距范围（2.6–2.9 Å）。

给 H 半径 0 的理由：PDBQT 里留下的极性氢**本来就应该坐在它的供体重原子上**，
把它当成一个独立球会让每个正常的氢键都报成重叠。

> 预制表内循环里 `d = r2.sqrt() - 2.0 * a.radius`，两个探针原子都按受体原子的
> 半径算；打分路径上则是 `rᵢ + rⱼ`（`scoring.rs::surface_distance`）。这是 Vina
> 自己的做法——制表时按类型分组，代价是 C–O 对用的是 `2·R_O` 而不是 `R_C + R_O`。

### 1.1 这里曾经是一个 bug：所有重原子都是 0.4 Å

早期实现对**所有**重原子返回同一个 `0.4 Å`。0.4 是共价键长，不是相互作用半径，
而 `d = r − (Rᵢ + Rⱼ)` 里的 `R` 决定每一项落在间距轴的哪个位置。后果：

| 项 | 需要的表面距离 | 换算成真实间距 | 真实间距合不合理 |
|---|---|---|---|
| `hb` | `d ≤ −0.5` | **0.3 Å**（O···O） | 否 |
| `g1` | `d = 0.5` | 1.3 Å（C···C） | 否 |
| `g2` | `d = 0` | 0.8 Å（C···C） | 否 |
| `hyd` | `d ≤ 0.5` | 1.3 Å（C···C） | 否 |

两个氧原子相距 0.3 Å 不是氢键，是一个原子和它自己的镜像。于是权重
`−0.587439`（Vina 函数里最大的单项）在**任何真实氢键能形成的距离上都恰好为零**。
实测：酚羟基对冠醚氧，氢键项贡献 −0.0032 kcal/mol（应为 −0.18 量级），
冠醚盒里的所有姿态能量都是 −0.00。

同一个常数也让**全部吸引项的最大值落在碰撞区内部**，于是打分函数的最优解
是"把配体埋进蛋白里"——这正是上一轮"没有有效排斥"这个发现的根因，
它当时是被一个碰撞过滤器掩盖掉的。修好之后，500 次随机采样的结果变成：

```
最佳放置（无冲突） : E = −0.815   最近受体原子 2.39 Å   ← 就是全局最优
最佳放置（无冲突） : E = −0.815   （两者差 0.000）
```

回归测试 `the_interaction_radii_place_the_terms_at_real_contact_distances`
用晶体学接触距离把这条钉死，不依赖任何参考实现的内部数值。

---

## 2. 空间形状函数

逐对能量由五个空间函数线性组合而成。定义如下（括号内是 `scoring.rs` 的实现）：

### 2.1 高斯项

```
g(d; c, w) = exp( −((d − c)/w)² )
∂g/∂d     = g · ( −2(d − c)/w² )
```

| 项 | 中心 `c` | 宽度 `w` | 权重（vina） | 权重（vinardo） |
|---|---|---|---|---|
| `g1` | `0.5` | `0.5` | `−0.035579` | `0` |
| `g2` | `0.0` | `0.5` | `−0.005156` | `0` |

实现：`scoring.rs::gaussian_term`（返回 `(值, 导数)`，调用方不做任何数值微分）。

### 2.2 排斥项

```
rep(d) = d²        当 d < 0
       = 0         当 d ≥ 0
∂rep/∂d = 2d       当 d < 0；否则 0
```

`rep ≥ 0` 恒成立——这是它作为"硬墙"的基本性质，有单测 `repulsion_is_always_non_negative` 盯着。
注意 `d < 0` 的分支在 `d = 0` 处是 C⁰（不是 C¹），这是 AutoDock 一贯的做法。

### 2.3 氢键项与疏水项

这两个在 Vina 里是硬截断，在本工具里是 **C¹ 三次 smoothstep**：

```
S(a, b, x) = 0                      当 x ≤ a
           = t²(3 − 2t)             当 a < x < b,  t = (x − a)/(b − a)
           = 1                      当 x ≥ b

∂S/∂x    = 6t(1 − t)/(b − a)        当 a < x < b；否则 0
```

于是

```
hb(d)  = 1 − S(−0.5,  0.0, d)     深度重叠时为 1，到 0 Å 归零
hyd(d) = 1 − S( 0.5,  1.5, d)     近程 apolar 接触为 1，到 1.5 Å 归零
```

**`hb` 的窗口曾经是 `S(−0.7, −0.5, d)`，即整体偏近 0.5 Å。** 正确的窗口让最大值
落在 O···O 的 2.7 Å（`d = −0.5`）、N···O 的 2.85 Å；旧窗口把最大值推到 2.5 Å，
处在任何真实氢键范围的边缘，并且已经进了碰撞区。

**为什么用 smoothstep 而不是硬截断。** Vina 在 `d = 1.5` 处硬切 `hyd`、
在 `d = −0.5` 处硬切 `hb`。两处都让能量关于距离 C⁰ 而非 C¹，于是：

* 每次有配体原子跨过这两个面，梯度就跳变一次；
* 拟牛顿法（Hessian 近似）会在跳变处采到错误的曲率信息并震荡。

改成 C¹ 过渡不改变势阱位置和深度，代价为零，但让 L-BFGS 在平台上不再打转。
单测 `smoothstep_is_c1` 直接在结点两侧检查单侧斜率都趋零。

> 节点位置会被 `spatial_derivatives_match_finite_differences` 的采样点列表依赖：
> 那个测试必须避开所有结点（`hb` 的 −0.5 / 0.0，`hyd` 的 0.5 / 1.5），
> 因为在结点处中心差分跨过曲率跳变。调窗口时要一起改。

---

## 3. 权重

```
VinaWeights {
    gauss1:   −0.035579,
    gauss2:   −0.005156,
    repulsion:  0.840245,
    hbond:    −0.587439,
    hydrophobic: −0.035069,
    intramolecular_scale: 0.006,
}
```

Vinardo（Quiroga & Villarreal, *PLoS ONE* **11**, e0163579, 2016）：

```
gauss1 = gauss2 = 0        // 丢掉两个高斯
repulsion   = −0.045       // 注意是负的：Vinardo 把 rep 当作"偏好 d 略正"而非墙
hbond       = −0.030
hydrophobic = −0.015
intramolecular_scale = 0.0075
```

两套权重都可以通过 `VinaScoring::new()` / `VinaScoring::vinardo()` 切换，
`ScoringFunction` 是一个 **object-safe trait**，所以 `dock()` 对两者是同一份代码。

---

## 4. 逐对能量的完整组装

```rust
// scoring.rs::VinaScoring::pair_energy
let c = kernels.eval(d);                     // 三个分量，各自带导数
let mut e = c.shape.0;                       // 形状 + 排斥：对所有原子对都加
if (a.can_donate() && b.can_accept()) || (a.can_accept() && b.can_donate()) {
    e += c.hbond.0;                          // 只对 供体–受体 加
}
if a.is_apolar() && b.is_apolar() {
    e += c.hydrophobic.0;                    // 只对 非极性–非极性 加
}
```

`pair_gradient` 是同一段逻辑，只是取 `.1`。

### 4.1 分子内项

```
E_total = E_inter + slope · E_intra,    slope = 0.006（vina）/ 0.0075（vinardo）
```

`E_intra` 是同一套逐对公式在**配体自身**图距 ≥ 4 的原子对上的求和
（`kinematics.rs::MIN_INTRA_BOND_DISTANCE = 4`，即 1-4 及更远）。
它阻止柔性配体把自己折回到身上。

**偏离 Vina。** Vina 为每个构象临时构建一张配体自身网格再读表。
本工具直接对预计算的邻居表求和，消除了插值误差——代价是每步多几十次求值，
换来的是分子内项精确。

---

## 5. 网格预制表

受体**只制表一次**，之后打分一个配体原子是 8 次查表，而不是对受体原子循环。

### 5.1 四个 slot

| slot | 内容 | 由哪些受体原子写入 | 被哪些配体原子读取 |
|---|---|---|---|
| `Shape` (0) | 高斯形状 + 排斥 | 每个原子 | 每个原子 |
| `HbFromDonor` (1) | 氢键 | 能供出氢键的 | 能接受氢键的 |
| `HbFromAcceptor` (2) | 氢键 | 能接受氢键的 | 能供出氢键的 |
| `Hydrophobic` (3) | apolar 接触 | 非极性的 | 非极性的 |

**这是一个有意偏离 Vina 的设计决定。** Vina 只用一张 `e_hb`，受体供体和受体受体
都写进去，配体供体和配体受体都读它。结果是**一对供体–供体会平白捡到一份氢键**。
拆成两张图让这个分解变成精确的，代价是每点多存一个 `f32`。
有单测 `hbond_term_only_for_opposite_polarity` 钉住这个行为。

映射规则在 `scoring.rs::weights_for_kind`：

| 配体原子类别 | Shape | HbFromDonor | HbFromAcceptor | Hydrophobic |
|---|:-:|:-:|:-:|:-:|
| `Hydrophobic` | ✓ | | | ✓ |
| `Donor` | ✓ | | ✓ | |
| `Acceptor` | ✓ | ✓ | | |
| `DonorAcceptor` | ✓ | ✓ | ✓ | |
| `Other` | ✓ | | | |

注意配体 donor 读的是 `HbFromAcceptor`——**读的是配体自己需要的，不是自己提供的**。

### 5.2 内存布局

```
index = ((ix + nx·(iy + ny·iz)) · STRIDE) + type·4 + slot
STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE = 10 · 4 = 40
```

每点每 slot **只有一个 `f32`**，没有第二份径向导数。理由见下一节。
WGSL 里的 `const STRIDE: u32 = 40u` 与 Rust 的 `grid::map_stride()` 由
单测 `shader_and_rust_agree_on_the_grid_stride` 绑定——不一致不会在任何地方报错，
只会让每一次查表整体偏移。

### 5.3 预制表的截断半径

`SpatialKernels::cutoff = 8.0` 是**表面距离**的上限，而内循环手里是**真实距离**，
所以真实距离的截断应当是 `cutoff + 2·R`：

```rust
let reach = kernels.cutoff + 2.0 * a.radius;
if r2 > reach * reach { continue; }
```

早期实现直接拿 `r2` 和 `cutoff` 比，等于把真实距离的截断设成了表面距离的阈值，
于是高斯项的尾巴被悄悄削掉一截：半径都是 0.4 Å 时削 0.8 Å，用上真实 XS 半径后
削 3.8 Å。这类错误不会让任何测试变红，只会让表和公式对不上。

---

## 6. 三线性插值与它的梯度

### 6.1 插值

设 `u = (u_x, u_y, u_z) ∈ [0,1)³` 是 `p` 在所在 cell 内的分数坐标，
八个角的权重

```
w_c = Π_{axis} [ c_ax · u_ax + (1 − c_ax) ]
E(p) = Σ_{c=0..7} w_c · F(g_c)
```

### 6.2 精确梯度

```
∂E/∂p = Σ_c (∂w_c/∂p) · F(g_c)

∂w_c/∂p_ax = (2·c_ax − 1) · Π_{axis' ≠ ax} [ c_ax' · u_ax' + (1 − c_ax') ] / spacing_ax
```

实现见 `grid.rs::interpolate_with_gradient`。注意 `(2c − 1)·u + (1 − c)` 这个写法
对 `c = 0` 和 `c = 1` 的角**统一成立**；若误写成 `(c − 1)`，`c = 1` 的角会
静默地把梯度贡献清零。

### 6.3 一个被明确拒绝的"改进"

有一种很自然的做法是额外存一份制表场的径向导数 `F′`，然后加上

```
Σ_c w_c · F′(|p − g_c|) · (p − g_c)/|p − g_c|
```

**这一项是错的。** 两个理由：

1. 三线性插值函数的梯度**不是**原场梯度的三线性插值。上面那个求和算出来的东西
   不是 `E(p)` 的导数。
2. 存储的 `F′` 是对**受体原子**位置求的偏导，而打分时探针原子是被固定的那一个。

所以本工具只计算它实际使用的插值函数的那个梯度。

### 6.4 由此产生的诚实代价

三线性插值只有 C⁰，所以**这个梯度在 cell 边界是不连续的**。这不是 bug，是
预制表方法的固有性质。本工具的应对是：

* 默认间距取 0.375 Å（AutoDock 默认），足够细；
* L-BFGS 每一步走 **Armijo 充分下降**检验，不满足就缩短步长（见
  [`lbfgs.rs`](../dock-core/src/search/lbfgs.rs)）。

`spacing` 是旋钮：减半会让不连续幅度降到 1/4，代价是 8 倍内存和预制表时间。

**这个效应在真实数据上被直接量到了，而且比预想的大。** 把生物素对进 crambin，
5 个返回姿势的 `|grad|` 都很大（2.3–5.9），而且**逐分量与中心差分符号相反**：

```
dof     analytic    numeric
tx       0.7632    -0.4956
ry       2.2074    -1.4849
tau0    -3.1266    +2.1669
```

这看起来像梯度算错了，其实不是。测量姿势到最近 cell 面的距离：

```
pose 1: closest approach to a cell face = 0.0000 cells
```

**每个返回姿势都恰好有一个配体原子坐在网格面上。** 于是在这个点上存在两个不同的
单边导数，它们大小相近、符号相反，而函数本身在那里取到极小值——所以沿任一方向
都走不动能量（实测：沿解析方向最好 −5.209，沿数值方向 −5.215，起点 −5.222）。

因此：

* **判断一个返回姿势是不是极小点，不能看梯度模长**，必须用线搜索。梯度大只说明
  它落在 cell 面上。`examples/audit_poses.py` 现在就是这么做的（`downhill` 列）。
* 解析梯度本身是**正确**的：在 10 组随机在盒内构象上，11 个 DOF 全部与中心差分
  吻合到 `err = 0.0000`。单测 `the_analytic_gradient_is_the_gradient_of_the_reported_energy`
  （butane + hexane）在 Rust 层把这条链整条钉死。

单测 `interpolated_gradient_matches_finite_difference` 用**严格落在 cell 内部**的
探针点（分数偏移 0.37 / 0.61 / 0.23）做中心差分比对——落在边界上的点有真实的折角，
中心差分会和任何单侧导数都不符，那是方法的性质而不是缺陷。

---

## 7. 原子分类

`AtomKind` 有五类：`Hydrophobic` / `Donor` / `Acceptor` / `DonorAcceptor` / `Other`。

分类**以原子自身的 PDBQT 类型为权威**，`HD` 标记只用于向重原子传播：

```
若该原子键连了一个显式极性氢（HD）      → DonorAcceptor
否则按自身类型：
    C / A 之类（不带极性）               → Hydrophobic
    N                                     → Donor
    NA                                    → Acceptor
    OA / SA                               → Acceptor
    OS（羟基氧，带极性氢）                → Donor
    HD                                    → Other（只是标记）
    其他                                  → Other
```

**一个曾经存在、并被测试钉死的真实 bug。** 早期实现用的是
"若所有邻居都无极性则判为 Hydrophobic"。羧基氧和酯氧只连碳，
于是被整批判成疏水——而羧基氧是本项目里最重要的受体之一。
单测 `carboxylic_oxygens_are_acceptors_not_hydrophobic` 防止它回来。

**一处有意偏离惯例。** 叔胺标为 `NA`（Acceptor），而 Meeko 与 AutoDock 惯例标 `N`
（Donor）。在生理 pH 下中性叔胺确实既可供氢也可受氢，`NA` 更接近化学事实；
调用方若需要旧惯例，直接改 PDBQT 的原子类型即可——分类是读类型得到的，不是猜的。

`amide_nitrogen_is_inert` 与 `kind_assignment_does_not_depend_on_atom_order`
分别钉住"酰胺氮不参与氢键"和"分类与原子编号顺序无关"。

---

## 8. 从原子梯度到构象梯度

`kinematics.rs::conf_gradient` 把每个原子的笛卡尔梯度映射到 DOF 向量上。

**平移。** 直接复制：配体局部坐标为 `qᵢ`，则

```
∂E/∂t = Σ_i  ∂E/∂rᵢ
```

**旋转。** 刚体部分对刚体簇内的力矩

```
τ_c = Σ_{i ∈ c}  qᵢ × (Rᵀ gᵢ)
```

再经 SO(3) 右 Jacobian 转成旋转向量坐标：

```
∂E/∂θ = −J_r(θ)ᵀ · (Rᵀ τ)
```

**符号。** 这里最容易出错。`J_r` 把体坐标角速度映到世界坐标，
而 `dR = R·[J_r dθ]×`，因此 `∂E/∂θ = −J_rᵀ(Rᵀτ)`。
代码里的负号由 `kinematics.rs` 的**全 DOF 中心差分**测试锁定
（`conf_gradient` 的 FD 测试不是抽查某一项，而是对平移 + 三个旋转分量 +
每个扭转做完整比对），所以一旦有人改错符号测试立刻红。

`J_r(θ) = I − ((1−cos|θ|)/|θ|²)[θ]× + ((|θ|−sin|θ|)/|θ|³)[θ]×²`，
小角度展开 `I − ½[θ]× + ⅙[θ]×²`（`kinematics.rs::so3_right_jacobian`）。

**扭转。** 绕世界坐标下的扭转轴 `n`：

```
∂E/∂τ_k = Σ_{i ∈ 子树 k}  n · (rᵢ × gᵢ)
```

---

## 9. GPU 求值路径

`gpu/energy.wgsl` 用 compute shader 对一批 (原子, 探针点) 求能量，
用 shared memory 做树形归约。

**与 CPU 的一致性是被测试强制的**：`gpu_energies_match_the_cpu_interpolation`
在真实 GPU 上跑，断言 GPU 能量与 CPU 三线性插值差 < 1e-3。

### 9.1 为适配 downlevel 而做的布局

WGSL 的 downlevel 路径只保证 **4 个 storage buffer**。初版用了 5 个
（energies / coords / atom_data / grid / weights），在某些设备上直接失败。

解决办法是**把原子类型和 4 个权重打包进同一个 buffer**：

```
ATOM_STRIDE = 8   // [type, w0, w1, w2, w3, pad, pad, pad]
```

权重用 `f32::to_bits()` 存成 `u32`，WGSL 里 `bitcast<f32>` 取回。
绑定重编号为 0..4（4 storage + 1 uniform）。
单测 `the_kernel_stays_within_the_downlevel_storage_buffer_limit` 守住上限，
`shader_and_rust_agree_on_the_atom_stride` 守住两个 stride 常量。

### 9.2 uniform 的 16 字节对齐

`GridParams` 末尾补了 `_pad: [u32; 3]`，使结构体大小为 64 字节、
`min` 落在偏移 0、`spacing` 落在偏移 48——WGSL uniform 地址空间的
`struct` 成员对齐要求。缺了这块 padding，`min_binding_size` 与实际布局不符，
pipeline 验证会失败。单测 `grid_params_matches_the_wgsl_uniform_layout` 断言偏移。

---

## 10. 一句话总结偏差清单

| # | 偏离 | 位置 | 理由 |
|---|---|---|---|
| 0 | 每元素 XS 半径，而非统一 0.4 Å | `types.rs::interaction_radius` | 0.4 Å 让全部吸引项落进碰撞区，氢键项在任何真实距离上恒为零 |
| 1 | 4 个 slot 而非 3 个 | `grid.rs::MAPS_PER_TYPE` | 消除 donor–donor 伪氢键 |
| 2 | hb / hyd 用 C¹ smoothstep | `scoring.rs` | 避免梯度跳变拖垮拟牛顿 |
| 3 | 分子内项直接求和而非临时配体网格 | `search/mod.rs` | 去掉插值误差 |
| 4 | 梯度只保留插值权重导数项 | `grid.rs::interpolate_with_gradient` | 存径向导数是错的 |
| 5 | 叔胺标 `NA` 而非 `N` | `types.rs` | 更接近化学事实；可改 |
| 6 | L-BFGS 每步 Armijo 回退 | `search/lbfgs.rs` | 插值梯度 C⁰ 边界需要 |
| 7 | 原子分类以自身 PDBQT 类型为权威 | `types.rs` | 邻居统计会把羧基氧误判为疏水 |
| 8 | 报告前过滤重叠姿势 | `docking.rs` | 结构断言，不依赖经验权重集 |

第 0 条是一个 **bug 修复，不是设计选择**。第 2 条的 `hyd` 窗口（`0.5 → 1.5`）
相对 Vina 的分段线性版本偏保守，会让最强的疏水吸引落在 1.3–2.3 Å（C···C），
仍然低于 2.0 Å 碰撞阈值的上沿——这是 smoothstep 化的代价，**未与 Vina 交叉验证**。

**最重要的诚实声明**：这些 XS 半径是按参考非键接触距离取的，
**没有与 AutoDock Vina 的二进制做过数值比对**（见
[`VERIFICATION.md`](VERIFICATION.md) §1 的阻塞项）。因此本工具输出的绝对能量
**不应**直接与文献或 Vina 输出的 kcal/mol 数字比较。

出处：AutoDock Vina 1.2（Trott & Olson, *J. Comput. Chem.* **31**, 455, 2010;
Apache-2.0）、Vinardo（Quiroga & Villarreal, *PLoS ONE* **11**, e0163579, 2016;
CC-BY）、AutoDock 4.2（Morris et al., *J. Comput. Chem.* **29**, 2789, 2008;
GPL-2.0）、Meeko（Morris et al., *PLoS ONE* **17**, e0163573, 2022; LGPL-2.1）。
全部为按公开方程的独立实现，未复制上游代码。
