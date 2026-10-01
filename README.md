# Open Docking

[English](README.en.md) | [中文](README.md)

分子对接工具。Rust 引擎负责亲和力网格预制表、经验打分与构象搜索，
Python 前端负责化学预处理（RDKit 芳香性 / 质子化 / 电荷 / 极性氢），
外加一个 PyQt6 + ModernGL 的交互式 3D 工作台。

对接流程本身是经典且经过验证的：受体预制为亲和力网格（三线性插值），
配体用迭代局部搜索 + 拟牛顿下降寻优，打分采用 AutoDock Vina 经验函数的形式。
在此之上提供 **一阶解析梯度**、**纯安全 Rust**（`#![forbid(unsafe_code)]`）、
以及**可用的 GPU 批量打分路径**。

> **两条必读**
> 1. **回对接只做了一小批，结果不好就是不好。** `scripts/redock_benchmark.py`
>    把已知结合在晶体结构里的配体重新对接回去：引擎在四个复合物里两个把
>    RMSD 压到 2 Å 以内（生物素 1.17 Å、苯甲脒 1.19 Å），
>    口袋自动选出的盒子只有一个（3PTB 1.26 Å）。1HVR 那 46 原子的柔性配体
>    仍然是 21 Å，**没有为了好看去调参**。数字和它的采集方式都写在
>    [`docs/VERIFICATION.md`](docs/VERIFICATION.md) §3.7。
> 2. **绝对能量不可与 AutoDock Vina 或文献值直接比较**（未做交叉验证）。
>    相对排序、姿势质量、几何合理性可以参考。
>
> 完整限制见 [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)。

## 安装

需要 Rust 1.75+、Python 3.9+、maturin。**本仓库不含预编译 wheel，需要自行构建一次。**

```bash
python -m pip install -r requirements.txt

# CPU 版
python -m maturin build --release -m dock-py/Cargo.toml --out dist
python -m pip install dist/opendocking-0.1.0-*.whl

# 需要 3D 工作台时再装（引擎本身不需要任何图形栈）
python -m pip install PyQt6 moderngl numpy-stl
```

可选的 GPU 版：

```bash
python -m maturin build --release -m dock-py/Cargo.toml --out dist-gpu --features gpu
python -m pip install --force-reinstall --no-deps dist-gpu/opendocking-0.1.0-*.whl
odcli info        # 期望：gpu: compiled in and an adapter is available
```

只实测过 **Windows 11 x86_64 + NVIDIA RTX 3050 Laptop**。
macOS、Linux、AMD 独显、Intel 核显、Apple Silicon **未测**。

## 使用

### 命令行

完整 CLI 的命令名是 **`odcli`**。

| 子命令 | 作用 |
|---|---|
| `info` | 打印引擎版本、打分参数、GPU 状态、并行后端 |
| `prep-receptor` | `.pdb` → 受体 PDBQT |
| `prep-ligand` | 任意结构 → 配体 PDBQT + 性质报告 |
| `rec-grid` | 预制表并写出 AutoDock `.map` 文件 |
| `dock` | 对接单个配体或整个目录 |
| `sites` | 列出受体里的候选结合位点（带衬里残基） |
| `split` | 多 MODEL PDBQT 拆成每构象一个文件 |
| `workbench` | 启动交互式 3D 工作台（等同 `odgui`） |

一段可以直接跑通的完整流程（仓库自带示例数据）：

```bash
cd examples

# 受体 PDB -> PDBQT
odcli prep-receptor -r receptor.pdb -o rec_prep.pdbqt

# 配体 SDF -> PDBQT（同时打印扭转数、自由度数、原子分类）
odcli prep-ligand -l ibuprofen.sdf -o ibuprofen_prep.pdbqt

# 看看这个受体上有哪些候选位点
odcli sites -r 1crn_prep.pdbqt

# 预制表 + 对接。搜索盒要么显式给六个数，要么用 --auto-box 取第 N 个位点
odcli dock -r 1crn_prep.pdbqt -l ibuprofen_prep.pdbqt \
           --auto-box 2 -e 8 -o poses.pdbqt
```

搜索盒仍然**必须给出**——只是现在有两种给法。`odcli` 一直拒绝从受体猜：
全蛋白的盒子在内存里极大，作为搜索区域又毫无用处。
`--auto-box N` 取的是 `odcli sites` 列出的第 N 个位点，
`--box-padding` 控制每个方向留多少余量（默认 4 Å）。
位点比配体还小时会自动撑到引擎下限（`2 × 半径 + 1 Å`）并说明撑了哪根轴；
自己手给的盒子不会被改动。

常用选项：

```
-e, --exhaustiveness N     独立搜索轨迹数（默认 8，32 算彻底）
-m, --num_modes N          输出多少个去重后的构象（默认 9）
--rmsd-cutoff R            判为同一构象的 RMSD 阈值（默认 1.0 Å）
--scoring {vina,vinardo}   打分函数
--spacing S                网格间距（默认 0.375 Å）
--mode {mc,lga,both}       全局搜索策略
--seed N                   固定随机种子（可复现）
--json                     以 JSON 输出
--auto-box N               用第 N 个位点代替六个盒子数值
```

### Python

```python
from opendocking import Receptor, Ligand, GridBox, dock

receptor = Receptor.from_pdbqt("rec_prep.pdbqt")
box_ = GridBox.from_center_size((0., 0., 0.), (20., 20., 20.))
maps = receptor.precalculate(box_, scoring="vina", spacing=0.375)

ligand = Ligand.from_pdbqt("ibuprofen_prep.pdbqt")
result = dock(ligand, maps, exhaustiveness=8, num_modes=9)
print(result.summary())
result.write_pdbqt("poses.pdbqt")
```

完整 API 见 [`docs/API.md`](docs/API.md)。

### 3D 工作台

`odgui` 是独立入口点，不依赖任何特定命令名。

```bash
odgui                                       # 打开空窗口，从 File 菜单加载
odgui -r rec_prep.pdbqt -l lig.pdbqt -p poses.pdbqt
odgui --check                               # 只检查能否启动，不开窗口
```

左键旋转、右键平移、中键变焦、滚轮缩放、`Frame all` 重新取景；
File 菜单按 `Load receptor… / Load ligand… / Open poses… / Dock now` 加载与对接。

**显示方式**（控制面板 `display`）：

| 模式 | 说明 |
|---|---|
| `Space-filling` | 每个原子一个球，键画成细线 |
| `Ball and stick` | 缩小原子 + 沿键的圆柱，每半根按各自原子的颜色着色 |
| `Skeletal` | 只画键，即键线式 |
| `Ribbon` | 沿骨架的平滑条带，按几何估计的二级结构着色 |
| `Cartoon` | 条带 + 侧链细棍 |

后两种需要蛋白骨架。没有骨架的结构会自动回退到球棍，并在状态栏说明。
状态栏同时会写明当前每个结构的键是**从文件读来的**、**按残基模板推的**、
还是**按距离推断的**——球棍图里一根画错的键和画对的键看起来一样可信，
所以这句话不能省。

**接触分析**（控制面板 `interactions`）：

姿势放进受体之后，**埋在疏水沟里的姿势和粘在蛋白表面的姿势从外面看一模一样**。
工作台现在会把这件事说出来：

- 接触原子之间画**虚线**，按类型着色，面板里有对应图例
  （氢键近白、极性青、疏水紫、其他灰蓝）；
- `interactions` 表格逐残基列出氢键数、接触数和最近距离，
  **氢键多的排前面**——只看最近距离会把「擦边碰到」排在氢键前面；
- **点任意一行，相机就飞到那个残基的接触点上**。

氢键是**几何判据**：H···受体 ≤ 2.6 Å 且 D–H···A 夹角 ≥ 120°，**两个方向都查**
（对接姿势里供体通常是受体）。这不是量子计算，不验证受体孤对电子朝向、
不建模水桥、不给能量。每个接触都带回距离和角度，阈值可以自己收紧。

> **残基名从哪来**：合成受体（`rec_prep.pdbqt`）的残基全叫 `REC`，
> 所以对它只能报「有 N 个接触」而**无法归属到残基**——表格会是空的。

**候选结合位点**（控制面板 `Find pockets` / `site`）：

加载受体后，工作台会**自动搜索口袋**，把搜索盒放在找到的第一个位点上，
并把那一行选中——你能看见它从哪来，也能点别的行换一个。
点任意一行，**盒子、三个坐标输入框和相机一起动**；
输入框和引擎拿到的是同一组数字。

> **以前这里放的是受体质心**，也就是所有原子的算术质心。
> 对球状蛋白那正好在致密核心里，搜索区域坐在实体蛋白中，
> 而三个输入框把结果显示得像个经过深思熟虑的答案。
> 找不到位点时会明说「没有找到，框还在原处，**这不是建议**」。

搜索的是**封闭空间**，不是结合位点。每个位点报告中心、尺寸、埋藏分数
（三个轴里有几个方向两侧都是蛋白）和**衬里残基**——
把「(12, 7, 0) 附近有个沟」变成可以人工核对的东西。
一条平贴在蛋白表面的配体得分为 0，永远不会被列出来。

选中的位点会**画出来**：它自己的网格点渲染成一层半透明的体积（`site volume` 可关）。
表格里写「沟，衬里 ARG 17A / ASN 14A / THR 2A」是一个**说法**，
一团夹在这些原子之间的体积才是读者自己的核对，
再多几列数字也替代不了。结晶水和缓冲添加剂**不算**衬里残基，
辅因子算——它是真的化学。

搜索在**工作线程**上跑：实测隐花匠 327 原子 0.16 s，链霉亲和素 1001 原子 1.6 s，
血红蛋白 4779 原子 **11.0 s**。大受体要等，但界面不会僵。
> 这不是坏了，是它确实没有残基身份。用真实蛋白才有残基编号。

画面本身开了 **4x 抗锯齿**、**深度雾**（空气透视，让大体积受体的前后关系读得出来）、
双光源加边缘光，背景是渐变而非纯色。雾的范围跟随相机距离，
所以近看小分子时它不掺和，缩远看大受体时它干活。

## 使用注意

- **搜索盒必须显式给出**（六个数，或 `--auto-box N`）。
  一个覆盖整蛋白的盒子既会吃掉几十 GB 内存，又等于没有搜索区域；
  自动位点比全蛋白 box 小得多（crambin 实测小 3.8 倍，仍装下全部 16 个配体原子）。
  `exhaustiveness` **默认跟着盒子走**（不给 `-e` 时按体积算，并把用的值印出来）：
  它是蒙特卡洛步数，步数摊在一个越大的盒子里越稀，**采样不足会返回一个坏姿势，
  而坏姿势和盒子放错了在界面上长得一模一样**。实测 39×26×41 Å 的盒子
  在 16 时是 12.88 Å，在 64 时是 **1.26 Å**，两秒。**手动改过之后它就不再跟随**。
- **口袋搜索找的是封闭空间，不是结合位点。** 一条平贴在表面的配体得分为 0，
  不会被列出；最大的位点也不一定是你这个配体想要的那个。
  它是一份可以挑的候选清单，**由你来挑**（表格第 5 列 `vol Å³` 就是排序用的那个量）。
- **配体贴合得很紧时，位点会被找到但排在很后面。** crambin 上布洛芬所在的位点
  只有 **12 Å³**，排在 16 个里的**第 9**。这不是 bug：搜索量的是配体**留下**的空间，
  不是它占据的空间，配体塞得越满，剩下的空间越小。口袋比配体宽松时
  （上表那四个复合体都是），真位点每次都排**第一**。
- **又长又弯的裂隙会得到一个大盒子，大盒子难搜。** 3PTB 的位点是一条
  31 × 18 × 33 Å 的裂隙，盒子是 39 × 26 × 41 Å。没有任何一个盒子能既装下
  9 原子的配体又保持小——这是「一位点一盒」的真实上限。**按体积给足采样量**有用：
  同一个盒子 exhaustiveness 16 读出 12.88 Å，64 读出 **1.26 Å**，都用不了两秒。
- **1HVR 是最差的一个，如实留着。** 46 原子的柔性配体 + 30×38×32 Å 的盒子，
  exhaustiveness 64 时 21.24 Å、128 时 10.53 Å——随采样单调变好，
  说明是采样不足而不是位点不对。
- **从自动 box 重新对接，crambin 的参考能量能复现，姿势不能。**
  对接布洛芬到参考姿势所在位点得 −5.36 kcal/mol（参考 −5.50），
  但 RMSD 是 4.24 Å：能量复现了，落在另一个极小点上。
  crambin + 布洛芬有接近简并的结合模式，这是能量面的性质，不是 box 放错了。
- **位点体积上限 1500 Å³ 是启发式**，不是推导出来的：它现在只挡真正巨大的空腔。
  它曾经还有第二个**偶然**的职责——清掉膨胀产生的外表面团块；膨胀去掉后
  那个职责也没了，crambin 上它已经**什么都不删**（16 → 16），检查里对此有显式断言。
- **小空腔在默认探针下找不到。** T4 溶菌酶 L99A 专门改造出一个埋藏空腔，
  本工具在它里面报告 **0 个**密闭空腔：空腔约 100 Å³，1.4 Å 探针把每个原子
  都膨胀到这个量级，直接把它填满了。降低默认值更糟——0.5 Å 探针会把表面沟槽
  闭成几十个假空腔。所以默认值不变，改为**在找不到密闭空腔时告诉你更小的探针会看到什么**
  （`odcli sites` 会打印 `1.4 A -> 0, 1.1 A -> 0, 0.9 A -> 1, …`）。
  **一个无法行动的空结果只是一句话。**
- **大受体上搜索会明显变慢**：4779 原子的血红蛋白要 11 秒。它在工作线程上跑，
  界面不会僵，但确实要等。
- **参数名用下划线**：`--center_x`，不是 `--center-x`。
- **`use_gpu` 默认关闭。** GPU 端用 `f32` 累加，与 CPU 差约 1e-6，
  让默认值随硬件变化会破坏跨机可复现性。要用就显式传 `use_gpu=True`，
  再读 `report_backend` 里的回退原因。
- **不要用 `|grad|` 判断姿势收敛了没有。** 网格是三线性插值，只有 C⁰，
  而实测每个返回姿势都有原子坐在 cell 面上，那里 `|grad|` 常见为 2–6
  却哪儿都走不动。要判断极小性请沿线搜索。
- **改过 `dock-py/python/` 下的 `.py` 必须重新构建并安装 wheel**，
  否则 `pytest --pyargs` 测的还是上一次的代码。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **先读这个**：已知限制与不适用范围 |
| [`docs/API.md`](docs/API.md) | Rust 与 Python 的完整 API 参考 |
| [`docs/SCORING.md`](docs/SCORING.md) | 打分公式推导，与 Rust 实现逐项对照 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 数据流、并行模型、内存布局、零拷贝契约 |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | **诚实验证报告**：跑了什么、什么没验证 |
| [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) | 出处、许可与 clean-room 声明 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 开发流程、硬性约束、本地验证清单 |
| [`SECURITY.md`](SECURITY.md) | 安全策略 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |
| [`CITATION.cff`](CITATION.cff) | 引用元数据 |

## 许可

**GPL-3.0-or-later**，全文见 [`LICENSE`](LICENSE)。
由于分发 GPL-3.0 代码，任何二进制分发都必须同样以 GPL-3.0 提供完整源码。
功能形式与权重的出处见 [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md)。
