# Open Docking

[English](README.en.md) | [中文](README.md)

分子对接工具。Rust 引擎负责亲和力网格预制表、经验打分与构象搜索，
Python 前端负责化学预处理（RDKit 芳香性 / 质子化 / 电荷 / 极性氢），
外加一个 PyQt6 + ModernGL 的交互式 3D 工作台。

对接流程本身是经典且经过验证的：受体预制为亲和力网格（三线性插值），
配体用迭代局部搜索 + 拟牛顿下降寻优，打分采用 AutoDock Vina 经验函数的形式。
在此之上提供 **一阶解析梯度**、**纯安全 Rust**（`#![forbid(unsafe_code)]`），
以及一条**可选的 GPU 批量打分路径**——它**不在搜索热路径上**，见下。

> **两条必读**
> - **回对接只做了一小批，结果不好就是不好。** 已知结合在晶体结构里的配体被
>    重新对接回去，问它能不能回来. `scripts/redock_benchmark.py` 的那次运行在
>    四个复合物里把两个的 RMSD 压到 2 Å 以内（生物素 1.17 Å、苯甲脒 1.19 Å），
>    口袋自动选出的盒子只有一个（3PTB 1.26 Å），而 1HVR 那 46 原子的柔性配体
>    仍然是 21 Å，**没有为了好看去调参**.
>    **下面这批数字带着一个前提：它出自一个此后被修好并重新构建过的引擎.**
>    `docs/VERIFICATION.md` 逐条记着那次修的是什么、漏掉了哪一次构建.
>    `scripts/redock_benchmark.py` 的语料是运行时现下的结构体、没有钉版本，
>    那次运行也没有留下指纹供日后比对.
>    所以今天重跑它是**一次新的实验**，不是对同一个读数的复核.
>    数字和采集方式仍写在 `docs/VERIFICATION.md` §3.7，
>    `scripts/redock_benchmark.py` 的采样设置也在那一节里.
>    本文因此不替它换成新数字.
> - **绝对能量不可与 AutoDock Vina 或文献值直接比较**（未做交叉验证）。
>    相对排序、姿势质量、几何合理性可以参考。
>
> 完整限制见 [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)。

## 安装

需要 Rust 1.75+、Python 3.9+、maturin。**本仓库不含预编译 wheel，需要自行构建一次。**
这两个下限不是随手写的：`Cargo.toml` 的 `rust-version` 与
`dock-py/pyproject.toml` 的 `requires-python` 就是这两个数，
`scripts/docs_claims_check.py` 每次运行都会把它们和本文对照。

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

只实测过 **Windows 11 x86_64 + NVIDIA RTX 3050 Laptop**，本文所有 Å、kcal/mol
和秒的读数都是 `scripts/redock_benchmark.py`、`scripts/pocket_benchmark.py` 和
`scripts/gpu_cpu_parity_check.py` 在这台机器上跑出来的。
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

搜索盒仍然**必须给出**——只是现在有两种给法. `odcli` 一直拒绝从受体猜：
全蛋白的盒子在内存里极大，作为搜索区域又毫无用处.
`--auto-box N` 取的是 `odcli sites` 列出的第 N 个位点，
`--box-padding` 控制每个方向留多少余量（默认 4 Å），这两个数由
`scripts/docs_claims_check.py` 每次对着 `cli.py` 重钉.
位点比配体还小时会自动撑到引擎下限（`2 × 半径 + 1 Å`）并说明撑了哪根轴；
自己手给的盒子不会被改动。

常用选项：

```
-e, --exhaustiveness N     独立搜索轨迹数（不给就按盒子体积算，见下；32 算彻底）
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

**键盘**（`?` 随时打开这张表，窗口自己知道每个键此刻是否有效）：

| 键 | 作用 |
|---|---|
| `←` `→` | 上一个 / 下一个姿势 |
| `↑` `↓` | 姿势表聚焦时同上；接触对表聚焦时切换接触对 |
| `W` `A` `S` `D` | 相机上下左右环绕 |
| `F` | 框住当前选中的一切：接触对 → 残基 → 姿势 → 都不选就框全部 |
| `K` | 按面板里的盒子、exhaustiveness 和种子跑一次对接 |
| `C` / `V` | 显隐接触虚线 / 位点体积云 |
| `M` | 切换下一种显示方式 |
| `E` | 把这次运行导出成文件 |
| `?` | 显隐快捷键表，`Esc` 关闭 |

两处是刻意的，不是省事：

- **`↑` `↓` 不装成窗口快捷键。** `QTableWidget` 自己就用它们移动当前行，
  而那次移动发出的是和点击**同一个信号**、走**同一个槽**——所以表格按方向键时，
  选中的接触对照样高亮、居中、并在状态栏报名。硬装一份就是一个手势两个处理器，
  而没被测过的那个就是会出问题的那个。表里仍然列出它们，是因为
  「此刻哪些键有效」是读者需要知道的，不是为了让你再按一遍。
- **`F` 不重新计算取景，而是重放选中项已经挣来的那一次相机移动**，走同一个
  viewport 调用. 原来的实现把相机从接触对中点挪开 8.80 Å、拉远 3.8 倍，
  一条 2.05 Å 的接触线就从 32 px 的高亮缩到 5 px，而
  `scripts/workbench_interaction_check.py` 并不测量这五个数.
  **它们是阈值式的记载**：五个数是该脚本 detail 字符串里写死的字面量，
  搜 `8.80 A off` 就能找到，描述的是旧实现曾经的样子.
  `scripts/workbench_interaction_check.py` 断言的是**今天**的行为：
  接触对仍留在两原子中点 1.0 Å 内，相机位移与距离变化都小于 0.5 Å.
  「我选中了它，然后让它框住它，
  结果它变小了」——一个自我撤销的手势，出现在一个自称「框住选中项」的键上最不合适。

`K` 在搜索还在跑的时候再按一次，会**拒绝并说明原因**，不排队：两次搜索共用同一对
线程槽正是结果会丢的方式，而一个排了队却始终没跑起来的请求，就是静默失败。
搜索跑在**工作线程**上，窗口不会僵住，面板会写明现在到了哪一阶段。

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

氢键是**几何判据**：H···受体 ≤ 2.6 Å 且 D–H···A 夹角 ≥ 120°，这两个数就是
`dock-py/python/opendocking/workbench/contacts.py` 的 `HBOND_MAX` 和
`HBOND_MIN_ANGLE`，`scripts/docs_claims_check.py` 每次运行都重读它们.
**两个方向都查**（对接姿势里供体通常是受体）.
这不是量子计算，不验证受体孤对电子朝向、
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
把「(12, 7, 0) 附近有个沟」变成可以人工核对的东西，那三个坐标是编出来的、
不是读数，`odcli sites` 印出来的行才是真的.
一条平贴在蛋白表面的配体得分为 0，永远不会被列出来，这一行为由
`scripts/pockets_check.py` 断言。

选中的位点会**画出来**：它自己的网格点渲染成一层半透明的体积（`site volume` 可关）。
表格里写「沟，衬里 ARG 17A / ASN 14A / THR 2A」是一个**说法**，
一团夹在这些原子之间的体积才是读者自己的核对，
再多几列数字也替代不了。结晶水和缓冲添加剂**不算**衬里残基，
辅因子算——它是真的化学。

搜索在**工作线程**上跑，`scripts/pocket_benchmark.py` 实测隐花匠 327 原子 0.16 s，
链霉亲和素 1001 原子 1.6 s，血红蛋白 4779 原子 **11.0 s**. 大受体要等，但界面不会僵。
> 这不是坏了，是它确实没有残基身份。用真实蛋白才有残基编号。

**姿势可信度**（控制面板 `trust`）：

从**姿势文件**读进来的姿势和一个刚跑完的 run 不是一回事。文件里存的是坐标，
不是构象——引擎的梯度、它的出盒罚项都是**记忆搜索过程中**的性质，
那时它被测过。所以面板会先说清楚这一点，再逐条列出四条契约：

| 契约 | 它问的是什么 |
|---|---|
| `stationarity` | 这个姿势是被引擎评分的那个场的驻点吗 |
| `line-search resolution` | 线搜索能分辨的宽度是多少 |
| `field support` | 配体最近的受体原子有多近 |
| `inside the box` | 有没有原子跑出了网格 |

每行有**两个独立通道**：状态用**自己的颜色 + 自己的斜体**同时写出来
（`holds` / `fails` / `unmeasured`），数值在 `measured / threshold` 列，
数字本身来自这一行 tooltip 指名的那个对象，不是另算一遍。三种状态是三个
不同的答案：`unmeasured` **不是通过**，它是「结果没带这个契约需要的数」，
并且按自己的颜色和自己的斜体印出来。

一条已知的失败**压过**未测：`no` 会点名是哪条契约失败了，未测的那些仍然照列。

> 三个状态的完整定义和每一条为什么这么分，都写在面板自己那页里——
> 这是读者第一眼会去看的地方，不该只存在于源码注释里。

**导出**（`E` 键，或 File 菜单 `Export run to a file`，同一个方法）：

把每个姿势连同能量、分项、判定、接触和出处一起写成文件，路径打在状态栏。
**一个姿势都没有的时候它会说明并且不写任何文件**——这也是一个回答。

写出来的东西里，`0.0` 和 `null` 不是一回事：前者是**确实测到了零**，
后者是**没测**，带 `state` 和 `because` 说明为什么没测.
未测的条目另汇总在 `absent_index` 里.
写出这三样的是 `dock-py/python/opendocking/workbench/app.py`，它逐字段决定.
**而这里有一个覆盖缺口：没有任何门禁钉住导出文件的格式.**
`scripts/pose_trust_check.py` 不读这个导出——它通篇没有 `absent_index`，
也没有 `json`，唯一的 "export" 是一个 detail 字符串里的一个词，
所以上面这个区分由源码而不是由门禁担保.

> 导出的坐标和 GUI 里逐位相同，GUI 与 CLI 导出的两个文件也逐字节相同。
> **这不等于窗口是对的**，只说明窗口不是第二种意见。

画面本身开了 **4x 抗锯齿**（`scripts/docs_claims_check.py` 对着
`dock-py/python/opendocking/workbench/app.py` 的 `MSAA_SAMPLES` 重钉这个数）、
**深度雾**（空气透视，让大体积受体的前后关系读得出来）、
双光源加边缘光，背景是渐变而非纯色。雾的范围跟随相机距离，
所以近看小分子时它不掺和，缩远看大受体时它干活。

## 使用注意

- **搜索盒必须显式给出**（六个数，或 `--auto-box N`）. 一个覆盖整蛋白的盒子既会吃掉
  几十 GB 内存，又等于没有搜索区域；自动位点比全蛋白 box 小得多，
  `scripts/pockets_check.py` 在 crambin 上断言了这一点，逐个位点实测小 5.8–29.7 倍.
  这个区间**树里没有任何脚本会重算**，它依赖全蛋白 box 的取法，所以只当量级看.
  `scripts/pockets_check.py` 确实会在运行时算一个比值，但只有一处、只针对排第 9 的那个位点.
  它的分母是 `scripts/pockets_check.py` 自己写死的 22 Å 立方，不是它量出来的盒子.
  `scripts/pockets_check.py` 印出来的是 1505 对 10648 Å³，也就是 7.1 倍.
  那个 22 Å 也**不是** `dock-py/python/opendocking/core.py` 里的
  `REFERENCE_BOX_SIDE`，后者是 20 Å.
  所以这个区间连同它的取法都没有仪器.
- **`exhaustiveness` 默认跟着盒子走**（不给 `-e` 时按体积算，并把用的值印出来）.
  它是蒙特卡洛步数，步数摊在一个越大的盒子里越稀，**采样不足会返回一个坏姿势，
  而坏姿势和盒子放错了在界面上长得一模一样**.
  39×26×41 Å 的盒子上有一条标定点：exhaustiveness 16 是 12.88 Å，
  64 是 **1.26 Å**，两秒——而 `scripts/redock_benchmark.py` 并不打印它.
  **这两个读数不是任何脚本的输出**：它们写死在 `scripts/redock_benchmark.py`
  的源码注释里，`scripts/pockets_check.py` 只是复述了同一句.
  `scripts/pockets_check.py` 断言的是体积规则对这个盒子答 64.
  `scripts/redock_benchmark.py` 每个盒子只跑一个由体积选出的 exhaustiveness，
  它在结构上就打印不出 16 与 64 这一对.
  **手动改过之后它就不再跟随**.
- **口袋搜索找的是封闭空间，不是结合位点。** 一条平贴在表面的配体得分为 0，
  不会被列出（`scripts/pockets_check.py` 断言）；最大的位点也不一定是你这个配体
  想要的那个。
  它是一份可以挑的候选清单，**由你来挑**（表格第 5 列 `vol Å³` 就是排序用的那个量）。
- **配体贴合得很紧时，位点会被找到但排在很后面.** crambin 上布洛芬所在的位点
  只有 **12.3 Å³**，在默认返回的 **12** 个位点里排**第 9**（放开位点数上限时是
  16 个，第 9 名不变），这个排名和这个体积由 `scripts/pockets_check.py` 与
  `scripts/docs_claims_check.py` 每次从 `examples/1crn_prep.pdbqt` 重算.
  这不是 bug：搜索量的是配体**留下**的空间，
  不是它占据的空间，配体塞得越满，剩下的空间越小。口袋比配体宽松时
  （上表那四个复合体都是），真位点每次都排**第一**。
- **又长又弯的裂隙会得到一个大盒子，大盒子难搜.** 3PTB 的位点是一条
  31 × 18 × 33 Å 的裂隙，盒子是 39 × 26 × 41 Å，
  也没有任何一个盒子能既装下 9 原子的配体又保持小，这三个数出自
  `scripts/redock_benchmark.py`，属于开头说过的那次旧构建.
  盒子尺寸另有一处活证据：`scripts/pockets_check.py` 每次运行都断言
  体积规则对它答 64，并把它自己的体积算成 41574 Å³.
  裂隙跨度和配体原子数没有这样的门禁.
  这是「一位点一盒」的真实上限。**按体积给足采样量**有用：
  同一个盒子 exhaustiveness 16 是 12.88 Å、64 是 **1.26 Å**，都用不了两秒，
  但它同样只是 `scripts/redock_benchmark.py` 注释里的标定点，没有脚本会重算.
- **1HVR 是最差的一个，如实留着。** 46 原子的柔性配体 + 30×38×32 Å 的盒子，
  exhaustiveness 64 时 21.24 Å、128 时 10.53 Å，这三个数出自
  `scripts/redock_benchmark.py` 那次已被修掉并重新构建过的旧运行.
  `scripts/redock_benchmark.py` 每个盒子只跑一个 exhaustiveness，
  64 与 128 这一对同样没有脚本打印过.
  盒子尺寸 30×38×32 Å 是 `scripts/pockets_check.py` 携带的字面量，
  它对这个尺寸断言答 64.
  随采样单调变好，说明那次读数是采样不足而不是位点不对——
  但这是旧读数的内部一致，不是今天重测的结果.
- **从自动 box 重新对接，crambin 的参考能量能复现，姿势不能.**
  干这件事的是 `scripts/pockets_check.py`，不是 `scripts/redock_benchmark.py`——
  后者的四个用例是 1STP / 3PTB / 2NNQ / 1HVR，**里面没有 crambin**.
  `scripts/pockets_check.py` 这次运行打印的是 −5.50 kcal/mol，
  对同一个脚本现算的参考能量 −5.50，RMSD 3.73 Å.
  能量落在同一个高度，姿势落在另一个极小点上，
  而这条检查自己就写明「不是一次复现，也不被当成一次复现」.
  旧的 −5.36 kcal/mol / 4.24 Å 这一对，`scripts/pockets_check.py` 今天的输出不是它，
  树里也没有任何一次有记录的产生过它. 证据在 `scripts/pockets_check.py`
  自己的注释里（第 1083-1084 行），它把 −5.42 kcal/mol / 3.93 Å 记作修好之前的值，
  把 −5.36 说成「第三个数字，既不是修正前的，也不是现在的」.
  同一段注释自己称作「现在」的 −5.30 kcal/mol / 4.35 Å 也不是
  `scripts/pockets_check.py` 今天打印的，所以那段注释记录的是过去的状态，不是实测.
  所以这一对按「来源不明」记下来，而不是按「修好之前的那一版」——
  **后者等于给它安一个树里没有任何证据的出处**.
  crambin + 布洛芬有接近简并的结合模式，这是能量面的性质，不是 box 放错了.
- **位点体积上限 1500 Å³ 是启发式**，不是推导出来的，
  `scripts/docs_claims_check.py` 每次运行都从 `pockets.py` 重读它：
  它现在只挡真正巨大的空腔.
  它曾经还有第二个**偶然**的职责——清掉膨胀产生的外表面团块；膨胀去掉后
  那个职责也没了，crambin 上它已经**什么都不删**（放开位点数上限时 16 → 16；
  默认上限 12 → 12），两个计数都由 `scripts/docs_claims_check.py` 显式断言。
- **小空腔在默认探针下找不到.** T4 溶菌酶 L99A 专门改造出一个埋藏空腔，
  本工具在它里面报告 **0 个**密闭空腔：空腔约 100 Å³，1.4 Å 探针把每个原子
  都膨胀到这个量级，直接把它填满了，`scripts/pocket_benchmark.py` 之所以把 1L96
  放进语料就是为了演示这件事.
  降低默认值更糟——0.5 Å 探针会把表面沟槽
  闭成几十个假空腔。所以默认值不变，改为**在找不到密闭空腔时告诉你更小的探针会看到什么**
  （`odcli sites` 会打印 `1.4 A -> 0, 1.1 A -> 0, 0.9 A -> 1, …`）。
  **一个无法行动的空结果只是一句话。**
- **大受体上搜索会明显变慢**：4779 原子的血红蛋白要 11 秒，与上面那张表同一次
  `scripts/pocket_benchmark.py` 运行。它在工作线程上跑，
  界面不会僵，但确实要等。
- **参数名用下划线**：`--center_x`，不是 `--center-x`。
- **`use_gpu` 默认关闭，而且 GPU 不在搜索热路径上。** GA、蒙特卡洛和 L-BFGS
  **从不**调用 GPU 路径：它只有一个调用方，是 `evaluate_conformations` 这个批量接口。
  要用就显式传 `use_gpu=True`。默认值不随硬件变化，是为了保住跨机可复现性。

  **这条路径的盒外行为曾经是另一个函数**（Vulkan，RTX 3050 Laptop，驱动 572.83，
  1crn 受体 + 生物素，22 Å 盒，0.5 Å 间距，512 个构象），除驱动版本外每个数都出自
  `scripts/gpu_cpu_parity_check.py` 记录的那次运行. 盒内两条后端一致到 `f32`
  累加舍入的量级（最差 3.772e-05 kcal/mol，427 行排序完全一致，
  这正是 `dock-py/python/opendocking/core.py` 里说的「`1e-7` 相对量级」所预期的）.
  盒外那 85 行则全部超出该量级，最差 3222.32 kcal/mol，
  而且两个后端把排序弄反了；`scripts/gpu_cpu_parity_check.py` 再也测不出这个数，
  因为它现在测的是修好的 kernel，盒外最差是 1.890e-03。以上是一次旧构建的读数，
  不是本文的断言。

  原因是 `energy.wgsl` 当时对每个突出原子返回**固定的 `1000.0`**（一个哨兵值），
  而 CPU 是按 Å 逐轴累加的斜坡：GPU 那侧封顶在 `1000 × 出界原子数`，CPU 侧没有上界。
  同一个构象整体沿 x 平移（**修前**读数）：

  | 位移——缺陷记录；今天由 `scripts/gpu_cpu_parity_check.py` 测 | 出界原子 | CPU（修前） | GPU（修前） |
  |---|---|---|---|
  | 6.5 | 2 | 705.851 | 1999.808 |
  | 8.0 | 7 | 7882.264 | **6999.985** |

  位移 8 时 GPU 把一个 CPU 判为很差的姿势判为**更好**——而批量打分器存在的理由
  就是给构象排序，这也正是 `scripts/gpu_cpu_parity_check.py` 守着的那条性质.
  引擎自己的 fixture 又按构造把这个区域排除在外（`dock-core/src/gpu/mod.rs`
  断言 `reach < 6.0`），所以它一直没被测过。（`report_backend` 里本来就带着
  `backend` 字段，所以一次 GPU 运行和一次 CPU 运行**一直分得出来**。曾经会骗人的
  是另一种运行：在没有 `gpu` feature 的构建上 `use_gpu=True` 静默回退、而且不
  给理由——而「我要了 GPU、被拒绝了」和「它本来就在 CPU 上跑」不是同一句话，
  调用方分不出拒绝和请求被忽略。**2026-10-02 修复。** 契约是四个布尔组合上的异或：一份报告要么写着 `gpu` 后端，**要么**带一条非空的 `gpu_skip_reason`，
  两者从不同时出现，也从不同时缺失；所以请求了 GPU 却没跑成的调用一定带理由，
  而 `use_gpu=False` 的调用仍然不带——因为它没有拒绝任何东西。降级有七个活的
  原因，引擎在每处自己写出理由：`this build was compiled without the gpu feature`、
  `empty population`、`ligand has more atoms than the workgroup size`、
  `a ligand coordinate is too large for the f32 grid`、`no usable GPU adapter`、
  `could not pack the batch`、`the compute pass failed`。七个原因需要七种不同的
  应对，这正是 `backend == "cpu"` 单独不够用的理由。契约与断言它的测试都在
  `dock-core/src/search/mod.rs`：`evaluate_population` 的文档注释，以及
  `a_decline_is_always_self_explaining`。）

  核函数现在用与 CPU 完全相同的表达式（逐轴、逐 Å、对三个面求和），量的是**盒的
  面**而不是最后一个制表点——`estimate_dims` 向上取整，制表体积可能略大于盒——
  两边共用 `grid.rs::out_of_box_violation_per_axis` 这一个定义。
  **但源码里看得到修复，不等于修复被证明过**，所以这两道门要分开认：

  - `shader_and_rust_agree_on_the_out_of_box_penalty` 是**文本**检查：它解析常量，
    并要求核函数含有那个逐轴表达式。它抓得住「退回平哨兵」，抓不住符号写反或
    `bmax` 传错——它从不解两边的数。
  - `the_cpu_and_gpu_paths_agree_outside_the_box_too` 才是**数值**检查：它把构象
    一路滑出盒外，逐点比两条后端，并断言没有排序反转。

  后者**在没有适配器的机器上会跳过而不是失败**（打印 `NOT MEASURED` 后直接返回），
  所以它在 CI 上变绿**不构成**这份源码被证明过。**要复核就跑门禁**：
  `scripts/gpu_cpu_parity_check.py` 自己报总数，盒内盒外分开测，在没有适配器的机器上
  跳过且计入总数。
- **不要用 `|grad|` 判断姿势收敛了没有。** 网格是三线性插值，只有 C⁰，
  而返回姿势**总有原子落在网格面附近**.
  `examples/audit_poses.py` 在这棵树上的这次运行里，5 个姿势的 `|grad|` 是 4.0–8.3、
  L2 范数 5.3–12.5，沿 −grad 走一步的能量变化只有 1e-9–1e-8 kcal/mol 量级.
  **这个区间跟着构建和机器走**，不是常数.
  `docs/VERIFICATION.md` 记着并行搜索的走位随线程数变、不同 CPU 返回不同位姿集，
  两次独立构建给出过两个不同的位姿集.
  早先的 2–6 是照 `docs/VERIFICATION.md` 里更早一次 2.3–5.9 的读数往外取整的，
  不是 `examples/audit_poses.py` 的输出，也并不包住它今天报的数.
  要判断极小性请沿线搜索.
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

**GPL-3.0-or-later**，全文见 [`LICENSE`](LICENSE)，这个标识符里的 `3.0` 是许可证编号
而不是对本项目的测量，所以盯着那个文件的是 `scripts/check_text_encoding.py`.
由于分发 GPL-3.0 代码，任何二进制分发都必须同样以 GPL-3.0 提供完整源码，
`Cargo.toml` 与 `dock-py/pyproject.toml` 的 `license` 字段写的是同一个字符串.
功能形式与权重的出处见 [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md)。
