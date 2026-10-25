# 3D-CWT v3 Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 修正3D-CWT物理与实现错误，提高可测量性能，并以Python交错网格求解器完成有限尺寸光学本征计算。

**Architecture:** 分离结构/材料、纵向波导、Green核、耦合系数和有限域求解。先建立低速可信参考实现，再加入可选优化后端；有限域离散保留为稀疏广义非Hermitian矩阵铅笔，不显式求逆。

**Tech Stack:** NumPy、SciPy；Numba可选；pytest；基准用perf_counter及峰值内存测量。首版CPU complex128，不以GPU或某种新框架作为正确性前提。

**Status:** 设计建议，尚未实施或修改仓库。新版本目录/分支待用户确认；不推送、不发布。默认首版TE正方晶格；三角晶格和载流子时域自洽模型为后续独立范围。

---

## 一、不可变的数学与工程约定

1. 明确exp(+iωt)、面内exp(−iG·r)、xi积分正指数、长度单位、四波顺序、纵向场归一化。
2. 推导依据：Liang博士论文 DOI 10.14989/doctor.k18283 第3章(3.27)–(3.35)、第4章(4.21)–(4.22)；2011作者预印本 arXiv:1107.1772。2011任意侧壁文章 DOI 10.1364/OE.19.024672 尚缺全文，复杂侧壁功能不能直接宣称逐式认证。
3. Green核必须通过源外ODE、源点导数跳跃、界面连续性和外向/衰减边界测试。使用独立的辐射kz与高阶衰减gamma约定；分层核用左右满足边界的齐次解及Wronskian构造，避免原代码式混用指数基底。稳定实现可使用散射/反射递推，不能盲目相乘增大指数的T矩阵。
4. 多层源端xi(z′)与接收端xi(z)分开；不能默认加权平均系数乘已积分场是任意纵向结构的严格结果。
5. C矩阵全部为逆长度。chi必须含−k0²/(2beta0)；无量纲化只能在边界明确进行并回到有量纲结果。
6. 交错网格法属于有限差分离散；有限元与有限差分应近似同一个包络PDE，但不需要复制其有限元矩阵。
7. 接口区分物理尺寸L、周期a、网格Nx/Ny、高阶截断D；不沿用含混的size/resolution。
8. 旧SEMI_solver的载流子输运不是这次光学求解器的隐含移植内容，若需保留其功能需独立确定范围。

## 二、建议模块

```
src/cwt3d/
  conventions.py
  geometry.py
  materials.py
  waveguide.py
  greens.py
  quadrature.py
  fourier.py
  coupling.py
  infinite.py
  finite/grid.py
  finite/assembly.py
  finite/eigensolver.py
  finite/fields.py
  results.py
  cache.py
benchmarks/
tests/
examples/square_finite.py
```

接口不要把不可序列化的积分闭包、锁、磁盘缓存和物理模型放在同一对象。模型初始化不自动保存或启动进程。

## 三、依次执行的任务与门槛

每项按“写最小失败测试 → 单独运行确认失败 → 最小实现 → 运行通过 → 审查diff”执行。以下命令为未来实施命令与预期门槛，不是已运行测试。

### Task 1：建立包与约定
- Create: `pyproject.toml`、`src/cwt3d/__init__.py`、`src/cwt3d/conventions.py`。
- Test: `tests/test_import.py`、`tests/test_conventions.py`。
- 在干净环境验证核心导入；用显式单位换算和Fourier重构检查符号。
- Run: `python -m pytest tests/test_import.py tests/test_conventions.py -q`。
- Gate: 缺失可选Numba时有明确回退。

### Task 2：冻结原错误回归
- Create: `tests/regressions/test_chi_prefactor.py`、`test_green_operator.py`、`test_fourier_dft.py`。
- 首先针对旧方法隔离执行，再针对新API写同一物理断言；保存两种输出，禁止把旧结果作为唯一真值。
- Gate: 旧错误确实触发；新版本不能通过绕过测试。

### Task 3：几何与Fourier基准
- Create: `geometry.py`、`fourier.py`；Test: `tests/test_fourier.py`。
- 圆孔用Bessel解析式（零阶极限显式处理），一般形状用直接高精度积分为参考；三角孔先可信积分，解析式通过极限测试后再加。
- FFT路径测试非对称/平移单胞，校验xy顺序、正负号、采样中心相位和分辨率收敛。仅在实介电常数条件成立时使用共轭对称缓存。
- Run: `python -m pytest tests/test_fourier.py -q`。

### Task 4：平均波导
- Create: `waveguide.py`；Test: `tests/test_waveguide.py`。
- 用解析对称平板根、界面场连续性、归一化尾部收敛独立验证。先写失败断言，再实现稳定传播。
- 首版优先显式给定a/k0的计算入口，旧自动选a方式可作为独立设计辅助函数保留。

### Task 5：均匀Green核
- Create: `greens.py`；Test: `tests/test_greens_homogeneous.py`。
- 分别实现辐射/倏逝分支；有限差分检查源外ODE，单侧解析导数检查跳跃条件；测试复数介电常数及平方根分支。
- Run: `python -m pytest tests/test_greens_homogeneous.py -q`。

### Task 6：分层Green核
- Modify: `greens.py`；Test: `tests/test_greens_layered.py`。
- 左右边界齐次解/Wronskian作为数学定义；实现稳定层间递推。检查均匀介质退化、互易性、界面条件和独立一维离散参考解。
- 多PC层/不同孔形源必须加入测试，不能只用单层常xi掩盖坐标错误。

### Task 7：参考耦合矩阵
- Create: `quadrature.py`、`coupling.py`、`infinite.py`；Test: `tests/test_coupling.py`。
- 对照论文实现κ、μ、ν、varsigma、ζ、χ，保留各矩阵分量及计算误差信息。
- 单层/双层、平移不变性、长度尺度变换、对称性、截断与纵向积分收敛分别测试；无损条件下检查应Hermitian部分及辐射功率平衡，不误要求整个C Hermitian。
- 所有C项尺度变化必须一致；不能以加fastmath或取绝对值隐藏符号错误。

### Task 8：交错网格索引及边界消元
- Create: `finite/grid.py`；Test: `tests/test_grid.py`。
- 明确四分量各自的位置、索引、插值目标和Nx/Ny的含义；往返field/vector测试必须精确通过。
- 约束Rx左、Sx右、Ry下、Sy上入射端为零，用消元而非1e8惩罚。矩形非方网格、最小网格和角点单独测试。

### Task 9：直接稀疏装配
- Create: `finite/assembly.py`；Test: `tests/test_assembly.py`。
- 从局部模板直接构建COO再转CSC/CSR，输出A,B及索引元数据；禁止先建稠密矩阵再转换。
- 对小网格与独立手算/朴素装配比较；用制造解检验导数符号、插值及预期精度阶。
- nnz随自由度线性增长；这不代表LU内存也线性增长，基准要记录填充。

### Task 10：非Hermitian广义本征求解
- Create: `finite/eigensolver.py`；Test: `tests/test_eigensolver.py`。
- 数学问题A v=lambda B v；不假定B Hermitian正定。
- 对每个shift分解K=A−sigma B一次；定义T(x)=solve(K,Bx)，对T用ARPACK的eigs(...,which='LM')，回算lambda=sigma+1/mu。不显式构造B逆或K逆。
- 不把sigma再次传给T上的eigs，避免二次shift变换。
- 小问题与scipy.linalg.eig(A,B)对比；测试非Hermitian B、奇异/近奇异shift及ARPACK未完全收敛，明确返回已收敛子集和诊断而非零结果。
- 验证原始矩阵铅笔的残差，不只检验T的残差；过滤无穷/伪本征值。
- SciPy的eigs(A,M=B)要求M满足Hermitian及正定/半正定条件，不适用于任意交错插值矩阵，详见官方文档。

### Task 11：场、功率与模式跟踪
- Create: `finite/fields.py`、`results.py`；Test: `tests/test_finite_modes.py`。
- 返回复本征值、四分量复包络、功率/损耗分解、归一化、残差和收敛标记；不能只返回一张强度图。
- 模式匹配采用场重叠/简并子空间，不按每次eig输出索引固定模式。
- 验证包络PDE的积分能量收支；定义清楚alpha幅度阈值、功率损耗2alpha及其垂直/侧向分解。

### Task 12：加速与缓存
- Create: `cache.py`、`benchmarks/bench_coupling.py`、`bench_finite.py`；Test: `tests/test_backend_parity.py`。
- 先建立reference结果。共享分层积分节点，预计算Θ、xi和Green块，用批量矩阵收缩代替大量Python回调与重复自适应二重积分；对角线不光滑Green核用分区积分或匹配的高阶规则验证收敛。
- Numba用于纯数值热点、稀疏模板装配；默认complex128，不默认fastmath。
- 缓存按结构、k0/a、积分精度、D、代码版本生成key；worker返回内存结果，主进程一次落盘。避免每个系数读写整份npy和多进程/BLAS线程超额并行。
- 报告冷启动（含JIT）、热运行、缓存命中、装配、LU、Arnoldi、峰值内存、nnz和LU填充，固定相同误差门槛再比较耗时；不给无测量加速倍数。

### Task 13：端到端与文献复现
- Create: `examples/square_finite.py`、`tests/test_end_to_end.py`、`docs/validation.md`。
- 干净环境运行：结构→纵向模→C→有限域本征模→四分量场→损耗和报告。
- 固定博士论文表3.1/第43页目标参数；注意该旧脚本目前实际加载外部历史文件，不能把头部注释当已核实运行输入。
- 分别扫描Fourier截断、纵向积分精度、平面网格、边界策略及shift覆盖。网格/截断收敛和求解器残差分别记录。
- 对公开文献图/表复现时记录数字化误差和预印本/期刊版本，不能凭图片目测声称高精度一致。

## 四、已完成的可行性小实验（不是有限尺寸求解器）

执行了一个40阶合成复非Hermitian广义本征问题，其中B非Hermitian。采用上述稀疏LU+LinearOperator变换求4个特征对，并与dense generalized eig独立对照：
- max generalized residual: 1.024437536573842e-16
- max distance to dense reference eigenvalues: 7.667477763817488e-16

该结果只验证广义shift-invert求解策略；未验证PCSEL离散、边界条件或文献器件结果。

## 五、验收

- 干净环境完成完整光学例子。
- 原始公式不一致回归与边界/场索引测试通过。
- 非Hermitian广义本征问题与独立小矩阵参考一致，原始残差可追踪。
- 有限域网格、辅助波截断和积分精度分别收敛；能量平衡在离散误差内闭合。
- 加速与reference在同一误差标准下比较，公开真实时间与内存，未测量不得声称更快。
- 不覆盖旧历史结果；新结果写独立版本目录，不将异常/失败写为数值零。

Sources:
- Current repository: `4b3b063308e71f49973510e68b8d772b8eb0f1ea` (legacy baseline in this repository)
- Liang thesis: https://doi.org/10.14989/doctor.k18283
- 2011 author manuscript: https://arxiv.org/abs/1107.1772
- SciPy eigs requirements: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
