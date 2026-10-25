# Task 12：prepared Green、安全显式缓存与实测性能

## 范围与实现

基于 `v3-refactor` / `82e55e8` 的未提交修改；没有改动 legacy、Fourier 数学、模式求解或有限域求解器，没有 commit/push。本任务不包含 Task 13 的文献器件复现或完整 CLI 文档交付。

- `build_coupling(..., backend='reference')` **仍为默认回归锚点**；显式 `backend='prepared'` 启用 `PreparedLayeredGreen`。
- 每个 kernel 的固定 `_nodes` 仅组装一次相同 DtN 系统，执行一次相同 SVD pole gate，再做一次 SciPy LU。每次广播调用去重 source 坐标，以多 RHS `lu_solve` 求解；没有显式 inverse。接收 shape 复用，same-element 的局部源 Green 保持原公式。
- 两 backend 共享原来的波导模式、source/receiver 层 xi、按 `m²+n²` 复用的 radial overlap、固定 q/F、同层 Duffy 两三角积分；未使用降低精度的加速。
- 支持广播、receiver derivative 及 source left/right limits、界面/半空间、内部 cutoff、passive complex epsilon、厚层及 pole 诊断。prepared 在**构造时**发现 pole，而 reference 在首个实际坐标对求值时发现；prepared 即使空 batch 也已执行 pole gate。
- LU/pivot 为 immutable bytes；调用返回新数组。继承 reference 的最多 512 nodes 及原数值动态范围限制。准备存储为 O(nnode²)；一次求值的 RHS/解为 O(nnode × unique_sources)，另外有广播输出与 receiver shape/source 索引。非常大的广播请由调用者分批，不承诺无限 batch 内存。
- 全程 complex128，无 Numba、无 JIT、无 fastmath、无新依赖。**JIT cold compilation = N/A**；下文 cold/steady 指实际构造/求值或磁盘 miss/hit。
- 顺带补齐 `CouplingResult.fourier_order`、`backend`、`sampling_shapes`；极端 period 导致非有限/未解析 beta0² 时提前明确失败。

## 显式缓存 API 与安全边界

```python
from cwt3d.coupling import build_coupling
from cwt3d.cache import CouplingCache

reference = build_coupling(mode, period_um, pc_layers)  # 默认不读写磁盘
cache = CouplingCache(cache_dir)                       # 构造也不 mkdir/write
result = cache.build(mode, period_um, pc_layers,
                     cutoff=3, quadrature_order=12,
                     fourier_order=64, backend='prepared')
key = cache.key(mode, period_um, pc_layers,
                cutoff=3, quadrature_order=12,
                fourier_order=64, backend='prepared')
readonly = CouplingCache(cache_dir, readonly=True)
snapshot = readonly.load(key)   # 缺失/过期返回 None；损坏抛 CacheError
cache.clear(key)                # 只删除这个 key
cache.prune()                  # 显式调用，非隐式后台清理
```

用户必须提供 `cache_dir`；不使用仓库目录默认值或系统共享位置。只读实例禁止 store、miss 重算写入、clear 和 prune。并发 miss 可能重复计算；atomic replace 仅保证不会暴露半写快照，不保证并发调用成功。Windows 打开的 reader 可能阻止 replace，产生 PermissionError/WinError 5；并发 load/store/clear/prune 的竞态不由此 API 解决。调用者须串行访问或自行重试文件系统错误，没有跨进程锁或自动 retry。

**Key**：canonical JSON + SHA256，含完整 stack/k0/interfaces、period、beta、实际 mode `_value`/归一化 field 与插值 mesh/materials 的 dtype/shape/bytes digest（不是仅 index 或 beta）、PC layer indices、circle/triangle 的全部几何和 complex epsilon；sampled 的网格尺寸/dtype/数据 bytes digest；D/q/F/beta_rtol/backend；实际 `src/cwt3d/**/*.py` 代码内容 digest、schema、NumPy/SciPy 版本、字节序。没有 callable repr、对象 id、路径或时间戳假充稳定身份。源码 digest 有意保守：相关包内任意 Python 源码变化均 miss，包括文档字符串更改。

**格式**：单文件 NPZ 内 numeric arrays + uint8 JSON manifest，始终 `allow_pickle=False`。读前检查 ZIP 条目白名单/重复条目/解压字节预算及每个 NPY header 的 shape × itemsize 与真实 payload 大小，拒绝 object dtype 和伪造巨型 shape；再核验 key document、数组 checksum/shape/dtype 与 metadata checksum。返回原来的 bytes-backed `CouplingResult`，包括所有诊断数组与 tuple 元数据。错误/不匹配文件抛 `CacheError`，不会作为 hit；过期或数值版本/算法过时为 miss。损坏文件需显式 `clear(key)` 后重算。

**标量 period 政策（R2）**：直接 build 与 cache document/key/build 共用 `_positive_scalar` 的有限正实标量验证；相同数值的 Python float、NumPy float64 与合法 0D ndarray 都规范为 Python float，再编码为同一 float hex。它们的 key、完整 provenance 和 metadata 相同，支持交叉 store/load/hit，不保留无必要的原始 period dtype/container 身份。非标量、complex（即便虚部为 0）、bool、非正和非有限值仍拒绝。该规范仅针对 scalar period，不折叠几何或 mode field 的 dtype/shape/bytes，完整输入误配保护不变。

**结果绑定与公开 store**：`build_coupling` 在计算前从真实输入生成 `CouplingResult.provenance`，其值为不可变 canonical JSON 字符串；完整包含上列 key 所需内容，不从 xi 或 beta 推断几何/模式。frozen dataclass 禁止普通赋值，调用者修改 `cache.document(...)` 返回字典不会改动结果快照；数组仍直接以 immutable bytes 保存。`store(key, result, document=...)` 保持原签名，写前必须逐字匹配结果 provenance 与 key document；load 在 metadata checksum 后再次匹配。不同几何、same-beta 不同 mode field、beta_rtol、precision/backend/stack 等普通误配均拒绝；手工构造且缺少 provenance 的结果不能 store。无需重算 C 验证缓存。所有直接 build 也执行 provenance/source digest，因此本轮重新测量完整 coupling/cache/finite，而非替换旧报告 digest。

**写入**：在用户指定目录内创建唯一私有临时文件，flush/fsync 后 `os.replace`；正常失败会清理临时文件，旧快照保持完整。无 pickle、无系统 temp 写入。进程硬终止可能留下 `.cwt3d-*.tmp`，prune 不冒险自动删除这些文件。

默认 max_age=604800 秒（7 天）、max_bytes=268435456（256 MiB），均可配置；size 预算约束每个 entry，显式 prune 也按该预算约束识别条目的磁盘总量。prune 只识别规范命名、通过 ZIP/NPY layout 检查且具有本 schema/有效 key document 的文件，先过期再最旧；**不执行数组/metadata checksum 或 provenance 完整审计**。因此 ZIP/layout/key 仍合法但 checksum 损坏的已识别 entry 也可因过期或超 size 被删除，不保证保留所有 corrupt 文件。无关文件、不通过识别的损坏/伪装文件、symlink 与 crash-left temp 均不删除。`clear` 仅精确 key，不递归清目录。

**信任边界**：checksum 不是签名。能同时重写数组和 manifest 的目录写入者可伪造科学结果；不要把共享不可信写目录视作认证数据库。frozen dataclass/provenance 也不是防伪签名；刻意以 `dataclasses.replace` 或私有状态篡改伪造结果不在该普通 API 误配保护的保证内。安全读取防止 pickle 代码执行、意外损坏误命中和常见 header/ZIP 资源攻击，不声称对所有恶意文件/文件系统竞态提供沙箱。

## 可重跑命令与环境

在仓库根目录（Windows Git Bash）执行：

```bash
.venv/Scripts/python.exe benchmarks/bench_coupling.py --cache-dir .validation-scratch/task12-cache --repeats 3
.venv/Scripts/python.exe benchmarks/bench_finite.py --repeats 3
.venv/Scripts/python.exe -m pytest -q --basetemp=.validation-scratch/task12-full-pytest
git diff --check
```

正式小报告为 `docs/benchmarks/{coupling,finite}.{json,csv}`，保留每次样本，不只保留最好一次。脚本从独立进程运行，导入 NumPy 前固定 OpenBLAS/MKL/OMP/BLIS/VecLib/NumExpr 线程环境为 1；没有测/锁 CPU 频率或独占系统。顺序执行 coupling 和 finite，避免两个 benchmark 相互竞争。硬件 Windows 11 AMD64、AMD Family 25 Model 97 Stepping 2、16 logical CPUs；Python 3.13.14、NumPy 2.5.4、SciPy 1.18.1。报告没有主机名、账号、个人路径或 token。

R2 源码冻结后完整重跑时 digest：`ca0a55fec89c1e6195bd2cab977ccd896b66dd36923db0b0d905bba5a615de1c`。上一轮四份原始报告已原样归档到 `docs/benchmarks/historical-3bccd999a638/`（复制前后 SHA256 一致），其 JSON 仍标记 `3bccd999a6383a8ce9db9585e5d44d19652879e068f287c18b2f969bc13b026d`。更旧的 `historical-f8bd39fa2326/` 保留；历史报告不代表当前源码。最终表来自同一轮实际运行，不跨轮选择最好样本。微秒量级结果受系统调度影响；一次较早试跑的单层 D10 中位加速曾降到约 1.08×，不能当作稳定硬件上限。

## Coupling：相同精度的六组实测

每个 backend warm-up 1 次，再测 3 次取中位数；全部 D10 reference 均真实完成 3 次，未省略。q=12，F=64。单 PC：偏心圆；双 PC：不同层的偏心圆 + 不等腿直角三角形。模式和材料配置可直接在 `benchmarks/common.py` 重建。

| 几何 | D | reference ms | prepared ms | speedup | cache cold miss ms | cache hit median ms |
|---|---:|---:|---:|---:|---:|---:|
| 单层圆 | 1 | 34.279 | 15.408 | 2.225× | 20.087 | 7.543 |
| 单层圆 | 3 | 118.899 | 58.932 | 2.018× | 64.487 | 7.448 |
| 单层圆 | 10 | 750.162 | 395.607 | 1.896× | 420.910 | 7.619 |
| 双层圆/三角 | 1 | 106.743 | 34.677 | 3.078× | 39.393 | 8.273 |
| 双层圆/三角 | 3 | 388.615 | 145.615 | 2.669× | 147.181 | 8.861 |
| 双层圆/三角 | 10 | 2460.089 | 955.800 | 2.574× | 962.256 | 7.849 |

六组的 `max(abs(C_reference-C_prepared))` 和四个 eigenvalues **集合最优匹配**误差均实际为 0（本组数据恰好 bytes 数值一致，不承诺所有结构 bitwise 相等）。全数组 backend tests 使用容差而非依赖巧合零差。eigenvalue 使用 Hungarian assignment，未按 raw index 比较。

独立 q 收敛基准采用 **reference q24**（另算 q18，D/F/geometry 不变），不是以同 backend 自洽当物理准确性证明：

| 几何 | D | q12 vs q24 max matrix error | q18 vs q24 max matrix error |
|---|---:|---:|---:|
| 单层圆 | 1 | 2.824e-17 | 1.694e-16 |
| 单层圆 | 3 | 2.203e-16 | 1.831e-16 |
| 单层圆 | 10 | 1.824e-11 | 2.253e-16 |
| 双层圆/三角 | 1 | 4.655e-17 | 6.645e-17 |
| 双层圆/三角 | 3 | 4.077e-17 | 6.631e-17 |
| 双层圆/三角 | 10 | 2.629e-14 | 7.295e-17 |

单层 D10 的 q12 误差并非机器精度；q12 vs q24 匹配 eigenvalue 误差约 2.447e-11 um^-1。若要求更严的物理积分精度，应提高 q，两 backend 同步提高；不能以 parity=0 宣称积分误差=0。表中固定 q12 为公平性能工况，而非全参数精度证书。低 D 的 q18 差值受 roundoff 主导，不强行宣称逐点单调收敛。

### 初始化、steady 与不加速案例

12×12 广播求值中位数（reference / prepared）：单层 8.748 / 1.323 ms；双层 9.895 / 1.290 ms。构造中位数：单层 35.500 / 152.600 us，双层 30.400 / 108.000 us，prepared 构造更贵。

**单个坐标对、每次重新构造的 cold 总耗时确实没有加速**：单层 117.600 / 181.800 us；双层 131.900 / 188.600 us。prepared 为批量积分/重复调用服务，不推荐为一次 scalar 求值创建 prepared kernel。源码形状/局部 Green 仍含 Python 循环，Fourier、mode sampling、Duffy 与四波矩阵汇总并未被全部向量化；未预设或宣称 10×/100×。

缓存每条 cold-miss 前只清理该精确 key；hit 的 3 次读取均与 cold 及无缓存 prepared 的**完整 result**相等。命中耗时包含 source/key digest、ZIP/NPY 安全校验和重建快照。测后已删除本轮精确 key，实测 cache 目录为空；scratch 主机策略对 idle 24h 项目另行清理。正式 JSON/CSV 不含 cache 内容，git 不加入临时 cache。

## 稀疏有限域：真实 assembly/LU/iteration/残差/内存

使用双 PC prepared C，D3/q12/F64，固定域 40×40 um；每级 warm-up 1 次、计时 3 次，再单独做 tracemalloc run。三网格完全相同 `sigma=0.03+0.03j, k=6, tol=1e-9, residual_tol=1e-9, seed=33, ncv=30, maxiter=5000`。均返回 converged；未改写数学求解器、未以错误 dense/legacy 结果作速度或正确性基线。

| Nx=Ny | DOF | assembly ms | LU ms | iteration ms | solve total ms | nnz A/B | nnz L+U | solves | max residual | traced peak bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 0.3800 | 0.2739 | 4.2944 | 7.0231 | 1920/480 | 5692 | 78 | 4.245e-17 | 507838 |
| 16 | 1024 | 0.4839 | 1.1674 | 22.3962 | 27.9693 | 7936/1984 | 33478 | 160 | 5.254e-12 | 2181085 |
| 32 | 4096 | 0.8822 | 5.7401 | 96.4678 | 113.1658 | 32256/8064 | 193092 | 201 | 1.530e-12 | 9945287 |

LU fill = (nnz L+nnz U)/nnz(A-sigma B) 分别约 2.965、4.218、5.986，完整样本在 JSON。`solve total` 包括输入快照/验证/残差评估，不等于 LU+iteration；assembly 另外测量。残差为原 pencil 的 Frobenius normwise 残差，不是 transformed operator 残差。网格变更后的近 shift 子集不一定对应相同物理 mode，本表不宣称跨网格跟踪/谱收敛或完整器件验证。

`tracemalloc_peak_bytes` 来自另一次独立 traced run：Python/NumPy 已注册分配，不是进程 RSS，也未涵盖所有 native SuperLU/BLAS 内存。**RSS 未测，JSON 显式 null**。不得把 traced peak 当作总内存上限。

## 验证与剩余边界

- 首先实际运行新 backend tests：4 组 coupling 因缺失 backend keyword 失败，backend validation 也失败；prepared 独立 API 当时未实现。随后实现后通过。cache 初次 RED 为模块缺失，仅算 scaffold，不冒充物理行为 RED。随后新增行为测试实际发现 store 接受 q5 result 配 q4 key（DID NOT RAISE CacheError）；增加写前/读后 metadata 与 key provenance 一致性验证后，该 RED 转 GREEN。
- 测试覆盖 full-array/matrix/eigenvalue parity、once-only SVD/LU、多 RHS、independent interface oracle/ODE/source jump/flux/reciprocity、near pole/thick/complex/strict-underflow、cache roundtrip/immutable metadata、逐项 key 变化/不同真实 mode、sampled bytes/shape、源码与 NumPy 版本、损坏/不匹配/object payload/巨型 header、atomic failure/readonly、age/size cleanup 不误删无关文件。
- 新 sampled metadata 测试曾使用不够大的 2×2 采样导致既有 Nyquist gate 正确拒绝；修正为 8×8 fixture，没有放宽生产 gate。
- R1 行为 RED：`pytest -q tests/test_cache.py -k 'real_input_mismatch or wrong_precision'` 实际 **3 failed, 3 passed, 7 deselected**；失败为不同几何、same-beta 不同 field、不同 beta_rtol 的真实计算结果被 store 接受（DID NOT RAISE），period/q 的旧保护已通过。绑定完整输入后该路径 GREEN，另加只读 provenance、字典隔离、完整 dataclass/bytes roundtrip、缺失 provenance、load provenance 与 checksum 独立验证。
- R2 行为 RED：新增 scalar period/invalid scalar/prune policy 回归后、修 source 前，实际 **12 failed, 5 passed, 14 deselected in 1.28 s**。reference/prepared 的合法 0D period 直接 build 成功，cache.build 均真实触发 `KeyError: 'float'`；另 10 个失败暴露 key/document 未验证 invalid period。修复后完整 cache suite **31 passed, 1 warning in 1.70 s**，包括 scalar 跨表示完整 result/provenance/key 相等、public store/load、mock 禁止重算的 hit、精确 clear 回读 None 及无关文件保留。既有 geometry/field/provenance 和 atomic-failure 回归继续通过。
- R2 prune 真实损坏测试：修改 C 数值但保留原 checksum，load 拒绝；fresh entry 不 prune，过期后该已识别 entry 被删除，malformed ZIP 与无关文件仍保留。此行为是保留策略，不是全面 integrity audit。
- 最终全套 **544 passed, 6 warnings in 8.84 s**；两 benchmark 均以原命令 repeats=3 顺序重跑，六组 reference/prepared 与三有限网格样本齐全，cache 目录运行前后均为空。程序化审核当前 source digest、JSON/CSV 行与逐字段一致性、repeat 数、median、speedup 和 LU fill；上一轮四报告复制前后 SHA256 验证一致。
- 全测试与 diff-check 的最终执行记录见 `docs/validation.md`。仅有刻意测试 ndarray 元数据修改触发的 NumPy deprecation warnings。
- 仍需独立规格审查与质量审查；本报告不把实现者自检称为双审通过。
