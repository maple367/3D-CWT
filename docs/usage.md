# v3 使用与交付

## 安装与运行

Python >=3.10，运行依赖仅 NumPy>=1.24、SciPy>=1.10。实测 Python 3.13.14、NumPy 2.5.4、SciPy 1.18.1。建议新虚拟环境，不要使用指向不同解释器的裸 `pip`。

```bash
python -m pip install .
python examples/square_finite.py --output results/v3-demo
python examples/square_finite.py --preset liang --output results/v3-liang
python scripts/validate_convergence.py --output results/v3-convergence
# 可复用已有扫描目录：只新增 extended_coupling.json/csv，拒绝覆盖这两个文件
python scripts/validate_convergence.py --extended-coupling-only --output results/v3-convergence
python -m pip install '.[dev]'
python -m pytest -q
```

demo 与 liang 使用同一个梁永论文结构（不是两种材料）；扩展 coupling-only 默认 D10/16/24/32、q48/F128，随后同D32单独 q64/F128、q64/F192；仅求4×4无限域谱，不重复有限网格。实测耦合合计200.64秒，仍未达到矩阵D收敛门槛，属预发布验证；指标与限制见 `literature_validation.md`。可用 `--extended-D`、`--extended-q/F`、`--refined-q/F` 控制测试规模，不能把小D smoke 视为高D数据。

demo 与 liang 使用同一个梁永论文结构（不是两种材料）；默认 demo 为 Nx=Ny=8、D3、q12，liang 为 Nx=Ny=16、D10、q24。二者 F64、k8、L=14.75 um=50a、sigma=0.03+0.03j um^-1、prepared 后端。参数和求得的参考频率全部写入 summary，非静默设置波长。默认目录独立于旧结果：`results/v3-demo`。

* `--Nx/--Ny`：物理域每轴网格单元数，非晶格周期数。
* `--L-um`：固定正方形物理域边长，单位 um；本例 50a 是自选数值验证域，不冒称论文有限域尺寸。
* `--D`：辅助 Fourier 模截断，m,n=-D..D 且 m²+n²>1；不是有限域网格。
* `--q`：每材料区间纵向积分阶数（同层 Green 对角尖点采用现有分片/Duffy 处理）。
* `--F`：几何 Fourier 积分控制；本例是解析几何 RightTriangleHole 的 shape_fourier 路径，非用 F×F 像素替代形状。
* `--backend reference|prepared`：相同数学式的逐对参考/预备 Green 实现；默认 prepared，不用 JIT。
* `--k`：目标本征对个数，绝非真空波数 k0。
* `--sigma`：复本征值 lambda=delta+i alpha 的 shift，单位 um^-1，不是频率。负实部写为 `--sigma=-0.03+0.03j`。
* source：内建固定文献材料/几何；不接受旧 model/dill 外部状态。k0 是 beta(k0)=2pi/a 外层 Brent 求根结果；不以放宽 beta_rtol 掩盖失谐。

求解仅返回 shift 附近子集。ARPACK tol=1e-10，原始 pencil Frobenius normwise residual 门槛 1e-9，seed=33、maxiter=10000，ncv 和实际 LU/迭代诊断见 summary.solver。没有自动变 shift、dense fallback、伪零模补齐；失败非零退出，无新成功 manifest。

## 输出与单位

每次四个文件：`modes.csv`、`fields.npz`、`summary.json`、`mode0.svg`。SVG 是真实返回第0模（离 sigma 最近，不保证最低阈值）的包络强度，无 matplotlib 依赖。

NPZ 仅数值数组，`numpy.load(path, allow_pickle=False)`：

* centers: (returned,4,Ny,Nx)，分量 (Rx,Sx,Ry,Sy)，完整复数。
* faces_Rx/Sx: (returned,Ny,Nx+1)；faces_Ry/Sy: (returned,Ny+1,Nx)。入射面严格零。
* C、C1D、Crad、C2D: (4,4) 复矩阵，um^-1；eigenvalues 为完整复 lambda。
* x/y_centers_um、x/y_faces_um 是物理坐标。

P=hx hy sum |Bv|²=1，场 um^-1；强度 sum_components |envelope|² 为 um^-2，**不是**包含微观载波干涉的电磁强度/瓦特。delta、alpha 和预算项单位 um^-1，alpha 保留符号，功率损耗标签为 2alpha。CSV 与 JSON 包含 edge、signed coupling、Crad、闭合误差和 residual；JSON 标清辐射分解适用范围。模式归一化及相位固定不改变 lambda。

非空目录默认拒绝。只有显式 `--overwrite` 且目录带 schema=cwt3d-demo-v1 的 summary 才允许覆盖四个已命名文件，不递归删除其他内容。求解完成后才写数据，summary 最后写；覆盖不是多文件原子事务，运行期间或 I/O 失败后可能暂时混合新旧文件，请勿并发写同目录。需要可事务性采集时始终使用全新目录。

## 缓存与无隐式外部依赖

示例/扫描均不调用磁盘 cache，不读旧 pickle、MPH、GPU 或 numba。显式 opt-in 缓存 API 在 `cwt3d.cache.CouplingCache`；目录由调用者提供并负责信任边界，默认最长7天/256MiB，`prune()` 为显式清理，不假装后台自动清理。过期为 miss；损坏报错，不静默信任。示例未添加缓存 CLI，避免把冷启动计算误报为命中加速。

## 可移交产物与干净环境

交付目录 `../3D-CWT-deliverables`（应位于临时清理范围外）：wheel、source ZIP、主要 demo/liang 结果和15个扫描结果、结果 ZIP、audit.json。临时构建环境和 pytest 数据在 `.validation-scratch`，可按本地临时文件策略清理。

```bash
uv venv .validation-scratch/task13-clean --python .venv/Scripts/python.exe
uv pip install --python .validation-scratch/task13-clean/Scripts/python.exe ../3D-CWT-deliverables/cwt3d-0.1.0-py3-none-any.whl
# 路径相对于仓库根目录；隔离验证时从非仓库 cwd 使用相应绝对路径执行
# -I 不允许从 cwd/PYTHONPATH 偷带包
.validation-scratch/task13-clean/Scripts/python.exe -I ./examples/square_finite.py --preset liang --output NEW_OUTPUT
```

wheel 只包含 cwt3d 包，示例/文档在 source ZIP。source ZIP 包含所有新 tests 及历史回归测试必需的最少 model 源文件，支持运行全套 tests，但不是旧 COMSOL 应用/数据全集；不包含 .git、虚拟环境、凭据、MPH/pth/原始大CSV。未发布到 PyPI；GitHub 源码分支属于预发布，不代表正式 Release。
