# UQRA PERF-01 Windows/Python 3.12 性能与内存基线

状态：完成

测量日期：2026-08-10（UTC evidence 时间见 JSON）

基线来源：`master` merge commit `7cfb034ceff26c06aadb9843a228d3dfea65e016`

机器可读证据：[`performance/UQRA_PERF_01_BASELINE.json`](performance/UQRA_PERF_01_BASELINE.json)

测量工具：[`../tools/performance/run_adaptive_baseline.py`](../tools/performance/run_adaptive_baseline.py)

## 1. 使用边界

本结果仅用于 UQRA 软件工程性能验收、后续 OPT-01/02 前后比较和回归诊断。它不是论文
性能结论，不是 scientific reproduction，不代表历史 replay，也不改变三个 reduced
benchmark 的 `software_benchmark` / `reduced` 身份。

时间和内存数字只对本 evidence 中固定的 Windows/Python/依赖锁、机器和测量方法有效。
不同机器、负载或依赖版本的绝对值不可直接比较。后续优化必须在同环境重新运行工具，并
同时验证行为 identity；性能收益不能替代完整逐轮行为回归。

## 2. 固定环境与输入

- Windows 11 `10.0.26200`；Python `3.12.13`；UQRA `0.3.0`；
- NumPy `1.26.4`、SciPy `1.15.2`、scikit-learn `1.6.1`；
- 依赖锁：`requirements/compatibility-py312.txt`，SHA-256
  `36e4f13d56b4e820a93dd7b1a06985cf083966255c50ee6d722b3ffcd108f0e0`；
- configs：`four_branch_reduced_v1.json`、`ishigami_reduced_v1.json`、
  `gayton_reduced_v1.json`；
- 每个 benchmark 的 candidate/test/reference seed、shape 和 SHA-256，以及每次运行的
  stable/contract/trace hash，均记录在 JSON evidence；三个重复的行为 identity 完全一致。

## 3. 测量方法

每个 benchmark 重复 3 次，每次启动全新 Python 子进程，不做 warmup。子进程在读取并校验
config 后开始 `time.perf_counter_ns`，计时范围是一次完整的内存内 `run_config`。同时记录：

- `tracemalloc` 在 timed run 中观察到的 Python allocation peak；
- Windows `GetProcessMemoryInfo` 的 process peak working set；
- timed run 前 working set 与进程 peak 的差值，用于降低解释进程基础占用时的歧义。

随后在同一子进程执行第二次相同行为，在 cProfile 下提取 controller、order design、fit、
inner loop、optimal design、DoI、sparse-PCE 和 artifact build 的累计时间。阶段数字是
**包含式 cumulative time，彼此重叠且不可相加**；总耗时来自不启用 profiler 的第一次运行。

工具会拒绝非 Windows/Python 3.12、少于 3 次重复，以及重复间行为 identity 变化。证据绑定
工具 SHA-256 和依赖锁 SHA-256。

## 4. 正式基线结果

以下为三次重复的中位数；MiB 使用 `1 MiB = 1,048,576 bytes`。

| Benchmark | 总耗时中位数 (s) | Python traced peak (MiB) | Process peak working set (MiB) | Peak delta (MiB) | Trace rows | Model calls | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FourBranch reduced v1 | 1.574 | 9.75 | 149.30 | 12.67 | 17 | 20 | `outer_qoi_converged` |
| Gayton reduced v1 | 1.989 | 9.83 | 149.30 | 12.60 | 25 | 30 | `outer_qoi_converged` |
| Ishigami reduced v1 | 6.726 | 9.94 | 150.78 | 14.27 | 37 | 70 | `max_order_reached` |

阶段累计时间中位数：

| Benchmark | Controller (s) | Fit (s) | Inner loop (s) | Optimal design (s) | DoI (s) | Artifact build (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FourBranch reduced v1 | 0.364 | 0.204 | 0.273 | 0.080 | 0.0002 | 0.0042 |
| Gayton reduced v1 | 0.644 | 0.302 | 0.467 | 0.134 | 0.0004 | 0.0047 |
| Ishigami reduced v1 | 2.845 | 1.728 | 2.418 | 0.373 | 0.0011 | 0.0118 |

完整 min/median/max、每次原始记录和所有阶段值以 JSON evidence 为准，表中数字仅作便于阅读
的舍入展示。

## 5. 工程观察与后续门

1. 三个 benchmark 的 Python traced peak 接近，process peak 也集中在约 149--151 MiB；当前
   输入规模下，运行时间比峰值内存更能区分路径成本。
2. Ishigami 的 trace rows、model calls 和总耗时最高；cProfile 显示 fit/sparse-PCE 与
   inner-loop cumulative time 是主要可观察路径。
3. DoI 构造与 artifact build 的累计时间相对较小。该观察只用于选择首批低风险维护工作，
   不能据此改变 DoI、artifact 或数学契约。
4. 首批 OPT-01 应优先处理不改变执行顺序的职责/重复逻辑，并先运行 TEST-01 全部行为门。
5. OPT-02 暂不启动。任何性能优化候选必须用同工具重新测量，报告原始 min/median/max，且
   stable/contract/trace identities 与设计点、逐轮数量、阶次、CV/QoI/Pf、停止原因一致。
6. OPT-03 继续等待单独裁决。

## 6. 复现命令

在 Windows/Python 3.12 锁定环境中，从仓库根目录执行：

```powershell
python tools/performance/run_adaptive_baseline.py `
  --repetitions 3 `
  --output specs/performance/UQRA_PERF_01_BASELINE.json
```

重新生成 evidence 会更新机器与时间相关数字；不得用另一环境的结果覆盖本基线后宣称直接
性能回归或收益。
