# M2.3 多问题 Benchmark 本地验收摘要

状态：已完成；本地与远端 required CI 均通过
日期：2026-08-05

## 范围与声明边界

本验收只覆盖 `purpose: software_benchmark`、`scale: reduced` 的软件行为。
FourBranch、Ishigami 和 Gayton 数据均为新生成的冻结 fixture，不是历史数据恢复，
不得声明为 `historical_replay` 或 `paper_production`，也不支持论文精度或效率结论。

## 三基准矩阵

| Benchmark | 保护行为 | 输入契约 | Candidate / Test / Reference |
| --- | --- | --- | --- |
| `four_branch_reduced_v1` | 多分支失效域与 DoI | 标准正态；`min(g1..g4) <= 0` | 192 / 256 / 4096 |
| `ishigami_reduced_v1` | 非线性与 `x1*x3` 交互 | 独立 `Uniform[-π,π]`；`a=7, b=0.1` | 256 / 384 / 4096 |
| `gayton_reduced_v1` | 局部可靠性失效域与 DoI | `Z~N(0,I)`；`X1=Z1, X2=Z2+3` | 192 / 256 / 4096 |

Gayton 的移位分布只用于 reduced 软件路径覆盖，明确不是论文分布。

## 冻结身份

| Benchmark | Candidate SHA-256 | Test SHA-256 | Reference SHA-256 |
| --- | --- | --- | --- |
| Ishigami | `dcd094b4...a4c6a` | `ee18a46c...e93c3` | `2e28f08f...0dff0` |
| Gayton | `3a80fce3...da80f` | `4d202ea0...8f629` | `d12f4e3c...169f3` |

完整 hash 和 seed 由对应实现模块中的 `INPUT_HASHES`、`SEEDS` 固化并在运行时自检。

## 固定结果

| Benchmark | Reference metric | Contract trace hash | Contract manifest hash |
| --- | --- | --- | --- |
| Ishigami | variance `13.814374748854757` | `697fb5aa...a91c` | `37ebf3d4...daad7` |
| Gayton | failure probability `0.108642578125` | `fdf6a793...0de9` | `db980133...be02` |

Raw trace 和 stable manifest 仍保留，用于同一运行环境内的逐位重复性诊断。跨平台
required gate 使用 contract hash：它保留状态转换、阶次、样本/DoI ID、预算、停止
原因和候选池身份，排除会随 BLAS/LAPACK 产生末位差异的 `cv_path` 与 `qoi` 浮点值。

## 本地验收

- Python 3.11：`52 passed`；
- Python 3.12：`52 passed`；
- 三个 benchmark 只能通过静态 registry 名称选择；
- config v1 不能选择 v2 registry benchmark；
- config v2 schema 的 benchmark 枚举与 registry 一致；
- candidate、test、reference 的 seed 和 SHA-256 相互独立；
- raw manifest/trace 在同一平台重复运行一致；跨平台结构行为锁定为 contract hash；
- 三个 benchmark 均触发 DoI 路径并通过 manifest v1 契约验证。

完成门证据：PR #7 的 Python 3.11/3.12 与 `Adaptive compatibility gate` 通过；
cross-platform contract hash 修复提交为 `7e2bdc8`。
