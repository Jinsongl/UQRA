# REG-01：U3 行为回归结案矩阵

## 1. 目的与判定口径

本矩阵统一记录 U3 中已经验证的行为、证据位置、比较口径和不能验证的历史项。
它汇总既有证据，不把 reduced 软件 benchmark 解释为博士论文历史 replay，也不以最终
失效概率接近替代逐轮 trace 等价。

状态定义：

- `verified`：存在可重复自动化测试或冻结、可核对的逐轮证据；
- `expected-difference`：差异属于已经裁决的 legacy 缺陷修正，预期不得相等；
- `recovered-related`：恢复了相关历史材料，但身份不足以证明 canonical 输入或逐轮结果；
- `unavailable`：所需历史身份或状态未归档，现有证据不能恢复或推断。

## 2. 已验证行为矩阵

| 行为 | 状态 | 比较口径 / 容差 | 固定证据 | 自动化入口 |
| --- | --- | --- | --- | --- |
| 共享二维三阶 Hermite 输入 | `verified` | 候选、测试和训练数组及 SHA-256 完全一致；重复构造只读 | 候选 `d039c83d…1406`；测试 `2ed85b45…34db`；训练联合 `3cd85228…8b10` | `tests/compatibility/test_adaptive_behavior_regression.py` |
| legacy 权重与预处理 | `verified` | 加权设计矩阵、加权响应、offset、scale 逐元素完全一致 | 最大绝对误差均为 `0.0` | 同上 |
| LARS 进入路径、六折 CV、截断 | `verified` | path、fold ID、截断长度完全一致；CV 数组绝对容差 `1e-15` | 路径 `[0, 2, 3, 4]`；CV 最大绝对误差 `0.0` | 同上 |
| RRQR 初始设计 | `verified` | 全局 ID 完全一致 | `[16, 9, 40, 34, 24, 3]` | 同上 |
| D-optimal 三轮全候选评分 | `verified` | 选中 ID 完全一致；评分 `rtol=atol=2e-12` | ID `[21, 29, 33]`；最大误差 `3.552713678800501e-15` | 同上 |
| S-optimal 三轮全候选评分 | `verified` | 选中 ID 完全一致；评分 `rtol=atol=2e-12` | ID `[6, 21, 35]`；最大误差 `8.881784197001252e-15` | 同上 |
| DoI 候选及局部到全局映射 | `verified` | 中心、排序后全局 ID 和映射逐项完全一致 | 中心 `[15, 36, 38, 37]`；21 个冻结全局 ID | 同上 |
| IDX-01 跨阶身份 | `expected-difference` | legacy 局部整数错指坐标；现代稳定全局 ID 的坐标错配必须为零 | 阶段 7 trace `da31f4be…ea31`；阶段 9 为 `66/66` legacy 错配、现代 `0` | `test_adaptive_behavior_regression.py`、`test_adaptive_history.py` |
| IDX-02 DoI 身份 | `expected-difference` | legacy 直接追加局部行号；现代实现必须先映射到全局 ID | 阶段 9 为 `32/32` legacy 错配、现代 `0`；诊断 `1e275fe9…595e` | 同上 |
| 累积观测、模型调用和 literal 停止 | `verified` | 每阶 ID 单调累积；调用数等于唯一 ID 数；候选哈希不变；状态与原因完全一致 | 阶次 `[1, 2, 3]`；`completed` / `literal_orders_completed` / order `3` | `test_adaptive_behavior_regression.py` |
| 三个 live reduced benchmark 身份 | `verified` | contract、trace 和 canonical JSON bytes/hash scope 由回归测试固定 | 身份仅为 `software_benchmark` / `reduced` | `tests/compatibility/test_adaptive_identity.py` |
| publication/portable 运行身份 | `verified` | manifest、trace、CLI JSON 和来源身份重复运行一致 | 冻结 manifest 与 Phase 11 摘要 | `tests/compatibility/test_adaptive_publication.py` |

完整数值、fold、DoI ID 和基线源码哈希见
[`ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md`](ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md)；
IDX 量化及历史材料身份见
[`ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md`](ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md)
与 [`ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json`](ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json)。

## 3. 历史证据结案矩阵

| 历史项 | 状态 | 可用证据 | 结案边界 |
| --- | --- | --- | --- |
| canonical FourBranch 约 10⁵ 候选池 | `unavailable` | 原配置目录缺失 | 不得用重新采样或 reduced fixture 替代 |
| canonical FourBranch 约 10⁶ 测试集 | `unavailable` | 未找到匹配 `McsE6R0` 的四分支文件 | 相关正态 MCS 池不能证明 canonical 身份 |
| canonical 输入 bundle | `unavailable` | LEG-01 在 Python 3.8/3.9 均复现入口缺文件阻塞 | 不能启动历史入口的完整数值执行 |
| 历史逐轮选点、QoI、Pf、停止位置 | `unavailable` | 仅恢复汇总数组，无逐轮 trace | 不得从汇总结果倒推逐轮行为 |
| 历史 RNG、候选排列和 CV shuffle 状态 | `unavailable` | 未归档 Python/NumPy 状态或排列 | 不得声称随机路径重现 |
| 正态/均匀 MCS 池及 `Branches_p10_*` 汇总数组 | `recovered-related` | inventory 固定来源、shape、dtype 和 SHA-256 | 只允许材料身份审计，不构成 canonical replay |

上述 `unavailable` 是 REG-01 的显式证据状态，不在本任务中提升为永久结论。
LEG-02 负责在继续恢复无果时形成永久 `unavailable` 裁决；REG-02 必须继承该状态，且
不得用最终 Pf 接近替代完整历史逐轮 trace 等价。

## 4. 本次复核与结论

正式环境为 Windows / Python 3.12。本任务定向执行行为、历史、身份和 publication
回归共 `15 passed`；完整 `tests/packaging tests/compatibility` suite 为 `78 passed`。
文档链接、`git diff --check`、UTF-8 无 BOM 和 v0.3.0 冻结证据零差异共同作为 PR
候选门。

REG-01 的交付结论是：已验证项、预期差异、相关恢复材料和不可验证项现在均有唯一
状态、证据位置与声明边界。下一动作是保持 LEG-02/REG-02 阻塞，等待新的历史资产或
明确恢复授权；不得自动启动 OPT-02/03。
