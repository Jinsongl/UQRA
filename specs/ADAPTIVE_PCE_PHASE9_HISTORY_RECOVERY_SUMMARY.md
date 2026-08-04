# 自适应 PCE 阶段 9：历史数据恢复与四分支追踪摘要

## 1. 结论

阶段 9 的恢复审计与可执行诊断已完成，但 canonical 四分支运行不能被宣称为已复现。源码配置指向的 `G:\My Drive\MUSE_UQ_DATA` 已不存在；在 `G:\03_Archive\UT_Work`、`G:\CloudSync`、`G:\05_Backup` 和 `F:\05_WorkingPapers` 中仅恢复到相关 MCS 池及四分支汇总数组，未恢复专用候选池、测试集、逐轮结果和随机状态。

机器可读身份、全部来源副本和 unavailable 项见 `ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json`。所有历史文件保持只读，未复制进仓库。

## 2. 恢复结果

| 内容 | 状态 | 形状 / dtype | 身份说明 |
|---|---|---|---|
| 历史正态 MCS 池 | recovered-related | `(10, 1000000)` / `float64` | SHA-256 `1291aa...c486`；可用于源码语义诊断，但不能证明是四分支 canonical 输入 |
| 历史均匀 MCS 池 | recovered-related | `(10, 1000000)` / `float64` | SHA-256 `b0ef49...4c81`；分布与 Hermite 四分支不符 |
| `Branches_p10_all.npy` | recovered-summary | `(3, 1)` / `float64` | 七份逐字节相同副本，SHA-256 `643688...fac6` |
| `Branches_p10_D.npy` | recovered-summary | `(3, 13)` / `float64` | 七份逐字节相同副本，SHA-256 `747e4f...402a` |
| `Branches_p10_mcs.npy` | recovered-summary | `(3, 13)` / `float64` | 七份逐字节相同副本，SHA-256 `06a8bd...94ac` |
| canonical 约 10⁵ 候选池 | unavailable | — | 原配置目录缺失，不能用重新采样替代 |
| canonical 约 10⁶ 四分支测试集 | unavailable | — | 未找到匹配 `McsE6R0` 的四分支文件 |
| 逐轮选点、QoI、Pf、停止位置 | unavailable | — | 汇总数组不包含逐轮身份 |
| 历史 RNG/CV 状态 | unavailable | — | 无法恢复原 `random.sample` 排列或 CV shuffle |

## 3. literal 索引影响量化

诊断使用已恢复正态 MCS 池的前两维，固定种子 `20260804`，每阶从 1,000,000 个历史坐标中抽取 100,000 个候选。保留 66 个诊断 ID，并在近原点的 256 个历史坐标中选取 32 个 DoI ID。该选择用于重放源码的字面索引语义，不是对缺失历史随机状态或选点路径的推断。

### IDX-01：跨阶局部索引复用

- 66/66 个保留局部 ID 在下一阶指向不同全局坐标；
- literal 排除错误地移除 66 个未观测坐标；
- 5 个上一阶真实已观测坐标再次出现在下一阶候选池，但均未被正确排除；
- 现代稳定-ID实现保持原 66 个全局身份，坐标身份错配为 0。

因此，IDX-01 在该历史坐标池上会从升阶后的第一轮就改变候选集合、Vandermonde 行、D/S 分数和后续选中 ID；最终 Pf 即使接近，也不能消除这条逐轮分叉。

### IDX-02：DoI 局部索引直接追加

- 32/32 个 DoI 局部 ID 与其应映射的全局候选 ID 不同；
- literal 行为追加 `0..31`，稳定-ID行为追加实际 DoI 全局 ID；
- 现代实现通过显式 `local_to_global` 映射将错配降为 0。

因此，IDX-02 会在首次 DoI 增广处改变模型调用坐标、累积观测和停止判定。诊断哈希为 `1e275fe9d6d9febc90717f9e9e4188b2067252db90abb0a83dbd17e6ede8595e`。

## 4. 逐轮解释边界

可确定的逐轮分叉顺序是：跨阶重新抽样 → IDX-01 身份错配 → 候选排除与评分输入变化 → 全局增广 ID 变化 → DoI 构造 → IDX-02 追加坐标变化 → 累积观测、QoI 与停止位置变化。现代实现以冻结候选池和全局稳定 ID 阻断前两处身份破坏。

不能确定的是原博士论文运行每轮具体选中了哪些 ID、哪一轮触发 DoI、在哪一轮停止以及最终 Pf；这些字段均明确为 unavailable。现有 `Branches_p10_*` 只允许验证汇总文件身份，不允许倒推出逐轮轨迹，也不允许以最终指标接近宣称算法等价。

## 5. 验证

新增 `archived_array_identity` 只读记录文件 SHA-256、形状和 dtype；`historical_literal_index_diagnostic` 固化可复现的 IDX-01/IDX-02 量化，并明确输出类型为 `source_semantics_diagnostic_not_historical_replay`。相应测试覆盖身份记录、确定性和两类缺陷计数。

阶段 9 的内部工作已结束；任何“canonical 历史复现”结论仍以恢复 inventory 中标为 unavailable 的原文件为前置条件。下一执行阶段为阶段 10：依赖和持续集成现代化。
