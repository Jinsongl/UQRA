# REG-02：Canonical FourBranch 历史逐轮 trace 等价状态结案

## 1. 结论

REG-02 对 canonical FourBranch 完整历史逐轮 trace 等价作出唯一结论：永久
`unavailable`。

该状态直接继承
[`LEG-02 永久 unavailable 裁决`](UQRA_LEG_02_PERMANENT_UNAVAILABLE_DECISION.md)：
比较所需的 canonical 候选池、测试集、逐轮结果和 RNG/CV 状态均不可获得，因此没有
合法的 reference trace 可用于判定现代实现相对历史运行是 `pass` 或 `fail`。最终 Pf、
汇总数组或 reduced benchmark 的接近程度不能改变这一状态。

## 2. 逐轮等价字段状态

| 比较字段 | 历史 reference 状态 | 可用现代/诊断证据 | REG-02 结论 |
| --- | --- | --- | --- |
| candidate/test 坐标及身份 | `unavailable` | Phase 7 冻结 Hermite fixture；三个 reduced benchmark identity | 不能映射到 canonical 历史输入 |
| 每阶候选排列与保留 ID | `unavailable` | Phase 9 IDX-01 source-semantics diagnostic | 可证明 legacy 索引风险，不能恢复原排列 |
| LARS 进入路径、fold 与 CV 截断 | `unavailable` | Phase 7 共享 fixture 上为 `verified` | 不能推广为原 FourBranch 运行路径 |
| RRQR 初始点、D/S 分数与逐轮选点 | `unavailable` | Phase 7 全候选三轮比较为 `verified` | 缺少历史输入及逐轮 ID，不能判定 canonical 等价 |
| DoI 构造和 local/global 映射 | `unavailable` | Phase 7 为 `verified`；Phase 9 IDX-02 为 `expected-difference` | 不能恢复历史首次 DoI 轮次或所选坐标 |
| 累积观测、模型调用、阶次和停止位置 | `unavailable` | literal reduced/fixture 行为为 `verified` | 缺少历史 trace，不能比较逐轮累计状态 |
| 每轮 QoI、Pf 和误差指标 | `unavailable` | `Branches_p10_*` 仅为 recovered summary | 汇总数组不得倒推出逐轮值 |
| RNG、candidate permutation、CV shuffle | `unavailable` | 无 canonical 状态 | 不能重放随机路径 |

字段口径与现有数值容差见
[`REG-01 行为回归结案矩阵`](UQRA_REG_01_BEHAVIOR_CLOSURE_MATRIX.md)。REG-01 的
`verified` 只证明冻结共享 fixture 或 live reduced benchmark 上的现代/兼容行为；它不
提供缺失的 canonical historical identity。

## 3. 状态传播规则

REG-02 使用以下确定性判定：

1. 完整逐轮等价要求输入身份、随机状态和逐轮 reference 同时存在；
2. LEG-02 已将这些前置证据永久标记为 `unavailable`；
3. 因此完整历史逐轮 trace 等价只能为永久 `unavailable`；
4. 不得把该状态改写为算法 `fail`，因为不存在可执行的完整比较；
5. 不得把该状态改写为 `pass`、`partial pass` 或“最终 Pf 接近”，因为逐轮身份与状态
   没有 reference。

这不会否定 Phase 7 的行为等价证据，也不会改变 IDX-01/IDX-02 已裁决为 legacy 缺陷
修正的 `expected-difference`。两类结论作用域不同：前者属于冻结可执行 fixture，后者
属于缺失身份下的 canonical 历史运行。

## 4. 允许和禁止声明

允许声明：

- UQRA-compatible 内核在冻结 fixture 上的预处理、LARS/CV、RRQR、D/S、DoI、累积
  观测和 literal 停止行为已经验证；
- 现有 benchmark 是 `software_benchmark` / `reduced`；
- Phase 9 诊断是 `source_semantics_diagnostic_not_historical_replay`；
- canonical FourBranch 完整历史逐轮 trace 等价永久 `unavailable`。

禁止声明：

- 已复现博士论文 FourBranch canonical 逐轮运行；
- recovered-related MCS 池或汇总数组就是 canonical 输入/trace；
- reduced benchmark 或 reconstructed baseline 是 historical replay；
- 最终 Pf 接近足以证明完整逐轮等价。

## 5. M6 结案与重开条件

REG-02 完成后，M6 在现有来源范围内结案：LEG-01/REG-01 提供可获得证据，LEG-02
永久关闭缺失资产状态，REG-02 将该状态传播到完整历史逐轮 trace 等价。M6 结案不恢复
论文正式算例，不改变公共 API、数学契约、现代逐轮行为、M4 触发门或 REL-04 恢复门。

只有新资产同时满足 LEG-02 规定的来源、内容身份、历史角色匹配和完整性条件时，才能
通过新的稳定任务重新打开。任何不满足该门的新材料仍归类为 `recovered-related`。

## 6. 验收

REG-02 是证据治理任务，不修改生产代码。候选验收包括：

- REG-01 与 LEG-02 状态逐项一致；
- 文档链接有效；
- Windows/Python 3.12 完整 packaging + compatibility suite 通过；
- `git diff --check`、UTF-8 无 BOM；
- `specs/releases/UQRA_V0.3.0_*` 零差异；
- 任务 PR 与合并后的 master required checks 全绿，随后完成纯文档 closure PR。

本地正式环境 Windows/Python 3.12 的完整 `tests/packaging tests/compatibility` 结果为
`78 passed`。
