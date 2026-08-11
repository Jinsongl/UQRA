# LEG-02：Canonical FourBranch 历史资产永久 unavailable 裁决

## 1. 裁决

LEG-02 将下列 canonical FourBranch 历史证据正式裁决为永久 `unavailable`：

- 约 10⁵ 的原候选池及其逐阶排列；
- 约 10⁶ 的原 FourBranch 测试集；
- canonical 输入 bundle；
- 逐轮选点、QoI、Pf、停止位置和完整 trace；
- Python/NumPy RNG 状态、候选排列和 CV shuffle 状态。

“永久”表示：在当前可审计归档和既有来源链内，恢复工作已经结案；后续计划、测试和
声明不得继续把这些字段当作可等待获得的证据，也不得以重新采样、相关历史数组、汇总
结果或 reduced benchmark 补写。它不表示这些文件从未存在。将来若发现具有可验证来源、
身份和内容的新资产，必须通过新的稳定任务重新审计和裁决，不能静默覆盖本结论。

## 2. 裁决依据

| 证据 | 已完成核查 | 对本裁决的支持 |
| --- | --- | --- |
| [`阶段 9 恢复摘要`](ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md) | 搜索原配置根及 G:/F: 相关归档，只恢复相关 MCS 池和汇总数组 | 原候选池、测试集、逐轮结果和随机状态未恢复 |
| [`阶段 9 inventory`](ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json) | 固定 recovered/unavailable 项、来源、shape、dtype 与 SHA-256 | `recovered-related` 不能升级为 canonical 输入 |
| [`LEG-01 环境审计`](UQRA_LEG_01_LEGACY_ENVIRONMENT_AUDIT.md) | Windows/Python 3.8.20 与 3.9.23 均可安装、导入历史快照 | replay 阻塞不是环境不可运行，而是入口在逐轮执行前缺少 `Branches_McsE6R0.npy` |
| [`REG-01 结案矩阵`](UQRA_REG_01_BEHAVIOR_CLOSURE_MATRIX.md) | verified、expected-difference、recovered-related、unavailable 已逐项分离 | 现有现代行为证据不能推导缺失的历史逐轮身份 |

阶段 9 与 LEG-01 是独立的恢复/运行核查，却得到相同缺失集合；REG-01 又明确了可替代
与不可替代边界。继续保持无限期等待不会增加证据，只会让 M6 状态长期含糊，因此满足
主计划“形成经证据支持的阻塞或 unavailable 正式结论”的完成门。

## 3. 禁止替代与允许声明

禁止：

- 用重新采样的 candidate/test 数据冒充历史文件；
- 用正态 MCS 池或 `Branches_p10_*` 汇总数组倒推逐轮 ID、QoI、Pf 或停止位置；
- 用 FourBranch reduced、Ishigami reduced 或 Gayton reduced 声明 historical replay；
- 用最终 Pf 接近替代完整逐轮 trace 等价。

允许：

- 继续把现有 reduced cases 用作 `software_benchmark` / `reduced` 软件回归；
- 继续引用 Phase 7 verified 行为和 Phase 9 source-semantics diagnostic；
- 由论文仓库独立建立带新身份的 reconstructed baseline，但必须明确不是 canonical replay。

## 4. 下游状态

LEG-02 完成后，REG-02 应顺序启动并直接继承本裁决：完整历史逐轮 trace 等价的状态为
永久 `unavailable`，不是 `pass`、`fail` 或“以最终 Pf 判断的近似通过”。REG-02 只负责
封闭 U3/M6 的状态传播和允许声明，不重新搜索资产，也不重新运行 canonical 入口。

本裁决不改变公共 API、JSON bytes、hash scope、数学契约、现代逐轮行为、M4 触发门或
REL-04 恢复门；`OPT-02/03` 仍未批准启动。

## 5. 验收与重开条件

LEG-02 候选只包含证据与治理文档。验收要求为文档链接有效、主计划与看板同步、
`git diff --check`、UTF-8 无 BOM、v0.3.0 冻结证据零差异，以及 required checks 全绿。

重开 canonical replay 必须同时提供：可验证来源、文件内容身份、与历史入口期望角色的
匹配证据，以及足以判断候选/测试/逐轮/RNG-CV 状态完整性的清单。未满足这些条件的
新文件仍只能分类为 `recovered-related`。
