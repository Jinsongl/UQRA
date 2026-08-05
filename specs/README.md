# UQRA 源码修正规范

本目录是 `E:\HRL_XDev\UQRA` 源码修正工作的统一文档目录。

以下新建文档均应存放在本目录：

- 算法裁决与源码歧义记录；
- 开发计划与阶段目标；
- 审计摘要与进度总结；
- 兼容配置和行为契约；
- 验证协议、追踪数据结构和验收标准；
- 修改或淘汰旧代码所需的迁移说明。

生成的实验结果、缓存、运行日志以及大型候选池或测试数据不得放入本目录。

## 当前文档

- [UQRA_COMPATIBILITY_DECISIONS.md](UQRA_COMPATIBILITY_DECISIONS.md)：规定稳定候选样本身份、跨阶累积观测、LARS/OED/DoI 兼容行为、已知源码歧义、实施顺序和验证门槛。
- [SOURCE_CORRECTION_HANDOFF.md](SOURCE_CORRECTION_HANDOFF.md)：用于直接启动源码修改的实施交接上下文，汇总工作边界、缺陷编号、修正顺序、验证目标、禁止事项和待裁决问题。
- [UQRA_PROJECT_DEVELOPMENT_PLAN.md](UQRA_PROJECT_DEVELOPMENT_PLAN.md)：UQRA 软件项目主计划，规定 U0--U6、项目边界、runner 发布门和向论文项目交付的唯一接口。
- [ADAPTIVE_PCE_DEVELOPMENT_PLAN.md](ADAPTIVE_PCE_DEVELOPMENT_PLAN.md)：记录兼容核心初版完成后的实现审查、真实 Hermite 验证、逐轮行为回归、历史数据恢复、环境现代化、publication 配置资格验证和软件发布过程。
- [DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md](DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md)：指向独立论文仓库完整复现计划的交接文件；本仓库只保留 runner 接口和项目边界。
- [ADAPTIVE_PCE_PHASE5_AUDIT_SUMMARY.md](ADAPTIVE_PCE_PHASE5_AUDIT_SUMMARY.md)：记录阶段 5 状态机审查、六类终止测试矩阵、trace 证据、不变量逐项审计和阶段 6 真实 Hermite fixture 结果。
- [ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md](ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md)：记录冻结 Hermite 输入上的 legacy/现代预处理、LARS/CV、RRQR、D/S、DoI、累积观测和停止位置逐轮行为回归。
- [ADAPTIVE_PCE_PHASE8_BENCHMARK_SUMMARY.md](ADAPTIVE_PCE_PHASE8_BENCHMARK_SUMMARY.md)：记录确定性二维 benchmark、四类终止 manifest、样本身份验收、稳定哈希和可重现运行命令。
- [ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md](ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md)：记录历史归档搜索、恢复/缺失边界，以及 IDX-01/IDX-02 在真实历史坐标池上的 literal 诊断。
- [ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json](ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json)：记录恢复文件的来源、形状、dtype、SHA-256、重复副本和 unavailable 项。
- [ADAPTIVE_PCE_PHASE10_ENVIRONMENT_CI_SUMMARY.md](ADAPTIVE_PCE_PHASE10_ENVIRONMENT_CI_SUMMARY.md)：记录 pyDOE2 移除、依赖锁、干净 Python 3.11 验证、Python 3.12 CI 和合并门槛。
- [ADAPTIVE_PCE_PHASE11_PUBLICATION_RELEASE_SUMMARY.md](ADAPTIVE_PCE_PHASE11_PUBLICATION_RELEASE_SUMMARY.md)：记录冻结投稿协议、敏感性分析、三类实现边界、最终代码审查和合并准备。
- [ADAPTIVE_PCE_PHASE11_FROZEN_MANIFEST.json](ADAPTIVE_PCE_PHASE11_FROZEN_MANIFEST.json)：保存投稿协议、输入/源码哈希、现代与 portable 逐案例结果及独立 overfit 统计。
- [releases/UQRA_V0.1.0_EVIDENCE.md](releases/UQRA_V0.1.0_EVIDENCE.md)：汇总 `v0.1.0` 算法基线、M1 版本化交付接口、环境锁、schema、示例配置、全新克隆验收、允许声明和禁止声明。
- [releases/UQRA_V0.1.0_EVIDENCE.json](releases/UQRA_V0.1.0_EVIDENCE.json)：保存上述交付证据的机器可读摘要与关键哈希。

## 命名约定

长期维护的规范文档采用说明性大写英文文件名，正文统一使用中文：

```text
<主题>_DECISIONS.md
<主题>_DEVELOPMENT_PLAN.md
<主题>_AUDIT_SUMMARY.md
<主题>_VALIDATION_PROTOCOL.md
```

每份长期维护文档应注明：目的、证据或权威来源、实施状态、未决事项和验证要求。只要源码行为、博士论文意图和投稿实验协议存在差异，就必须分别说明，不得混写为同一个算法版本。
