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
- [ADAPTIVE_PCE_DEVELOPMENT_PLAN.md](ADAPTIVE_PCE_DEVELOPMENT_PLAN.md)：规定兼容核心初版完成后的实现审查、真实 Hermite 验证、逐轮行为回归、历史数据恢复、环境现代化和投稿发布顺序。
- [ADAPTIVE_PCE_PHASE5_AUDIT_SUMMARY.md](ADAPTIVE_PCE_PHASE5_AUDIT_SUMMARY.md)：记录阶段 5 状态机审查、六类终止测试矩阵、trace 证据、不变量逐项审计和阶段 6 真实 Hermite fixture 结果。
- [ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md](ADAPTIVE_PCE_PHASE7_REGRESSION_SUMMARY.md)：记录冻结 Hermite 输入上的 legacy/现代预处理、LARS/CV、RRQR、D/S、DoI、累积观测和停止位置逐轮行为回归。
- [ADAPTIVE_PCE_PHASE8_BENCHMARK_SUMMARY.md](ADAPTIVE_PCE_PHASE8_BENCHMARK_SUMMARY.md)：记录确定性二维 benchmark、四类终止 manifest、样本身份验收、稳定哈希和可重现运行命令。

## 命名约定

长期维护的规范文档采用说明性大写英文文件名，正文统一使用中文：

```text
<主题>_DECISIONS.md
<主题>_DEVELOPMENT_PLAN.md
<主题>_AUDIT_SUMMARY.md
<主题>_VALIDATION_PROTOCOL.md
```

每份长期维护文档应注明：目的、证据或权威来源、实施状态、未决事项和验证要求。只要源码行为、博士论文意图和投稿实验协议存在差异，就必须分别说明，不得混写为同一个算法版本。
