# 博士论文数值复现计划：外部项目交接

## 1. 计划归属

博士论文正式规模的数值复现、统计聚合、表图重算和科学结论判定不属于
UQRA 软件项目。完整计划已经迁移到独立论文仓库：

- 仓库：[Jinsongl/adaptive-sparse-pce-rare-event](https://github.com/Jinsongl/adaptive-sparse-pce-rare-event)
- 仓库内文件：`DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md`
- 本地工作区：`F:\05_WorkingPapers\2026_A unified adaptive framework combining sparse PCE and sequential optimal design for efficient rare-event estimation\DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md`

迁移时已核对论文仓库副本与原 UQRA 计划全文的 SHA-256，二者一致：
`ad7317be40fa1f2b0bdf23b659e8ffde3e07f30567444253bbe808c73be87e0d`。

本文件不再管理论文复现阶段 R0--R7，只保留 UQRA 与论文项目之间的接口约定。

## 2. UQRA 向论文项目提供的接口

每次可用于论文正式计算的 UQRA 交付至少包含：

1. 固定的 Git commit 或 release tag，以及干净工作树声明；
2. Python 和依赖环境锁、安装命令及支持版本；
3. runner 配置、输入/结果 schema 和可执行入口；
4. 兼容性测试、逐轮行为回归和 required CI 结论；
5. candidate、test、QoI 和 reference 数据的身份规则；
6. manifest、输入、trace、源码树和结果摘要哈希；
7. 已知限制、缺失历史资产、允许声明和禁止声明；
8. 缺陷反馈所需的最小复现、配置、commit 和 trace 格式。

当前首个发布基线是 `v0.1.0`。它证明 modern UQRA-compatible 核心、缩减规模
确定性回归协议和 Python 3.11/3.12 环境可重复；它不证明博士论文历史
FourBranch 已重放，也不代表正式论文规模实验已经完成。

## 3. 项目边界

- 核心算法只在 UQRA 项目中维护；论文仓库不得复制或另行修改
  `uqra/adaptive` 后形成第二套正式实现。
- 论文项目只消费通过发布门的版本化 runner。发现算法或软件缺陷时，应将最小
  复现返回 UQRA；UQRA 发布修复版本后，论文项目升级并重跑受影响实验。
- UQRA 的缩减规模 benchmark 必须标记为 `purpose: software_benchmark` 和
  `scale: reduced`；论文正式运行必须标记为 `purpose: paper_production`。
- canonical UQRA 只用于只读溯源、历史证据和行为对照；新生成的数据不得标记为
  canonical 历史结果。
- simplified portable runner 只用于已冻结的历史对照，不得生成新的正式论文结果。

UQRA 软件路线由 `UQRA_PROJECT_DEVELOPMENT_PLAN.md` 管理；论文复现的阶段、
claim matrix、正式协议、实验执行和结论判定以独立论文仓库中的计划为准。
