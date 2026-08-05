# UQRA 项目进度看板

状态日期：2026-08-05  
维护范围：UQRA 软件项目  
主计划：[`UQRA_PROJECT_DEVELOPMENT_PLAN.md`](UQRA_PROJECT_DEVELOPMENT_PLAN.md)  
历史实施记录：[`ADAPTIVE_PCE_DEVELOPMENT_PLAN.md`](ADAPTIVE_PCE_DEVELOPMENT_PLAN.md)

## 1. 看板用途

本文件用于追踪 UQRA 软件项目的当前任务、验收证据、阻塞条件和下一动作。项目范围、
阶段定义和完成门槛以主计划为准；本看板不重新定义算法，也不管理论文正式规模实验。

### 1.1 状态标记

| 标记 | 状态 | 含义 |
|---|---|---|
| ✅ | 已完成 | 交付物已形成，并达到本任务的验收标准 |
| 🔄 | 进行中 | 当前正在处理，或下一步已经明确 |
| ⏳ | 待开始 | 前置条件已满足，但尚未开展 |
| ⛔ | 阻塞 | 缺少关键材料、用户决策或外部条件 |
| ➖ | 不适用/取消 | 经决策不再纳入当前论文 |

进度只按验收证据更新，不使用主观百分比。`paper_production`、论文表图和科学结论判定
由独立论文仓库管理，不进入本看板。

## 2. 当前发布与质量门

| 项目 | 当前值 | 状态/证据 |
| --- | --- | --- |
| 默认分支 | `master` | ✅ `v0.2.0` 已发布 |
| 当前工作分支 | `codex/update-progress-board` | 🔄 更新项目看板 |
| 最近 PR | [#4 Complete M0/M1 runner governance and delivery contracts](https://github.com/Jinsongl/UQRA/pull/4) | ✅ 已合并 |
| Python 3.11 | `42 passed` | ✅ [CI run 30972668606](https://github.com/Jinsongl/UQRA/actions/runs/30972668606) |
| Python 3.12 | `42 passed` | ✅ [CI run 30972668606](https://github.com/Jinsongl/UQRA/actions/runs/30972668606) |
| Required check | `Adaptive compatibility gate` | ✅ 通过 |
| 全新克隆验收 | Python 3.11.15，`42 passed`，smoke/full manifest 有效 | ✅ [`UQRA_V0.1.0_EVIDENCE.md`](releases/UQRA_V0.1.0_EVIDENCE.md) |
| 当前 Release | [`v0.2.0`](https://github.com/Jinsongl/UQRA/releases/tag/v0.2.0) | ✅ 指向合并提交 `3445464d` |
| 下一版本 | 待 M2 范围确定 | ⏳ 不提前承诺版本号 |

## 3. U0--U6 状态总览

| 阶段 | 状态 | 当前结论 | 下一动作 |
| --- | --- | --- | --- |
| U0 治理冻结与证据清单 | ✅ 已完成 | 项目/论文边界、主计划、历史计划和交接接口已固化 | 随新裁决持续维护证据清单 |
| U1 Legacy 基准恢复 | ⛔ 阻塞 | 可恢复证据有限；历史候选池、测试集和 RNG/CV 状态缺失 | 执行 Legacy 环境审计并形成正式阻塞结论 |
| U2 Modern 核心实现 | ✅ 已完成 | 核心算法完成，当前发布基线为 `v0.2.0` | 核心变更继续遵守裁决和回归门 |
| U3 内核与逐轮行为回归 | ✅ 已完成（可获得证据范围） | Phase 5--8 证据完整；历史 FourBranch trace 不可用 | 建立统一结案矩阵，与 U1 阻塞结论互链 |
| U4 通用软件 benchmark | 🔄 进行中 | 二维 Hermite 缩减 benchmark 完成；多问题扩展是当前主线 | 启动受控 benchmark registry/config v2 |
| U5 Runner 发布门 | ✅ 已完成 | M1 schema、CLI、示例、evidence、PR #4 和 `v0.2.0` 发布均完成 | 按版本化流程维护后续交付 |
| U6 版本化维护 | 🔄 进行中 | required CI 和版本化变更流程已建立 | 持续处理 benchmark、包装和依赖质量任务 |

## 4. 当前焦点

### ✅ 最近完成

| ID | 优先级 | 任务 | 验收证据 | 下一动作 |
| --- | --- | --- | --- | --- |
| REL-01 | P0 | 合并 M0/M1 交付 | PR #4；Python 3.11/3.12 和聚合 gate 全部通过 | ✅ 已合并到 `master` |
| REL-02 | P0 | 发布 `v0.2.0` | tag、GitHub Release、required gate | ✅ 已发布 |

### 🔄 进行中

| ID | 优先级 | 任务 | 前置条件 | 完成门 |
| --- | --- | --- | --- | --- |
| BENCH-01 | P1 | 设计受控 benchmark registry/config v2 | `v0.2.0` 已发布 | 禁止任意 Python 导入；配置能选择已注册 benchmark 并通过 schema 校验 |

### ⏳ 待开始

| ID | 优先级 | 任务 | 前置条件 | 完成门 |
| --- | --- | --- | --- | --- |
| BENCH-02 | P1 | FourBranch reduced benchmark | BENCH-01 | 固定输入/seed/hash、DoI 路径、manifest、trace 和重复性测试通过 |
| LEG-01 | P1 | Legacy Python 3.8/3.9 环境审计 | 无 | 依赖、可运行入口和失败类型形成可审计报告 |
| REG-01 | P1 | U3 行为回归结案矩阵 | LEG-01 可并行 | 已验证项与 `unavailable` 项逐项对应证据，无状态歧义 |

## 5. 后续队列

### ⏳ 待开始：U4 benchmark 扩展

| ID | 优先级 | 任务 | 主要保护行为 | 依赖 |
| --- | --- | --- | --- | --- |
| BENCH-03 | P1 | Ishigami reduced | 非线性和交互项 | BENCH-01 |
| BENCH-04 | P1 | Gayton reduced | 可靠性函数和局部失效域 | BENCH-01 |
| BENCH-05 | P1 | Damped oscillator reduced | 动态模型接口和高成本模型替身 | BENCH-01 |
| BENCH-06 | P1 | U4 多问题验收 | 至少三个不同性质 benchmark、schema、CI 和声明边界 | BENCH-02--05 中至少三个完成 |

每个 benchmark 必须记录：

- `purpose: software_benchmark`；
- `scale: reduced`；
- benchmark 名称和版本；
- RNG、seed、candidate/test/reference 的规模与 SHA-256；
- 预算、预期状态、`stop_reason` 和 trace hash/容差；
- 明确禁止 `paper_production` 声明。

### ⏳ 待开始：包装与跨环境质量

| ID | 优先级 | 任务 | 完成门 |
| --- | --- | --- | --- |
| PKG-01 | P2 | 构建并测试 sdist/wheel | 从仓库外安装、导入和运行 CLI 成功 |
| PKG-02 | P2 | 将主要元数据迁移到 `pyproject.toml` | 不再依赖旧 `setup.py upload` 发布逻辑 |
| PKG-03 | P2 | 建立 `uqra.__version__` 唯一版本源 | 包、CLI、manifest 版本一致 |
| PKG-04 | P2 | 清理 Python 3.12 转义警告 | 兼容性测试无对应 SyntaxWarning/DeprecationWarning |
| CI-01 | P2 | 扩展 Windows/Linux CI | Python 3.11/3.12 在支持平台通过 |
| SCHEMA-01 | P2 | 增加标准 JSON Schema 验证器测试 | 示例和生成 manifest 同时通过运行时与标准 schema 校验 |

### ⛔ 阻塞：历史资产

| ID | 阻塞项 | 已有证据 | 解除或关闭条件 |
| --- | --- | --- | --- |
| LEG-02 | 完整历史 FourBranch replay | Phase 9 inventory 与 IDX-01/IDX-02 诊断 | 找回原候选池、测试集、逐轮输出和 RNG/CV 状态，或正式结案为永久 `unavailable` |
| REG-02 | 完整历史逐轮 trace 等价 | U3 可获得证据范围已完成 | 依赖 LEG-02；不得以最终 Pf 接近替代 |

## 6. 已完成里程碑

| 里程碑 | 状态 | 主要证据 |
| --- | --- | --- |
| Phase 5--11 自适应实现与资格验证 | ✅ 已完成 | `ADAPTIVE_PCE_PHASE*_SUMMARY`、冻结 manifest |
| `v0.1.0` 软件基线 | ✅ 已完成 | PR #3、tag `v0.1.0`、required gate |
| M0 项目治理与边界 | ✅ 已完成 | commit `9ee5ed6` |
| M1 schema、CLI 与示例 | ✅ 已完成 | commit `c6fea07`；42 项兼容性测试 |
| M1 evidence 与 clean-clone 验收 | ✅ 已完成 | commit `c69032d`；`specs/releases/` |
| UQRA-MV1 | ✅ 已完成 | U5 交付门和 evidence package |
| `v0.2.0` runner contracts release | ✅ 已完成 | PR #4、merge `3445464d`、GitHub Release |

## 7. 更新流程

每次开始或完成任务时按以下顺序更新：

1. 为任务分配稳定 ID，不复用已关闭 ID；
2. 从 `⏳ 待开始` 移到 `🔄 进行中` 时，填写分支或第一项产物；
3. 远端审查仍使用 `🔄 进行中`，并填写 PR 和 CI 链接；
4. 只有验收门全部满足后才能标记 `✅ 已完成`；
5. `⛔ 阻塞` 必须写明外部依赖、已有证据和关闭条件；
6. 更新顶部状态日期、U0--U6 总览和当前焦点；
7. 若任务改变项目范围或算法契约，先更新主计划或兼容性裁决，再更新看板；
8. 同步更新相关 evidence、summary 和 release 文档，不在看板复制大段实验结果。

## 8. 不进入本看板的工作

- 正式规模 FourBranch 重复实验；
- PEM/Structural Safety 论文修改；
- 论文表格、图件和统计结论；
- WEC-Sim 工程耦合生产实验；
- 博士论文科学结论的 `pass/partial/fail` 判定。

这些任务由独立论文仓库管理；UQRA 看板只追踪版本化 runner、软件 benchmark、
兼容性证据、发布门和维护质量。
