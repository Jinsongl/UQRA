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
| 当前工作分支 | `codex/complete-m23-benchmarks` | 🔄 M2.3 多问题 benchmark 验收 |
| 最近 PR | [#6 Add M2.1 benchmark registry contract](https://github.com/Jinsongl/UQRA/pull/6) | ✅ 已合并；merge `cd51b66` |
| Python 3.11 | `47 passed` | ✅ [CI run 30978958356](https://github.com/Jinsongl/UQRA/actions/runs/30978958356) |
| Python 3.12 | `47 passed` | ✅ [CI run 30978958356](https://github.com/Jinsongl/UQRA/actions/runs/30978958356) |
| Required check | `Adaptive compatibility gate` | ✅ 通过 |
| 全新克隆验收 | Python 3.11.15，`42 passed`，smoke/full manifest 有效 | ✅ [`UQRA_V0.1.0_EVIDENCE.md`](releases/UQRA_V0.1.0_EVIDENCE.md) |
| 当前 Release | [`v0.2.0`](https://github.com/Jinsongl/UQRA/releases/tag/v0.2.0) | ✅ 指向合并提交 `3445464d` |
| 下一版本 | 待 M2.1--M2.3 完成范围确定 | ⏳ 不提前承诺版本号 |

## 3. 编号体系与里程碑路线

- `U0--U6`：长期工作流，用于表达项目责任边界和持续状态；
- `M0--M3`：交付里程碑，用于表达可验收的阶段成果；
- `BENCH/LEG/REG/PKG/CI/SCHEMA-*`：具体任务 ID，用于分支、PR、CI 和证据追踪。

| 里程碑 | 状态 | 所属工作流 | 任务映射 | 完成门摘要 |
| --- | --- | --- | --- | --- |
| M0 治理与边界 | ✅ 已完成 | U0 | 计划与边界文档 | 进入版本控制并明确项目/论文边界 |
| M1 Runner 契约与可审计发布（原 UQRA-MV1） | ✅ 已完成 | U5、U6 | schema、CLI、示例、evidence、clean-clone、REL-01/02 | `v0.2.0`、双版本 CI 和 required gate 通过 |
| M2.1 Benchmark 注册与配置契约 | ✅ 已完成 | U4、U6 | BENCH-01 | 静态 registry、config v2、双版本 45 项测试和 required gate 通过 |
| M2.2 首个多路径缩减基准 | ✅ 已完成 | U4 | BENCH-02 | FourBranch reduced 的输入身份、DoI、trace、重复性和 PR #6 required gate 通过 |
| M2.3 多问题 Benchmark 验收 | 🔄 进行中 | U4 | BENCH-03、BENCH-04、BENCH-06 | 三基准本地统一验收完成；Python 3.11/3.12 均 `52 passed`；待远端 CI/审查 |
| M3 包装与跨环境质量 | ⏳ 待开始 | U6 | PKG-01--04、CI-01、SCHEMA-01 | 包装、版本源、schema 验证和跨平台 CI 达到发布门 |

`LEG-01/02` 与 `REG-01/02` 是 U1/U3 的证据闭环任务，不属于 M2，也不因 M2 完成
而自动关闭。

## 4. U0--U6 状态总览

| 阶段 | 状态 | 当前结论 | 下一动作 |
| --- | --- | --- | --- |
| U0 治理冻结与证据清单 | ✅ 已完成 | 项目/论文边界、主计划、历史计划和交接接口已固化 | 随新裁决持续维护证据清单 |
| U1 Legacy 基准恢复 | ⛔ 阻塞 | 可恢复证据有限；历史候选池、测试集和 RNG/CV 状态缺失 | 执行 Legacy 环境审计并形成正式阻塞结论 |
| U2 Modern 核心实现 | ✅ 已完成 | 核心算法完成，当前发布基线为 `v0.2.0` | 核心变更继续遵守裁决和回归门 |
| U3 内核与逐轮行为回归 | ✅ 已完成（可获得证据范围） | Phase 5--8 证据完整；历史 FourBranch trace 不可用 | 建立统一结案矩阵，与 U1 阻塞结论互链 |
| U4 通用软件 benchmark | 🔄 进行中 | M2.1 registry/config v2 与 M2.2 FourBranch reduced 完成 | 决定并启动 M2.3 的下一个 reduced benchmark |
| U5 Runner 发布门 | ✅ 已完成 | M1 schema、CLI、示例、evidence、PR #4 和 `v0.2.0` 发布均完成 | 按版本化流程维护后续交付 |
| U6 版本化维护 | 🔄 进行中 | required CI 和版本化变更流程已建立 | 持续处理 benchmark、包装和依赖质量任务 |

## 5. 当前焦点

### ✅ 最近完成

| ID | 归属 / 优先级 | 任务 | 验收证据 | 下一动作 |
| --- | --- | --- | --- | --- |
| REL-01 | M0/M1 / P0 | 合并 M0/M1 交付 | PR #4；Python 3.11/3.12 和聚合 gate 全部通过 | ✅ 已合并到 `master` |
| REL-02 | M1 / P0 | 发布 `v0.2.0` | tag、GitHub Release、required gate | ✅ 已发布 |
| BENCH-01 | M2.1 / P1 | 受控 benchmark registry/config v2 | PR #6；Python 3.11/3.12 均 `45 passed`；required gate 通过 | ✅ 完成，作为 M2.2 注册入口 |
| BENCH-02 | M2.2 / P1 | FourBranch reduced benchmark | 三套独立冻结输入、显式失效定义、DoI/manifest/trace 重复性；PR #6 双版本 `47 passed` | ✅ 完成；不声明为历史 replay 或论文生产结果 |

### 🔄 进行中

| ID | 归属 / 优先级 | 任务 | 当前证据 | 完成门 |
| --- | --- | --- | --- | --- |
| BENCH-03 | M2.3 / P1 | Ishigami reduced | 冻结输入、非线性/交互、manifest/trace 固定 hash 和本地双版本测试通过 | 远端 CI/审查 |
| BENCH-04 | M2.3 / P1 | Gayton reduced | 冻结输入、局部失效域/DoI、manifest/trace 固定 hash 和本地双版本测试通过 | 远端 CI/审查 |
| BENCH-06 | M2.3 / P1 | 三基准统一验收 | registry/config/identity/禁止声明统一契约；本地双版本 `52 passed` | 远端 CI/审查 |

### ⏳ 待开始

| ID | 归属 / 优先级 | 任务 | 前置条件 | 完成门 |
| --- | --- | --- | --- | --- |
| LEG-01 | U1 / P1 | Legacy Python 3.8/3.9 环境审计 | 无 | 依赖、可运行入口和失败类型形成可审计报告 |
| REG-01 | U3 / P1 | U3 行为回归结案矩阵 | LEG-01 可并行 | 已验证项与 `unavailable` 项逐项对应证据，无状态歧义 |

## 6. 后续队列

### ⏳ 待开始：U4 benchmark 扩展

| ID | 归属 / 优先级 | 任务 | 主要保护行为 | 依赖 |
| --- | --- | --- | --- | --- |
| BENCH-05 | M2.3 / P1 | Damped oscillator reduced | 动态模型接口和高成本模型替身 | BENCH-01 |

每个 benchmark 必须记录：

- `purpose: software_benchmark`；
- `scale: reduced`；
- benchmark 名称和版本；
- RNG、seed、candidate/test/reference 的规模与 SHA-256；
- 预算、预期状态、`stop_reason` 和 trace hash/容差；
- 明确禁止 `paper_production` 声明。

### ⏳ 待开始：包装与跨环境质量

| ID | 归属 / 优先级 | 任务 | 完成门 |
| --- | --- | --- | --- |
| PKG-01 | M3 / P2 | 构建并测试 sdist/wheel | 从仓库外安装、导入和运行 CLI 成功 |
| PKG-02 | M3 / P2 | 将主要元数据迁移到 `pyproject.toml` | 不再依赖旧 `setup.py upload` 发布逻辑 |
| PKG-03 | M3 / P2 | 建立 `uqra.__version__` 唯一版本源 | 包、CLI、manifest 版本一致 |
| PKG-04 | M3 / P2 | 清理 Python 3.12 转义警告 | 兼容性测试无对应 SyntaxWarning/DeprecationWarning |
| CI-01 | M3 / P2 | 扩展 Windows/Linux CI | Python 3.11/3.12 在支持平台通过 |
| SCHEMA-01 | M3 / P2 | 增加标准 JSON Schema 验证器测试 | 示例和生成 manifest 同时通过运行时与标准 schema 校验 |

### ⛔ 阻塞：历史资产

| ID | 阻塞项 | 已有证据 | 解除或关闭条件 |
| --- | --- | --- | --- |
| LEG-02 | 完整历史 FourBranch replay | Phase 9 inventory 与 IDX-01/IDX-02 诊断 | 找回原候选池、测试集、逐轮输出和 RNG/CV 状态，或正式结案为永久 `unavailable` |
| REG-02 | 完整历史逐轮 trace 等价 | U3 可获得证据范围已完成 | 依赖 LEG-02；不得以最终 Pf 接近替代 |

## 7. 已完成里程碑

| 里程碑 | 状态 | 主要证据 |
| --- | --- | --- |
| Phase 5--11 自适应实现与资格验证 | ✅ 已完成 | `ADAPTIVE_PCE_PHASE*_SUMMARY`、冻结 manifest |
| `v0.1.0` 软件基线 | ✅ 已完成 | PR #3、tag `v0.1.0`、required gate |
| M0 项目治理与边界 | ✅ 已完成 | commit `9ee5ed6` |
| M1 Runner 契约与可审计发布（原 UQRA-MV1） | ✅ 已完成 | commits `c6fea07`、`c69032d`；42 项兼容性测试；U5 交付门和 `specs/releases/` |
| `v0.2.0` runner contracts release | ✅ 已完成 | PR #4、merge `3445464d`、GitHub Release |

## 8. 更新流程

每次开始或完成任务时按以下顺序更新：

1. 为任务分配稳定 ID，并标明所属 `M*` 里程碑或 `U*` 持续工作流；不复用已关闭 ID；
2. 从 `⏳ 待开始` 移到 `🔄 进行中` 时，填写分支或第一项产物；
3. 远端审查仍使用 `🔄 进行中`，并填写 PR 和 CI 链接；
4. 只有验收门全部满足后才能标记 `✅ 已完成`；
5. `⛔ 阻塞` 必须写明外部依赖、已有证据和关闭条件；
6. 更新顶部状态日期、U0--U6 总览和当前焦点；
7. 若任务改变项目范围或算法契约，先更新主计划或兼容性裁决，再更新看板；
8. 同步更新相关 evidence、summary 和 release 文档，不在看板复制大段实验结果。

## 9. 不进入本看板的工作

- 正式规模 FourBranch 重复实验；
- PEM/Structural Safety 论文修改；
- 论文表格、图件和统计结论；
- WEC-Sim 工程耦合生产实验；
- 博士论文科学结论的 `pass/partial/fail` 判定。

这些任务由独立论文仓库管理；UQRA 看板只追踪版本化 runner、软件 benchmark、
兼容性证据、发布门和维护质量。
