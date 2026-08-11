# UQRA 项目进度看板

状态日期：2026-08-10
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
| 默认分支 | `master` | ✅ `v0.3.0` 已发布 |
| 当前基线 / 文档分支 | `master` `693b1d2c3ba8da2608966d1840f36ee20d65a94a` / `codex/local-first-development-policy` | 🔄 回填治理 closure 并确立“本地开发优先、候选就绪后才触发 CI”的执行策略 |
| 最近合并 PR | [#21 Enforce mandatory task closure governance](https://github.com/Jinsongl/UQRA/pull/21) | ✅ 已合并；merge `693b1d2c3ba8da2608966d1840f36ee20d65a94a`；master compatibility run [31346375232](https://github.com/Jinsongl/UQRA/actions/runs/31346375232) 与 governance run [31346375225](https://github.com/Jinsongl/UQRA/actions/runs/31346375225) 全绿 |
| 暂缓 PR | [#15 Implement REL-04 controlled release automation](https://github.com/Jinsongl/UQRA/pull/15) | ⏳ 保持 Draft；代码与 required run `31152911246` 已通过，等待真实新版本恢复端到端验证 |
| Python 3.11 | 允许安装但未持续验证 | ➖ 不作为 M3 完成门；如取消安装支持，另行同步元数据、README、锁和发布说明 |
| 正式验证环境 | Windows + Python 3.12 | ✅ 唯一正式、持续验证环境；日常开发优先使用本地定向与完整测试，远端 CI 只在候选 PR/closure 阶段运行 |
| Required checks | `Adaptive compatibility gate`；`Task closure governance` | ✅ 两项已纳入 `master` branch protection；PR #21 合并后 master runs `31346375232`、`31346375225` 全绿 |
| 全新克隆验收 | Python 3.11.15，`42 passed`，smoke/full manifest 有效 | ✅ [`UQRA_V0.1.0_EVIDENCE.md`](releases/UQRA_V0.1.0_EVIDENCE.md) |
| 当前 Release | [`v0.3.0`](https://github.com/Jinsongl/UQRA/releases/tag/v0.3.0) | ✅ annotated tag 指向合并提交 `7c2bb050dc3e02882929811b5dd9c8878d17e7d5`；附件下载哈希和仓库外 Python 3.12 smoke 通过 |
| 下一版本 | `v0.3.x` 维护线 | 🔄 当前优先核心质量、行为基线和低风险优化；真实新版本出现时恢复 REL-04 |

## 3. 编号体系与里程碑路线

- `U0--U6`：长期工作流，用于表达项目责任边界和持续状态；
- `M0--M7`：交付里程碑，用于表达可验收的阶段成果；编号稳定表达范围，执行优先级由本看板确定；
- `BENCH/LEG/REG/PKG/CI/SCHEMA/MANIFEST/PROD/DATA/CV/PROV/BUILD/SEC/PATH/REL-*`：里程碑或持续工作流的具体任务 ID；
- `ARCH/TEST/PERF/OPT-*`：跨里程碑工程质量工作流任务 ID，不构成新里程碑，也不扩大 M6 的 Legacy 结案范围。

| 里程碑 | 状态 | 所属工作流 | 任务映射 | 完成门摘要 |
| --- | --- | --- | --- | --- |
| M0 治理与边界 | ✅ 已完成 | U0 | 计划与边界文档 | 进入版本控制并明确项目/论文边界 |
| M1 Runner 契约与可审计发布（原 UQRA-MV1） | ✅ 已完成 | U5、U6 | schema、CLI、示例、evidence、clean-clone、REL-01/02 | `v0.2.0`、双版本 CI 和 required gate 通过 |
| M2.1 Benchmark 注册与配置契约 | ✅ 已完成 | U4、U6 | BENCH-01 | 静态 registry、config v2、双版本 45 项测试和 required gate 通过 |
| M2.2 首个多路径缩减基准 | ✅ 已完成 | U4 | BENCH-02 | FourBranch reduced 的输入身份、DoI、trace、重复性和 PR #6 required gate 通过 |
| M2.3 多问题 Benchmark 验收 | ✅ 已完成 | U4 | BENCH-03、BENCH-04、BENCH-06 | 三基准统一 contract hash 与双版本 `52 passed`；required gate 通过 |
| M3 包装、契约一致性与跨环境质量 | ✅ 已完成 | U6 | PKG-01--04、CI-01、SCHEMA-01/02、MANIFEST-01/02 | 全部任务完成；Windows/Python 3.12 required CI、双 clean-install 和冻结证据通过 |
| M4 受控论文生产接口与下游交付契约 | ⏳ 待开始 | U5、U6 | PROD-*、DATA-*、CV-*、PROV-*、REL-* | 由论文仓库明确接口需求触发；不在 UQRA 管理论文参数、表图或统计结论 |
| M5 可复现发布与供应链自动化 | 🔄 基础能力完成；发布闭环暂缓 | U6 | BUILD-01、SEC-01/02、PATH-01、CI-02、REL-04 | BUILD/SEC/PATH/CI 已完成；REL-04 代码在 Draft PR #15，等待真实新版本执行端到端验收 |
| M6 Legacy 与行为回归证据结案 | 🔄 只读审计可启动 | U1、U3 | LEG-01/02、REG-01/02 | 可获得证据审计完成，不可恢复项正式标记 `unavailable`，统一结案矩阵无歧义 |
| M7 可选软件 Benchmark 扩展 | ⏳ 候选 | U4 | BENCH-05 及后续 BENCH-* | 仅在证明独立软件验收价值后启动；继续禁止历史 replay 和论文生产声明 |

`LEG-01/02` 与 `REG-01/02` 是 U1/U3 的证据闭环任务，不属于 M2，也不因 M2 完成
而自动关闭。

## 4. U0--U6 状态总览

| 阶段 | 状态 | 当前结论 | 下一动作 |
| --- | --- | --- | --- |
| U0 治理冻结与证据清单 | ✅ 已完成 | 项目/论文边界、主计划、历史计划和交接接口已固化 | 随新裁决持续维护证据清单 |
| U1 Legacy 基准恢复 | ⛔ 阻塞 | 可恢复证据有限；历史候选池、测试集和 RNG/CV 状态缺失 | 执行 Legacy 环境审计并形成正式阻塞结论 |
| U2 Modern 核心实现 | ✅ 已完成 | 核心算法完成，当前发布基线为 `v0.3.0` | 核心变更继续遵守裁决和回归门 |
| U3 内核与逐轮行为回归 | ✅ 已完成（可获得证据范围） | Phase 5--8 证据完整；历史 FourBranch trace 不可用 | 建立统一结案矩阵，与 U1 阻塞结论互链 |
| U4 通用软件 benchmark | ✅ 已完成（M2 范围） | M2.1--M2.3 已随 `v0.3.0` 发布；BENCH-05 保留为可选扩展 | 在 `v0.3.x` 维护契约与回归证据 |
| U5 Runner 发布门 | ✅ 已完成 | M1 schema、CLI、示例、evidence、PR #4 和 `v0.2.0` 发布均完成 | 按版本化流程维护后续交付 |
| U6 版本化维护 | 🔄 进行中 | `v0.3.0` 已发布；M5 BUILD/SEC/PATH/CI 基础能力已建立，REL-04 保持 Draft | 维护现有 gate；真实新版本出现时恢复发布闭环，不阻塞核心质量优化 |

## 5. 当前焦点

### ✅ 最近完成

| ID | 归属 / 优先级 | 任务 | 验收证据 | 下一动作 |
| --- | --- | --- | --- | --- |
| REL-01 | M0/M1 / P0 | 合并 M0/M1 交付 | PR #4；Python 3.11/3.12 和聚合 gate 全部通过 | ✅ 已合并到 `master` |
| REL-02 | M1 / P0 | 发布 `v0.2.0` | tag、GitHub Release、required gate | ✅ 已发布 |
| BENCH-01 | M2.1 / P1 | 受控 benchmark registry/config v2 | PR #6；Python 3.11/3.12 均 `45 passed`；required gate 通过 | ✅ 完成，作为 M2.2 注册入口 |
| BENCH-02 | M2.2 / P1 | FourBranch reduced benchmark | 三套独立冻结输入、显式失效定义、DoI/manifest/trace 重复性；PR #6 双版本 `47 passed` | ✅ 完成；不声明为历史 replay 或论文生产结果 |
| BENCH-03 | M2.3 / P1 | Ishigami reduced | 冻结输入、非线性/交互、contract hash；双版本 `52 passed` | ✅ 完成 |
| BENCH-04 | M2.3 / P1 | Gayton reduced | 冻结输入、局部失效域/DoI、contract hash；双版本 `52 passed` | ✅ 完成 |
| BENCH-06 | M2.3 / P1 | 三基准统一验收 | registry/config/identity/禁止声明统一契约；required gate 通过 | ✅ 完成 |
| REL-03 | U6 / P0 | 发布 `v0.3.0` | PR #12 merge `7c2bb050`；required run `31068171070`；Release 附件下载哈希及仓库外 Python 3.12 smoke 通过 | ✅ 完成；冻结证据不再修改 |
| BUILD-01 / SEC-02 / PATH-01 | M5 / P0 | 可复现构建、blob 绑定和 Windows 特殊路径 | PR #14；错误 hash/构建差异拒绝测试；空格、单引号、非 ASCII 路径通过 | ✅ 完成；required run `31146285191` |
| SEC-01 / CI-02 | M5 / P1 | 持续安全审计及唯一正式 gate 扩展 | 固定 `pip-audit`；依赖三分类；Windows/Python 3.12 全 gate；本机/CI 包摘要一致 | ✅ 完成；required run `31146285191` |
| M5 合并验收 | M5 / P0 | 将 BUILD/SEC/PATH/CI 基础能力合入 master | PR #14 merge `514075341a0bb1c198e3f9656d21a800868ea59c` | ✅ master run `31151712201` 全绿 |
| ARCH-01 | 工程质量 / P0 | `uqra/adaptive/` 架构、职责与数据流审计 | [PR #17](https://github.com/Jinsongl/UQRA/pull/17) merge `2d2be3ac467e1f3701b84c5c01bab203d4f8e40b`；[`UQRA_ARCH_01_ADAPTIVE_ARCHITECTURE_AUDIT.md`](UQRA_ARCH_01_ADAPTIVE_ARCHITECTURE_AUDIT.md) | ✅ 完成；PR run `31342024753`、master run `31342298045` 全绿；未修改生产代码或数学契约 |
| TEST-01 | 工程质量 / P0 | 关键行为与现有测试覆盖矩阵 | [PR #18](https://github.com/Jinsongl/UQRA/pull/18) merge `7cfb034ceff26c06aadb9843a228d3dfea65e016`；[`UQRA_TEST_01_ADAPTIVE_COVERAGE_MATRIX.md`](UQRA_TEST_01_ADAPTIVE_COVERAGE_MATRIX.md) | ✅ 完成；PR run `31342526075`、master run `31342818592` 全绿；补充四类高价值回归 |
| PERF-01 | 工程质量 / P0 | Windows/Python 3.12 性能与内存基线 | [PR #19](https://github.com/Jinsongl/UQRA/pull/19) merge `8cf3b228b8e83e36ee1b3aad19c8d9d4c992ec1e`；[`UQRA_PERF_01_ADAPTIVE_BASELINE.md`](UQRA_PERF_01_ADAPTIVE_BASELINE.md) 与机器可读 JSON | ✅ 完成；PR run `31343375325`、master run `31343613965` 全绿；仅用于软件工程验收 |
| 治理 closure | U0 / P0 | 强制任务 closure、PR 声明与确定性 CI 检查 | [PR #21](https://github.com/Jinsongl/UQRA/pull/21) merge `693b1d2c3ba8da2608966d1840f36ee20d65a94a`；master runs `31346375232`、`31346375225` | ✅ 生效；两项检查均已加入 `master` required checks |

### 🔄 进行中

| ID | 归属 / 优先级 | 任务 | 当前证据 | 完成门 |
| --- | --- | --- | --- | --- |
| SCHEMA-01 | M3 / P2 | 对齐 config v2、manifest 与 reduced benchmark 契约 | PR #8；发布 schema 覆盖 config v1/v2、Phase 8 与三个 reduced benchmark | ✅ 完成并合并 |
| SCHEMA-02 | M3 / P2 | 使用 Draft 2020-12 标准校验器验证 config/manifest/trace | PR #8；生成产物通过标准校验，错误 scenario 组合有拒绝测试 | ✅ 完成并合并 |
| MANIFEST-01 | M3 / P2 | 记录完整来源与环境身份 | PR #8；commit、branch、dirty、源码树 hash、Python/依赖和复现命令完整 | ✅ 完成并合并 |
| MANIFEST-02 | M3 / P2 | 记录输入和输出 artifact 身份 | PR #8；输入、trace、结果和摘要的实际大小及 SHA-256 经磁盘复核 | ✅ 完成并合并 |
| PKG-02 | M3 / P2 | 将主要元数据迁移到 `pyproject.toml` | PR #9；`setup.py` 及 upload/tag 逻辑已移除 | ✅ 完成并合并 |
| PKG-03 | M3 / P2 | 建立 `uqra.__version__` 唯一版本源 | PR #9；runtime、distribution、CLI 和 manifest 均报告 `0.2.0` | ✅ 完成并合并 |
| PKG-01 | M3 / P2 | 构建并测试 sdist/wheel | 两种包均含五个 schema；两个仓库外 Python 3.12.13 环境完成安装和 [`evidence`](releases/UQRA_M3_PACKAGING_ACCEPTANCE.md) 验收 | ✅ 完成；required run `31062848792` |
| PKG-04 | M3 / P2 | 清理 Python 3.12 警告 | UQRA 自身 SyntaxWarning/DeprecationWarning 已清理；严格 `compileall` 回归门和 compatibility `61 passed` | ✅ 完成；required run `31062848792` |
| CI-01 | M3 / P2 | 建立 Windows/Python 3.12 required CI | Windows job 覆盖锁、完整 suite、schema、构建、双 clean-install、manifest v2 和 warning 门 | ✅ 完成；required run `31062848792` |
| OPT-01 | 工程质量 / P1 | 首批低风险可维护性优化 | ✅ 完成：PR #23；merge `17fe310474a2e93f79e3e43046aa8bc6b110c41c`；PR compatibility run `31447921046`、governance rerun `31449154193`；master compatibility run `31449264976`、governance run `31449264993` 全绿；Windows/Python 3.12 本地 `tests/packaging tests/compatibility` 为 `78 passed` | 三个 reduced benchmark live contract/trace identity 已固定，共享私有 canonical JSON hashing 已实现；公共 API、JSON bytes、hash scope、数学契约和完整逐轮行为不变；下一动作是另行裁决下一稳定任务，`OPT-02/03` 不自动启动 |

当前实施优先级转为工程质量工作流：`ARCH-01`、`TEST-01`、`PERF-01` 已完成；首批
`OPT-01` 已选定为“固定 live benchmark identities 后统一私有 canonical JSON hashing”。
`OPT-01` 采用本地开发优先：在本地完成实现、定向回归、完整 compatibility suite、
文档和 `git diff --check` 后才首次推送并创建候选 PR；不为中间提交、探索性试验或
未就绪状态反复触发远端 CI。required checks 和合并后 closure 完成门保持不变。
`OPT-02/03` 暂不启动。M6 的只读证据审计可并行；
M4 仍由论文仓库明确需求触发。
REL-04 不再阻塞核心开发，恢复条件见后续队列。里程碑编号不表示强制串行。

### ⏳ 待开始

| ID | 归属 / 优先级 | 任务 | 前置条件 | 完成门 |
| --- | --- | --- | --- | --- |
| LEG-01 | U1 / P1 | Legacy Python 3.8/3.9 环境审计 | 🔄 本地候选：分支 `codex/leg-01-legacy-environment-audit`；两环境历史锁、导入、入口和测试失败已复核；形成 [`审计报告`](UQRA_LEG_01_LEGACY_ENVIRONMENT_AUDIT.md) | 等待本地证据复核后创建单一任务 PR；任务合并及 master required gates 全绿后再以 closure PR 标记完成 |
| REG-01 | U3 / P1 | U3 行为回归结案矩阵 | LEG-01 可并行 | 已验证项与 `unavailable` 项逐项对应证据，无状态歧义 |
| OPT-02 | 工程质量 / P1 | 低风险性能优化 | PERF-01、TEST-01 | 同环境前后数据证明收益；逐轮回归和正式 gate 通过 |
| OPT-03 | 工程质量 / P2 | 受控算法路径优化 | OPT-01/02 证据稳定后另行批准 | 单项独立提交；明确 bitwise 或数值容差契约，不以最终 Pf 接近替代逐轮等价 |

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

### ✅ M5 基础能力与暂缓的发布闭环

| ID | 归属 / 优先级 | 任务 | 完成门 |
| --- | --- | --- | --- |
| BUILD-01 | M5 / P0 | 可复现 wheel/sdist | ✅ 两次独立构建 SHA-256 一致，并固定时间戳和非确定性元数据来源 |
| SEC-01/02 | M5 / P0--P1 | 依赖审计与 Git blob 绑定 | ✅ 审计输入可追溯，错误 hash 有拒绝测试，风险按依赖角色分类 |
| PATH-01 | M5 / P0 | Windows 特殊路径 | ✅ 空格、单引号、非 ASCII 路径已在 Windows runner 回归 |
| CI-02 | M5 / P1 | release gate 自动验收 | ✅ 唯一正式 Windows/Python 3.12 job 已覆盖 BUILD/SEC/PATH |
| REL-04 | M5 / 暂缓 | tag/Release/附件回读自动化 | ⏳ Draft PR #15 保留实现；仅在真实新版本准备完成后恢复端到端发布验收 |

REL-04 暂缓期间保留 Draft PR #15 和 `uqra-release` Environment，但不执行非 dry-run 发布，不创建新 tag 或 Release，也不以 `v0.3.0` 演练覆盖逻辑。恢复条件为：真实新版本号已确定、版本准备 PR 已合并、候选提交的正式 gate 全绿、dry-run 通过，并由人工批准 annotated tag、Release、附件上传和下载回读。任何恢复动作都不得修改 `v0.3.0` tag、附件或四份冻结证据。

### 🔄 当前优先：工程质量工作流

本工作流不新增 M8，也不改变 M4--M7 的名称、编号和边界。`ARCH-01`、`TEST-01`、`PERF-01` 已建立结构、行为与性能基线；首批 `OPT-01` 先固定三个 reduced benchmark 的 live contract/trace identities，再统一私有 canonical JSON hashing。所有优化继续受 `uqra/adaptive/` 核心算法边界和既有数学契约约束；三个 reduced benchmark 始终保持 `software_benchmark` / `reduced`，不得声明为历史 replay、论文生产或 scientific reproduction。`OPT-02/03` 暂不启动。

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
| M3 包装、契约一致性与跨环境质量 | ✅ 已完成 | PR #9；PR #10 merge `1c8040bf`；Windows/Python 3.12 run `31062848792`；冻结 packaging evidence |
| `v0.3.0` 软件交付发布 | ✅ 已完成 | PR #12 merge `7c2bb050`；run `31068171070`；annotated tag、GitHub Release、冻结附件哈希和发布后 smoke |

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

为减少 CI/CD 消耗，稳定任务默认采用单一本地开发周期：允许多个本地小提交和反复测试，
但只在交付物及本地验收证据完整后推送候选分支。除修复远端环境特有失败外，不通过连续
推送把 CI 当作开发调试器。任务 PR 与合并后 closure 仍各自执行 required checks；可合并的
治理 closure 与路线更新应合并为同一个纯文档 PR，避免无意义的额外流水线。

## 9. 不进入本看板的工作

- 正式规模 FourBranch 重复实验；
- PEM/Structural Safety 论文修改；
- 论文表格、图件和统计结论；
- WEC-Sim 工程耦合生产实验；
- 博士论文科学结论的 `pass/partial/fail` 判定。

这些任务由独立论文仓库管理；UQRA 看板只追踪版本化 runner、软件 benchmark、
兼容性证据、发布门和维护质量。
