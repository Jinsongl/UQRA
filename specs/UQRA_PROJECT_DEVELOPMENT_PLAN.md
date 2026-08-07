# UQRA-compatible 自适应稀疏 PCE 项目开发计划

状态：执行中  
版本：2026-08-07 split-plan v5
项目仓库：`https://github.com/Jinsongl/UQRA`  
当前发布基线：`v0.3.0`（2026-08-06 发布）

执行进度、当前 PR、阻塞项和下一动作由
`specs/UQRA_PROJECT_PROGRESS_BOARD.md` 维护；本文件只定义项目范围、阶段和验收门。

## 1. 项目目标与边界

本计划只管理 UQRA 软件项目：以博士论文算法定义、冻结的 canonical UQRA 源码和 `UQRA_COMPATIBILITY_DECISIONS.md` 为权威基准，开发并持续完善现代 Python 可用、可复现、可测试的 UQRA-compatible 自适应稀疏 PCE runner 及其通用功能。

本项目不负责运行和完成论文正式规模算例，也不负责论文结构、论点、正文、图表取舍或投稿文件。FourBranch、Gayton、Ishigami、Damped oscillator 等非工程算例的小算量版本可作为 UQRA 通用 benchmark，用于软件功能、状态路径和回归测试；同名算例的大算量正式运行由论文仓库管理。WEC-Sim 等工程耦合算例不纳入 UQRA benchmark。论文项目只能消费本项目通过发布门的 runner 和证据包，不能在论文脚本中复制或另行修改核心算法。

## 2. 权威来源

1. 博士论文中的算法定义和数学说明；
2. 冻结的 canonical UQRA 关键源码；
3. `specs/UQRA_COMPATIBILITY_DECISIONS.md` 中已记录的算法裁决。

canonical 文件只读。核心算法限定在 `uqra/adaptive/`；为支持现代环境，可以在明确记录、具备回归测试且不改变数学契约的前提下修改公共依赖和兼容层。新冲突必须先形成源码/公式证据和兼容性裁决，再修改现代实现。

## 3. 开发阶段

### U0——治理冻结与证据清单

**状态：基本完成。**

工作内容：

- 冻结 canonical 关键文件并记录路径、时间戳和 SHA-256；
- 盘点入口、依赖、测试、候选池、历史输出和硬编码路径；
- 建立 canonical、dissertation、publication 三层行为差异清单；
- 记录缺失的历史候选池、测试集、RNG/CV 状态和外部数据。

完成门：权威文件、哈希、缺失资产和未裁决差异均有可审计记录。

### U1——Legacy 基准恢复

**状态：审计已完成；canonical replay 永久受历史资产缺失阻塞。**

工作内容：

- 建立隔离的 Python 3.8/3.9 legacy 环境；
- 记录 NumPy、SciPy、scikit-learn 及其他依赖版本；
- 运行并分类 canonical 测试失败；
- 尝试运行 `examples/Branches_AdapSPCE.py`；
- 保存可获得的逐轮输出，不合成缺失历史数据。

`LEG-01` 的正式结论为 `blocked by missing historical assets`：canonical FourBranch
候选池、测试集、逐轮输出及 RNG/CV 状态均未恢复，现有相关历史 MCS 池和汇总数组
不足以反推原运行。该结论关闭 Legacy 恢复审计，不代表成功复现，也不阻塞 M3、M4
或论文项目建立独立生成的 reconstructed baseline。证据见
`ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md` 和
`ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json`。

完成门：成功得到可追踪 legacy 基准，或形成经证据支持的
`blocked by missing historical assets` 正式结论。后者已经满足；除非发现具有可验证
来源的新历史资产，不再重新打开 canonical replay。

### U2——Modern UQRA-compatible 核心实现

**状态：核心完成。**

工作内容：

- 样本身份、候选池和真实模型调用不变量；
- 数学等价的 legacy `normalize` 预处理；
- 真正的 LAR/LARS 活跃路径、候选子模型 CV 与 OLS 重拟合；
- RRQR 初始化和增量 D/S-optimal 设计；
- 全局 exploration 与 DoI exploitation 双层加点；
- 内外循环、真实外层终止和有限过拟合回退；
- DoI local ID 到 global ID 映射及 legacy 审计字段；
- 配置、日志、manifest、随机种子和停止原因。

完成门：全部已裁决数学行为均有实现与定向测试，不存在未解释的算法旁路。

### U3——内核与逐轮行为回归

**状态：可获得证据范围内完成；完整历史逐轮回归受 U1 缺失资产阻塞。**

本阶段的“完成”限定为真实 Hermite 小型 fixture、可获得的 canonical 内核证据和
冻结输入上的逐轮回归。历史 FourBranch 候选池、测试集、逐轮输出及 RNG/CV 状态
不可恢复，因此不把完整历史实验轨迹一致性列为已完成，也不因此否定已通过的内核
与样本身份回归。

工作内容：

- 比较基排序、Vandermonde、权重和预处理数组；
- 比较 LARS 进入顺序、CV 路径、截断和 OLS；
- 比较 RRQR、D/S 候选评分及选中 global ID；
- 比较 DoI 集合和 local/global 映射；
- 比较逐轮样本、设计矩阵、QoI 和停止原因；
- 将差异分类为裁决修正、浮点容差、配置差异或阻塞缺陷。

完成门：小型 fixture 和可获得的 canonical 基准无未解释差异；最终 `Pf` 接近不能替代逐轮一致性。

### U4——通用软件基准与鲁棒性验证

**状态：缩减规模确定性软件基准已完成。**

工作内容：

- 固定确定性 benchmark，覆盖 converged、max-order、overfit fallback 和 runtime failure；
- 验证重复运行的 trace 与 manifest 哈希；
- 验证候选池、QoI/test set 和 reference MCS set 的身份分离；
- 使用 FourBranch、Gayton、Ishigami、Damped oscillator 等小算量、固定配置的通用 benchmark 验证软件行为；
- 小算量 benchmark 只验证功能、数值路径、状态机、接口和可复现性，不承担论文精度、效率或统计结论；
- WEC-Sim 等依赖工程软件、工程模型或外部仿真链的算例不属于 UQRA benchmark；UQRA 仅测试其通用调用接口或轻量测试替身（若存在）；
- 对论文项目提交的最小缺陷复现增加相应的通用回归测试。

完成门：所有通用状态路径和核心不变量通过。同名 benchmark 的大算量论文运行、正式统计和数值验收不属于本阶段完成条件。

### U4.1 Benchmark 身份标签

UQRA benchmark 的配置、目录和 manifest 必须明确标记：

- `purpose: software_benchmark`；
- `scale: reduced`；
- benchmark 名称和版本；
- candidate/test/reference 集规模及哈希；
- 允许声明仅限软件验证，不得标记为 `paper_production`。

### U5——正式 Runner 发布门

**状态：已完成。**

已完成固定 release tag、Python 3.11/3.12 环境锁、配置驱动入口、双版本兼容性
测试、required CI、版本化配置/runner manifest/trace schema、两个下游示例配置、
冻结 manifest、源码树哈希、行为回归摘要、已知限制和集中 evidence package。
2026-08-05 在全新克隆的独立 Python 3.11.15 环境中按锁文件安装成功，完整兼容性
套件为 `42 passed`，smoke/full 两个配置均生成通过契约校验的 manifest；同日
Python 3.12.13 完整兼容性套件亦为 `42 passed`。交付证据见
`specs/releases/UQRA_V0.1.0_EVIDENCE.md`。

交付物：

- 固定 UQRA Git commit 或 release tag；
- Python/依赖环境锁和可执行入口；
- 配置 schema、结果 schema 和示例配置；
- 自动化测试与行为回归报告；
- 运行 manifest、输入哈希和 trace 说明；
- 已知限制、缺失资产和允许声明清单。

完成门：从干净环境可执行缩减规模验收；CI 通过；发布证据包可以由论文项目独立消费。未经此门，不得向论文项目声明 runner 已可用于正式结果替换。

### U6——维护、缺陷处理与可追溯交付

**状态：持续。**

- 核心算法变更必须引用兼容性裁决并增加回归测试；
- 发布后的行为变化必须产生新版本，不得静默覆盖旧结果；
- 论文项目发现的问题以最小复现、配置、commit 和 trace 反馈；
- 已发布 config、manifest 和 trace schema 的行为变化必须提供版本升级或迁移说明；
- 下游 paper-production 配置只调用已发布的受控接口，不得绕过 runner 直接形成第二套执行契约；
- canonical 源码始终保持冻结；
- portable runner 不进入 UQRA 生产实现。

## 4. 交付里程碑与任务编号

`U0--U6` 表示长期工作流，不随单次发布关闭；`M0--M7` 表示可验收的交付
里程碑；`BENCH-*`、`LEG-*`、`REG-*`、`PKG-*`、`CI-*`、`SCHEMA-*`、
`MANIFEST-*`、`PROD-*`、`DATA-*`、`CV-*`、`PROV-*`、`BUILD-*`、`SEC-*`、
`PATH-*`、`REL-*`、`ARCH-*`、`TEST-*`、`PERF-*` 和 `OPT-*`
表示看板中的具体任务。任务 ID 不替代里程碑编号，每项任务必须在看板中标明所属
里程碑或持续工作流。里程碑编号用于稳定表达范围，不强制串行实施；实际优先级由
进度看板维护。

| 里程碑 | 状态 | 范围 | 完成门 |
| --- | --- | --- | --- |
| M0 治理与边界 | 已完成 | 主计划、项目/论文边界、兼容性裁决和证据清单 | 计划与边界进入版本控制 |
| M1 Runner 契约与可审计发布（历史称 UQRA-MV1） | 已完成 | config/manifest/trace schema、CLI、两个示例、evidence package、clean-clone 验收 | Python 3.11/3.12、required CI、发布证据和正式版本全部通过；对应 `v0.2.0` |
| M2.1 Benchmark 注册与配置契约 | 已完成 | 受控 benchmark registry、config v2 和 schema；对应 `BENCH-01` | 配置只能选择已注册 benchmark，禁止任意 Python 导入；PR #6 双版本 45 项兼容性测试和 required gate 通过 |
| M2.2 首个多路径缩减基准 | 已完成 | FourBranch reduced；对应 `BENCH-02` | 固定输入、seed、数据 hash、DoI 路径、manifest、trace、重复性测试和 PR #6 required gate 通过 |
| M2.3 多问题 Benchmark 验收 | 已完成 | `BENCH-03`、`BENCH-04`、`BENCH-06` | FourBranch、Ishigami、Gayton 通过统一 schema、身份、contract hash、双版本测试和 required gate |
| M3 包装、契约一致性与跨环境质量 | 已完成 | `PKG-*`、`CI-*`、`SCHEMA-*`、`MANIFEST-*` | wheel/sdist、版本源、运行时与发布 schema 一致、标准 schema 校验、完整来源/环境身份和 Windows/Python 3.12 required CI 达到发布门 |
| M4 受控论文生产接口与下游交付契约 | 待开始 | `PROD-*`、`DATA-*`、`CV-*`、`PROV-*`、`REL-*` | 论文仓库可通过版本化配置和冻结外部数据调用唯一 UQRA runner，并获得通过正式 schema 校验的 manifest、trace、来源环境身份和输入/输出哈希 |
| M5 可复现发布与供应链自动化 | 基础能力完成；发布闭环暂缓 | `BUILD-*`、`SEC-*`、`PATH-*`、`CI-*`、`REL-*` | BUILD/SEC/PATH/CI 已进入正式 Windows/Python 3.12 gate；REL-04 保留 Draft 实现，在真实新版本出现时完成端到端验收 |
| M6 Legacy 与行为回归证据结案 | 只读审计可启动 | `LEG-*`、`REG-*` | 可恢复证据完成审计；不可恢复项以带依据的 `unavailable` 正式关闭；统一结案矩阵不存在状态歧义 |
| M7 可选软件 Benchmark 扩展 | 候选 | `BENCH-*` | 仅在具有独立软件验收价值时增加 reduced benchmark，并完整进入 registry、schema、manifest、身份和 compatibility gate |

`LEG-*` 和 `REG-*` 属于 U1/U3 的证据闭环任务，不并入 M2 benchmark 交付，也不
因 M2 完成而自动关闭。

### M3 任务边界

M3 修复现有软件交付契约和实现之间的差异，不引入具体论文算例：

- `SCHEMA-01`：修复 config v2 与 runner manifest schema 的引用、benchmark 和必需字段不一致；
- `SCHEMA-02`：使用 Draft 2020-12 校验器实际验证 config、manifest 和 trace；
- `CI-01`：以 Windows + Python 3.12 作为唯一正式、持续验证矩阵，并将标准 schema、包装和 clean-install 验收加入 required CI；
- `MANIFEST-01`：统一记录 Git commit、dirty 状态、源码树 hash、Python/依赖版本和复现命令；
- `MANIFEST-02`：统一记录输入、结果文件、trace 和输出摘要的路径、大小与哈希；
- `PKG-01`：确保 schema、验证依赖及相关资源正确进入 wheel/sdist；
- `PKG-02`：将名称、版本、许可证、Python 范围、依赖、包发现和 CLI 元数据迁移到 `pyproject.toml`；
- `PKG-03`：建立 `uqra.__version__` 唯一版本源，并统一构建元数据、CLI 和 manifest 报告；
- `PKG-04`：清理 UQRA 自身在 Python 3.12 下的 SyntaxWarning/DeprecationWarning，并建立回归门。

M3 自 2026-08-06 起采用单一正式验证政策：仅 Windows + Python 3.12 构成持续验证和
完成门。Python 3.11 暂时保留为允许安装但未持续验证，不作为完成门；Ubuntu 上的
聚合或辅助 job 不构成 Ubuntu 软件兼容性声明。若以后取消 Python 3.11 安装支持，
必须同时更新 `requires-python`、README、依赖锁和发布说明。

### M4 任务边界

M4 提供不绑定具体论文参数的受控通用接口：

- `PROD-01`：定义 `paper-production config v1`；
- `PROD-02`：定义 `paper-production manifest v1`；
- `PROD-03`：实现严格校验、禁止任意代码导入的通用执行入口；
- `DATA-01`：定义外部 candidate/test/reference 数据集的角色、shape、dtype、分布、变量顺序、生成协议、文件身份和 SHA-256 契约；
- `CV-01`：记录实际 CV fold identity/hash，而不只记录 seed；
- `PROV-01`：定义 experiment、replicate、主种子及派生种子的身份规则；
- `PROD-04`：增加不承担论文数值结论的小规模 paper-production smoke fixture；
- `PROD-05`：完成干净环境及受支持 Python 版本验收；
- `REL-01`：发布包含生产接口、schema、迁移说明和证据包的新 UQRA 版本。

M4 完成后，具体 `four_branch_reconstructed_v1` 配置、正式规模 candidate/test/reference
生成、重复次数、批量调度、统计聚合、表图和科学结论仍由论文仓库管理。论文仓库可以
包装和批量调用 UQRA 入口，但不得复制核心算法或绕过版本化 runner 直接实例化一套
独立生产流程。

### M5 任务边界

M5 只改进软件发布和供应链质量，不改变 `uqra/adaptive/` 数学契约：

- `BUILD-01`：固定 wheel/sdist 归档时间及其他非确定性元数据，使同一提交重复构建得到相同 SHA-256；
- `SEC-01`：持续审计 Windows/Python 3.12 锁定依赖，并区分运行时、测试/构建和 GitHub Actions 风险；
- `SEC-02`：自动从 Git blob 字节计算锁文件 SHA-256，并将结果绑定安全审计证据；
- `PATH-01`：覆盖含空格、单引号和非 ASCII 字符的 Windows 仓库及临时路径；
- `CI-02`：在唯一正式 Windows/Python 3.12 job 中加入可复现构建、安全证据和特殊路径回归门；
- `REL-04`：自动化 annotated tag、GitHub Release、附件上传、下载回读及 SHA-256 校验，并保留人工发布批准点。

M5 不扩展 Ubuntu 或 Python 3.11 正式支持声明。Release 自动化不得重写既有 tag、覆盖
已发布附件或修改冻结证据；失败时必须在创建 tag 或公开 Release 前停止。

`BUILD-01`、`SEC-01/02`、`PATH-01` 和 `CI-02` 已通过 PR #14（merge
`514075341a0bb1c198e3f9656d21a800868ea59c`；master required run `31151712201`）进入
正式 gate。`REL-04` 的实现和 required run `31152911246` 保留在 Draft PR #15，但端到端
发布验收暂缓。仅当真实新版本号确定、版本准备 PR 合并、候选提交正式 gate 全绿、dry-run
通过且人工批准后，才恢复 annotated tag、Release、附件上传和下载回读；不得使用 `v0.3.0`
演练覆盖逻辑，也不得修改其 tag、附件或四份冻结证据。

### 工程质量工作流

工程质量工作流是跨里程碑的软件维护序列，不新增 M8，也不扩大 M6 的 Legacy 证据治理范围：

- `ARCH-01`：审计 `uqra/adaptive/`、runner、schema、benchmark 与证据生成入口的责任边界和数据流；
- `TEST-01`：建立关键行为、失败模式、现有测试和缺口的可追溯矩阵；
- `PERF-01`：在正式 Windows/Python 3.12 环境固定输入、随机种子和测量方法，建立时间与内存基线；
- `OPT-01`：实施不改变数学契约的低风险可维护性优化；
- `OPT-02`：实施有基线、有回归门且不改变输出契约的低风险性能优化；
- `OPT-03`：仅在前述证据稳定后单独批准算法路径优化，并明确 bitwise 或数值容差契约。

行为回归至少覆盖固定输入与种子、设计点顺序、逐轮样本数、模型阶次、系数、误差指标、Pf、
停止原因和产物身份；不得以最终 Pf 接近替代完整逐轮等价。性能证据仅用于软件工程验收，不能
转化为论文统计结论或 scientific reproduction 声明。三个 reduced benchmark 始终保持
`software_benchmark` / `reduced` 及相应禁止声明。

### M6 任务边界

- `LEG-01`：完成 Legacy Python 3.8/3.9 环境、依赖、入口和失败类型审计；
- `LEG-02`：若原候选池、测试集、逐轮输出及 RNG/CV 状态仍不可得，形成永久 `unavailable` 结论；
- `REG-01`：建立已验证行为、证据位置、容差及不可验证项的统一结案矩阵；
- `REG-02`：不得以最终 Pf 接近替代完整历史逐轮 trace 等价；缺失证据必须显式继承 `LEG-02` 状态。

M6 是证据治理结案，不以 reduced benchmark 冒充历史 replay，也不恢复论文正式算例。

### M7 任务边界

`BENCH-05` damped oscillator reduced 是候选项而非承诺。启动前必须证明它保护现有三个
reduced benchmark 未覆盖的软件行为；若启动，仍须标记 `purpose: software_benchmark`、
`scale: reduced`、`historical_replay: false` 和 `paper_production: false`。新增正式算法能力
或数学契约变化必须另行裁决，不能作为 benchmark 扩展静默进入 `v0.3.x`。

## 5. 向论文项目交付的唯一接口

每次正式交付必须是不可含混的证据包，至少包含：

1. UQRA commit/tag 和工作树状态；
2. 环境锁与运行命令；
3. runner 配置及 schema；
4. 测试和行为回归结论；
5. candidate/test/QoI/reference 数据身份规则；
6. manifest 与输入/输出哈希；
7. 已知限制和禁止声明；
8. runner 的功能成熟度、支持范围和已知限制。

论文仓库不得复制 `uqra/adaptive` 后形成第二套正式算法。需要算法变更时，回到本项目完成裁决、实现、测试和新版本交付。

## 6. 当前执行策略与最小下一目标

**M1 Runner 契约与可审计发布：已完成。** 历史文档中的 `UQRA-MV1` 与 M1
是同一里程碑，不再作为独立编号使用。

完成证据：固定 tag/commit、干净环境命令、版本化 schema、两个配置模板、完整
manifest、Python 3.11/3.12 的 42 项兼容性测试、全新克隆验收、确定性回归结果和
已知限制均已进入 evidence package。论文项目是否拥有 FourBranch 等算例资产不
阻塞 UQRA 通用功能发布，但不得用软件回归结果冒充论文算例复现。

M2.3 已通过 PR #7（merge `1d3dacd`）完成，证据见
`UQRA_M23_BENCHMARK_ACCEPTANCE.md`。**M3 包装、契约一致性与跨环境质量**已启动，
`SCHEMA-01/02` 与 `MANIFEST-01/02` 已通过 PR #8（merge
`116fdca5d1a212814efbb474f31f8ff3ab8d915d`；最终 required run
`31057779411`）完成并合并。`PKG-02/03` 已通过 PR #9（merge
`a72a624e45b1a3437d3335989d61954a3bd22959`；required run `31061159434`）完成。
`PKG-01/04` 与 `CI-01` 已通过 PR #10 的 Windows/Python 3.12 required run
`31062848792`，并以 merge `1c8040bf7b988b08d0488abaee84b09db956808f` 合并；
wheel/sdist、双 clean-install、schema、manifest v2 evidence 和 warning
门证据已冻结，M3 完成。`v0.3.0` 已通过 PR #12（merge
`7c2bb050dc3e02882929811b5dd9c8878d17e7d5`；最终 required run `31068171070`）发布。
M5 的 BUILD/SEC/PATH/CI 基础能力已经完成，`REL-04` 保持 Draft 并按上述条件恢复，
不再阻塞核心开发。`ARCH-01` 已形成
`UQRA_ARCH_01_ADAPTIVE_ARCHITECTURE_AUDIT.md`，当前最小下一目标是完成 `TEST-01` 和
`PERF-01`，再依据审计结果分批实施 `OPT-01/02`；`OPT-03`
必须单独裁决。M6 的只读审计可以并行但应保持独立提交或 PR；M4 仍由论文仓库的明确
版本化接口需求触发；M7 保持候选。所有变更继续通过唯一正式 Windows/Python 3.12 gate，
且不得覆盖既有 tag、Release 附件或冻结证据。

## 7. 历史文档关系

- `ADAPTIVE_PCE_DEVELOPMENT_PLAN.md`：既有阶段 5–11 的详细开发记录；
- `DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md`：指向独立论文仓库完整复现计划的交接接口，本仓库不再管理 R0--R7；
- `ADAPTIVE_PCE_PHASE*_SUMMARY.md`：已完成工作的证据摘要；
- 本文件：自 2026-08-05 起的 UQRA 项目主计划与项目边界。

历史文件保留用于追溯，但不得再把论文算例执行或论文重写任务纳入 UQRA 开发阶段。
