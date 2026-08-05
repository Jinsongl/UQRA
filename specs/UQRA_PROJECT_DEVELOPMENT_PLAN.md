# UQRA-compatible 自适应稀疏 PCE 项目开发计划

状态：执行中  
版本：2026-08-05 split-plan v2  
项目仓库：`https://github.com/Jinsongl/UQRA`  
当前发布基线：`v0.2.0`

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

**状态：未完成；受历史资产和陈旧环境阻塞。**

工作内容：

- 建立隔离的 Python 3.8/3.9 legacy 环境；
- 记录 NumPy、SciPy、scikit-learn 及其他依赖版本；
- 运行并分类 canonical 测试失败；
- 尝试运行 `examples/Branches_AdapSPCE.py`；
- 保存可获得的逐轮输出，不合成缺失历史数据。

完成门：成功得到可追踪 legacy 基准，或形成经证据支持的 `blocked by missing assets` 结论。阻塞状态不能被写成成功复现。

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
- canonical 源码始终保持冻结；
- portable runner 不进入 UQRA 生产实现。

## 4. 交付里程碑与任务编号

`U0--U6` 表示长期工作流，不随单次发布关闭；`M0--M3` 表示可验收的交付
里程碑；`BENCH-*`、`LEG-*`、`REG-*`、`PKG-*`、`CI-*` 和 `SCHEMA-*`
表示看板中的具体任务。任务 ID 不替代里程碑编号，每项任务必须在看板中标明所属
里程碑或持续工作流。

| 里程碑 | 状态 | 范围 | 完成门 |
| --- | --- | --- | --- |
| M0 治理与边界 | 已完成 | 主计划、项目/论文边界、兼容性裁决和证据清单 | 计划与边界进入版本控制 |
| M1 Runner 契约与可审计发布（历史称 UQRA-MV1） | 已完成 | config/manifest/trace schema、CLI、两个示例、evidence package、clean-clone 验收 | Python 3.11/3.12、required CI、发布证据和正式版本全部通过；对应 `v0.2.0` |
| M2.1 Benchmark 注册与配置契约 | 进行中 | 受控 benchmark registry、config v2 和 schema；对应 `BENCH-01` | 配置只能选择已注册 benchmark，禁止任意 Python 导入，并通过 schema 与回归测试 |
| M2.2 首个多路径缩减基准 | 待开始 | FourBranch reduced；对应 `BENCH-02` | 固定输入、seed、数据 hash、DoI 路径、manifest、trace 和重复性测试通过 |
| M2.3 多问题 Benchmark 验收 | 待开始 | `BENCH-03--06` | 至少三个不同性质的缩减 benchmark 通过统一 schema、CI 和声明边界验收 |
| M3 包装与跨环境质量 | 待开始 | `PKG-*`、`CI-*`、`SCHEMA-*` | wheel/sdist、版本源、标准 schema 校验和支持平台 CI 达到发布门 |

`LEG-*` 和 `REG-*` 属于 U1/U3 的证据闭环任务，不并入 M2 benchmark 交付，也不
因 M2 完成而自动关闭。

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

## 6. 当前最小下一目标

**M1 Runner 契约与可审计发布：已完成。** 历史文档中的 `UQRA-MV1` 与 M1
是同一里程碑，不再作为独立编号使用。

完成证据：固定 tag/commit、干净环境命令、版本化 schema、两个配置模板、完整
manifest、Python 3.11/3.12 的 42 项兼容性测试、全新克隆验收、确定性回归结果和
已知限制均已进入 evidence package。论文项目是否拥有 FourBranch 等算例资产不
阻塞 UQRA 通用功能发布，但不得用软件回归结果冒充论文算例复现。

当前最小目标为 **M2.1 Benchmark 注册与配置契约**，对应看板任务 `BENCH-01`。
M2.1 完成后才启动 M2.2；新增正式发布应使用新版本号，不得静默覆盖 `v0.2.0`。

## 7. 历史文档关系

- `ADAPTIVE_PCE_DEVELOPMENT_PLAN.md`：既有阶段 5–11 的详细开发记录；
- `DISSERTATION_NUMERICAL_REPRODUCTION_PLAN.md`：指向独立论文仓库完整复现计划的交接接口，本仓库不再管理 R0--R7；
- `ADAPTIVE_PCE_PHASE*_SUMMARY.md`：已完成工作的证据摘要；
- 本文件：自 2026-08-05 起的 UQRA 项目主计划与项目边界。

历史文件保留用于追溯，但不得再把论文算例执行或论文重写任务纳入 UQRA 开发阶段。
