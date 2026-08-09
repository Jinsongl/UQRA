# UQRA TEST-01 自适应行为覆盖与缺口矩阵

状态：完成

基线：`master` merge commit `2d2be3ac467e1f3701b84c5c01bab203d4f8e40b`

环境门：Windows、Python 3.12、`requirements/compatibility-py312.txt`

范围：`uqra/adaptive/`、runner、schema、三个 reduced benchmark、manifest/artifact/provenance

## 1. 方法与优先级

本矩阵逐条检查测试断言，而不是依据测试名称或行覆盖率推断保护程度。优先级采用：P0
表示安全或不可恢复损失，P1 表示核心行为无可靠保护，P2 表示可信边界/集成缺口，P3
表示维护性增强。本次没有运行 coverage 工具，因此不声明覆盖百分比。

证据层分为：

- **直接断言**：对可观察字段、数组、错误或文件 bytes 逐项断言；
- **固定 hash**：固定输入或 contract hash 间接保护完整离散行为；
- **重复性**：仅证明同环境重复运行一致，不证明与既有发布基线一致；
- **方案**：当前无足够契约，记录具体测试设计但不擅自冻结新容差。

## 2. 关键行为—现有测试—证据—缺口矩阵

| 关键行为 | 现有保护与位置 | 证据强度 | 缺口/结论 | 优先级 |
| --- | --- | --- | --- | --- |
| 固定输入、seed、候选池身份 | `test_shared_canonical_hermite_inputs_are_frozen`；`test_four_branch_reduced_config_is_registered_and_repeatable`；`test_m23_reduced_benchmark_contract` | 数组直接相等、seed/hash 直接断言 | 三 benchmark 输入身份充分；不得改为论文输入声明 | 已保护 |
| 设计点顺序 | `test_rrqr_scores_and_each_selected_id_match_canonical`；`test_candidate_pool_is_frozen_and_observations_accumulate`；Ishigami/Gayton 固定 contract trace hash | 每轮选择直接断言 + 固定离散 hash | FourBranch 只有同环境重复性，未固定公开 expected contract hash | P2 |
| 每轮样本数量与累计性 | `test_multi_round_inner_loop_records_budget_and_refit_evidence`；`test_cumulative_observations_and_literal_stop_position_match_declared_behavior`；trace 的 `evaluated_before/after` | 直接断言预算增长、集合累计；固定离散 hash 间接覆盖两个 benchmark | 未用统一参数化测试逐行断言三个 benchmark 的 count/order 对应关系 | P2 |
| polynomial order 与状态转换 | `test_cumulative_observations_and_literal_stop_position_match_declared_behavior`；`test_trace_stage_matrix_covers_each_declared_transition` | order 列表和 transition 链直接断言 | reduced 三基准的完整 stage/order 序列主要依赖 hash/repeatability | P2 |
| LARS active path、CV folds/path | `test_lars_entry_cv_path_and_truncation_match`；`test_lars_path_is_preserved_separately_from_canonical_columns`；新增 `test_fixed_lars_fixture_preserves_coefficients_cv_error_and_predictions` | canonical 对比、`1e-15` CV 对比、固定 fixture 容差 | BLAS/sklearn 升级仍需正式 Windows gate 判断；不承诺跨版本 bitwise | 已保护 |
| 系数、intercept、prediction | 新增 `test_fixed_lars_fixture_preserves_coefficients_cv_error_and_predictions` | 固定输入、active IDs、系数、intercept、prediction 直接断言 | controller 每 order 的系数未进入 `RoundTrace`，无法由现有 manifest 逐轮审计 | P1（观测性） |
| QoI / Pf / reference metric | Ishigami variance、Gayton reference Pf 直接固定；FourBranch reference Pf 仅重复运行；contract hash 明确排除 `qoi` | 两个 reference metric 精确断言；逐轮 QoI 仅 raw trace repeatability | 缺少声明后的跨平台数值容差；不得用最终 Pf 接近替代逐轮等价 | P1 |
| 停止状态与停止原因 | `test_six_terminal_outcomes`；Phase 8 四 scenario；reduced manifests | 六终态、末状态、state.stop_reason 直接断言 | 已覆盖 declared terminal outcomes | 已保护 |
| runtime failure | `test_six_terminal_outcomes[runtime_failure]` | status/reason、transition、error type/message 直接断言 | 尚未参数化 Vandermonde shape、non-finite QoI、model length mismatch 等来源 | P2 |
| config/schema 拒绝 | schema mismatch、registry/module path、v1/v2、paper production、unknown fields；本次让 mismatch 同时通过 schema 与 runtime 拒绝断言 | Draft 2020-12 + Python runtime 双层 | missing/duplicate scenarios 与坏 output path 主要由 runtime 分支覆盖不足 | P2 |
| manifest stable hash | repeatability；新增 artifact identity 篡改拒绝测试 | 变更 artifact SHA 后 `validate_manifest` 失败 | `validate_manifest` 不读磁盘，磁盘 bytes 由 CLI 集成测试独立核验，职责应保持区分 | 已保护 |
| artifact bytes/身份/布局 | `test_cli_materializes_complete_manifest_evidence_package` | 对实际文件 size/SHA、输入 shape/dtype、trace scopes 直接断言 | 写入中断、只写部分文件和既有路径冲突没有原子性契约 | P2 |
| provenance 正常路径 | 同一 CLI evidence-package 测试 | Git/environment/source tree/reproduce command 直接断言 | 已有正常路径 | 已保护 |
| provenance Git 不可用 | 新增 `test_provenance_fails_closed_to_explicit_null_git_identity` | commit/branch/dirty/source tree 显式 `null` | subprocess 非零和 Git executable 缺失目前都折叠为 `_git=None`；契约可接受但粒度有限 | 已保护 |
| 产物身份与禁止声明 | `test_m23_reduced_benchmark_contract`、manifest schema | purpose/scale/replay/production、datasets、trace identity 直接断言 | scientific reproduction 禁止语义主要由文档与字段组合表达，无单独字段 | 已保护 |

## 3. 本次新增的最小保护网

### 3.1 固定 sparse-PCE 数值刻画

- **测试**：`test_fixed_lars_fixture_preserves_coefficients_cv_error_and_predictions`
- **设置**：18 点固定一维多项式 design，固定 6-fold CV 与 seed 19。
- **动作**：执行 `fit_lars_path`。
- **断言**：active/selected IDs、完整 CV path、系数、intercept、CV error 和 prediction。
- **捕获故障**：预处理、LARS prefix、CV 选择、系数缩放或 predict 列映射漂移。

### 3.2 schema/runtime 同步拒绝

- **测试**：扩展 `test_config_v2_schema_rejects_benchmark_scenario_mismatch`。
- **设置**：把 FourBranch 的 scenario 从 `reduced` 改为 `converged`。
- **动作**：分别执行 Draft 2020-12 validator 与 `validate_config`。
- **断言**：两层都拒绝同一无效组合。
- **捕获故障**：发布 schema 与 Python runtime 约束漂移。

### 3.3 artifact identity 篡改

- **测试**：`test_manifest_rejects_tampered_artifact_identity`
- **设置**：生成有效 manifest，替换 output summary SHA-256。
- **动作**：执行 `validate_manifest`。
- **断言**：stable manifest hash 校验失败。
- **捕获故障**：artifact inventory 在不更新顶层身份时被修改。

### 3.4 Git provenance 降级

- **测试**：`test_provenance_fails_closed_to_explicit_null_git_identity`
- **设置**：模拟所有 Git 查询不可用。
- **动作**：生成 provenance。
- **断言**：Git 和 source-tree 未知值显式为 `null`，复现命令仍保留输入输出路径。
- **捕获故障**：Git 不可用时伪造空字符串身份或 provenance 结构缺失。

## 4. 待实施的具体高价值案例

| 建议测试 | 设置与动作 | 必须断言 | 测试层 | 优先级/前置 |
| --- | --- | --- | --- | --- |
| `test_reduced_traces_preserve_round_order_counts_and_ids` | 参数化三个 frozen configs，执行一次 runner | 每行 `evaluated_after == len(ids)`；新增 IDs 是累计序列的有序 suffix；order/stage 序列符合状态机；固定 expected contract hash | contract/integration | P1；先为 FourBranch 冻结并评审 expected contract hash |
| `test_reduced_qoi_trace_matches_declared_tolerance` | 固定三基准与正式 Windows 依赖锁 | 每个 order/inner round 的 QoI/Pf/variance 与冻结向量在明确 `rtol/atol` 内一致 | numerical regression | P1；先由项目裁决跨平台容差，不能沿用 contract hash 排除策略冒充数值等价 |
| `test_controller_trace_exposes_each_order_fit_identity` | 固定 controller fixture | 每次 refit 的 active IDs、系数、intercept、CV error 可从 trace/artifact 审计 | contract | P1；需要先批准 trace schema 扩展，属于公共契约变化，不在 TEST-01 实施 |
| `test_runtime_failure_sources_are_classified` | 参数化坏 Vandermonde、坏 model 长度、non-finite QoI、重复 coordinate | status/reason、error type/message、末 transition、已评估状态一致 | unit/state-machine | P2；无 schema 变更 |
| `test_config_runtime_rejects_all_shape_errors` | 参数化 missing/unknown/duplicate scenarios、坏 output、非对象 runner | schema 和 runtime 同时拒绝；错误类别稳定 | contract | P2；避免绑定完整人类错误文本 |
| `test_artifact_write_partial_failure_contract` | 在第二个文件写入时注入 I/O failure | 明确保留/清理策略，无 manifest 指向缺失文件 | integration | P2；先决定原子写入契约 |

## 5. Suite 质量风险

1. reduced benchmark 测试重复执行完整 runner，正式 suite 时间受 scikit-learn/BLAS 性能影响；
   PERF-01 应测量但不得通过减少契约断言换取速度。
2. 部分测试只比较两次运行的 hash；它能发现不确定性，不能发现两次一致地产生的新错误。
3. contract trace 有意排除 `cv_path` 与 `qoi`。它适合离散跨平台契约，但不能证明完整数值
   等价；raw trace、固定数值容差和 contract hash 必须分别表述。
4. `tmp_path` 测试依赖可写临时目录；本地沙箱需显式提供 workspace basetemp，正式 Windows
   CI 的 runner temp 已满足该条件。
5. 测试使用 monkeypatch 隔离 Git 不可用路径，不调用真实损坏仓库，属于最低可靠测试层；
   正常 Git 与实际文件 bytes 仍由 CLI 集成测试覆盖。

## 6. 执行顺序与完成判定

1. 本次先补系数/CV、schema/runtime、artifact tamper、provenance fallback 四项低成本测试。
2. 运行正式范围 `tests/packaging tests/compatibility`，并由唯一 Windows/Python 3.12 CI gate
   验收。
3. PERF-01 固定并测量三个 reduced benchmark，不修改上述行为保护。
4. OPT-01 前优先实施无需 schema 变更的 P2 参数化失败测试。
5. 逐轮系数观测性、QoI/Pf 容差、FourBranch 固定 contract hash 均须单独裁决后实施；不得
   在 OPT-01 中静默改变 trace schema 或数学契约。

TEST-01 已形成关键行为、现有测试、证据位置、缺口与具体方案的可追溯矩阵，并补齐四项
不改变生产代码的高价值回归。PERF-01 可在此保护网稳定后开始；OPT-02/03 仍不启动。
