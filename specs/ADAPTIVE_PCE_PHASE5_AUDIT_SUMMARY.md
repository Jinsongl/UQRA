# UQRA 兼容自适应稀疏 PCE：阶段 5 审查与阶段 6 验证摘要

## 1. 审查范围

本次审查覆盖：

- `uqra/adaptive/controller.py` 的双层控制流、状态转换和终止出口；
- `uqra/adaptive/state.py` 的样本身份和运行时不变量；
- `uqra/adaptive/profiles.py` 的阶内停止和预算配置；
- `tests/compatibility` 中的状态机、身份、数值内核和真实 Hermite fixture；
- `UQRA_COMPATIBILITY_DECISIONS.md` 第 3--6 节及源码缺陷表。

审查基线分支为 `agent/adaptive-pce-compatibility`。阶段 5 开始前的实现提交为 `4c9a9e8653fdb07272f8ddb4ca62a50b7050432e`。

## 2. 审查发现及处理

### 已修正的 P1 问题

1. **阶内循环只执行一次**：原控制器每阶最多执行一轮全局增广和一轮 DoI 增广，无法表达文档要求的多轮“增广—重拟合—稳定检查”。现已实现受阶次预算、候选池、QoI 稳定条件和 `max_inner_iterations` 约束的显式循环。
2. **运行失败没有形成结果对象**：原实现捕获异常后仅设置 `runtime_failure`，随后重新抛出异常，调用方无法取得规定的状态、停止原因和失败 trace。现已返回 `AdaptiveResult(status="runtime_failure")`，并记录异常类型和消息。
3. **trace 缺少决策证据**：原 trace 不能恢复阶次目标、预算、增广前后样本数、剩余预算、重拟合次数和回退证据。现已补齐这些字段并记录每次状态转换。

### 修正后结论

未发现仍会阻止进入真实 Hermite fixture 的阶段 5 正确性问题。残余风险主要是尚未完成 legacy/现代逐轮数值对照和历史大型输入恢复；这些工作属于阶段 7 和阶段 9，不能由当前测试结果替代。

## 3. 控制器状态转换

控制器使用 `ControllerState` 显式记录以下状态：

| 状态 | 进入条件 | 可到达的下一状态 |
| --- | --- | --- |
| `created` | 控制器完成参数校验和状态初始化 | `order_started`、`terminated` |
| `order_started` | 开始新多项式阶次 | `initial_design_completed`、`terminated` |
| `initial_design_completed` | 补齐当前阶次初始目标或确认无需补点 | `initial_fit_completed`、`terminated` |
| `initial_fit_completed` | 在全部累积观测上完成 LARS/CV/OLS | `global_refit_completed`、`inner_loop_stopped`、`terminated` |
| `global_refit_completed` | 完成一批全局增广并重新拟合 | `doi_constructed`、`global_refit_completed`、`inner_loop_stopped`、`terminated` |
| `doi_constructed` | 从未评估全局候选构造 DoI 并保存映射 | `doi_refit_completed`、`inner_loop_stopped`、`terminated` |
| `doi_refit_completed` | DoI 局部选择映射为全局 ID、评估并重拟合 | `global_refit_completed`、`inner_loop_stopped`、`terminated` |
| `inner_loop_stopped` | QoI 稳定、预算耗尽、候选耗尽、无新增点或达到显式迭代上限 | `order_completed`、`terminated` |
| `order_completed` | 保存阶次 QoI、CV 和完整诊断 | 下一阶 `order_started`、`overfit_detected`、`terminated` |
| `overfit_detected` | 连续三阶 CV 严格上升 | `overfit_rebuild_completed`、`terminated` |
| `overfit_rebuild_completed` | 使用全部累积观测在 $p-1$ 唯一重建一次 | `terminated` |
| `terminated` | 到达任一有限终止出口 | 无 |

每条 trace 同时保存 `transition_from` 和 `transition_to`。自动化测试断言相邻 trace 的前后状态连续。

## 4. 阶内循环停止出口

| `inner_stop_reason` | 条件 |
| --- | --- |
| `inner_qoi_stable` | 相对 QoI 变化连续满足配置的容差和次数 |
| `order_budget_reached` | 已评估样本数达到当前阶次预算 |
| `candidate_pool_exhausted` | 冻结候选池中没有未评估候选 |
| `no_candidates_added` | 本轮全局和 DoI 阶段均未增加样本 |
| `max_inner_iterations` | 达到显式安全迭代上限 |

停止 trace 记录当前阶次目标、预算、剩余预算、迭代编号、重拟合总数、模型调用数和候选池哈希。

## 5. 六类最终结果测试矩阵

| 结果状态 | `stop_reason` | 触发场景 | 自动化覆盖 |
| --- | --- | --- | --- |
| `converged` | `outer_qoi_converged` | 投稿配置满足相邻阶 QoI、accuracy 和稳定次数 | 是 |
| `nonconverged` | `max_order_reached` | 最高阶仍不满足收敛条件 | 是 |
| `converged_after_rebuild` | `overfit_rebuild_converged` | 三阶 CV 上升，唯一降阶重建后满足正常条件 | 是 |
| `overfit_fallback` | `overfit_rebuild_not_converged` | 重建模型有限但不满足正常收敛条件 | 是 |
| `runtime_failure` | `runtime_failure` | 模型、Vandermonde、QoI 或拟合产生无效状态 | 是 |
| `completed` | `literal_orders_completed` | literal legacy 保留字面控制流并遍历请求阶次 | 是 |

测试还断言：所有结果最后一条 trace 均进入 `terminated`；重建使用的样本 ID 等于触发时全部累积观测；异常出口记录 `error_type` 和 `error_message`。

## 6. Trace 证据字段

`RoundTrace` 当前保存：

- 阶次、阶段、阶内迭代编号；
- 前一状态和后一状态；
- 累积已评估全局 ID；
- LARS 进入路径、选定路径前缀和 CV 路径；
- 全局/DoI 新增 ID、DoI 全局候选及局部行号；
- QoI、阶次初始目标和阶次预算；
- 增广前后样本数及剩余预算；
- 累计重拟合次数和真实模型调用数；
- 冻结候选池哈希；
- 阶内停止原因和 DoI 回退方式；
- 过拟合重建阶次及重建使用的样本 ID；
- 运行失败的异常类型和消息。

这些字段足以恢复每轮预算消耗、拟合次数、回退输入和终止路径。

## 7. 十二项运行时不变量审计

| 编号 | 不变量 | 实现证据 | 测试状态 |
| --- | --- | --- | --- |
| 1 | 一个 `global_id` 只对应一个坐标 | 冻结只读候选池；按 ID 重算坐标哈希集合 | 通过 |
| 2 | 候选数组哈希跨阶不变 | `candidate_hash` 每次断言；每条 trace 保存哈希 | 通过 |
| 3 | 已评估 ID 不重复 | `evaluate_new` 和 `assert_invariants` 双重检查 | 通过 |
| 4 | 同一坐标最多调用一次真实模型 | `evaluated_coordinate_hashes` 在模型调用前检查 | 通过 |
| 5 | 单输出 benchmark 调用数等于唯一坐标数 | 每轮状态断言及端到端 fixture | 通过 |
| 6 | 每个响应具有对应全局 ID | `y_by_global_id` 键集合必须等于已评估 ID 集合 | 通过 |
| 7 | DoI 局部选择经过显式映射 | `map_doi_local_ids`；trace 同时保存局部和全局 ID | 通过 |
| 8 | 新增样本选择前属于未评估集合 | 全局选择限定 candidate IDs；DoI 映射前检查 | 通过 |
| 9 | 升阶不删除历史观测 | 状态跨阶复用；训练数据始终由累计 ID 重建 | 通过 |
| 10 | `active_path` 保留进入顺序 | 路径不排序；选定列必须是路径前缀 | 通过 |
| 11 | 最优设计并列处理确定 | 稳定候选顺序和首个最大值规则；重复运行测试 | 通过 |
| 12 | 每次终止记录有限枚举原因 | `STOP_REASONS` 校验和六类结果矩阵 | 通过 |

## 8. 真实 canonical Hermite fixture

阶段 5 审查通过后，已进入并完成阶段 6 的首个真实 Hermite fixture：

- 多项式类：`uqra.Hermite(d=2, hem_type="probabilists")`；
- 多指标顺序基准：`(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)`；
- 对二阶归一化 Vandermonde 使用解析值逐元素断言；
- fixture RNG：NumPy `Generator(PCG64)`，种子 `314159`；
- CV 种子：`2718`；
- 候选池哈希：`12d21aa864778e0967425bc5f309b4af1be01bd83c467602cfbf7b57b3166ce0`；
- 测试集哈希：`6c36b08f9f9ece715b4c1ff26554520389a8304b6c62540a0fd8a67198702630`；
- trace 哈希：`c48d45f5e90fd76fda6ac325c7fb0deb1fb5bed4fe030b65c74352dd986c4820`；
- 唯一真实模型调用数：12；
- 重复运行候选池、测试集和 trace 哈希一致。

## 9. 验证结果与下一阶段

验证命令：

```text
python -m pytest tests/compatibility -q
```

结果：20 passed。另有一项来自 legacy `pyDOE2` 使用弃用 `imp` 模块的警告，不属于本次兼容实现回归。

阶段 5 验收通过，阶段 6 的首个真实 Hermite fixture 通过。下一步进入阶段 7：建立 canonical legacy 与现代实现的逐轮行为追踪，依次比较预处理、LARS、CV、RRQR、D/S 候选分数、DoI 映射、累积观测和停止位置。
