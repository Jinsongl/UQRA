# UQRA 兼容自适应稀疏 PCE：阶段 8 确定性 benchmark 摘要

## 1. Benchmark 定义

阶段 8 使用低成本、确定性的二维 probabilists Hermite benchmark 验证完整控制器：

\[
y(\xi_1,\xi_2)
=1+0.8\xi_1+0.35(\xi_2^2-1)+0.1\xi_1\xi_2.
\]

冻结配置：

- RNG：NumPy `Generator(PCG64)`；
- 输入种子：`424242`；
- CV 种子：`8080`；
- 候选池形状：`(2, 96)`；
- 测试集形状：`(2, 128)`；
- 候选池哈希：`8d704fb57e7a4f61c8d1cd79cfe5ae2797d876127731b3cea75bee06391dfb20`；
- 测试集哈希：`0b4abeeea9cb8974268b6f15e9f2f970135151da83f990aa2bb11c81697df6a4`；
- DoI 中心：冻结测试集前四个坐标；
- DoI 初始半径：`0.55`；
- 最小 DoI 数量：4；
- 最优设计：S-optimal；
- 最大阶内迭代：3。

候选池和测试集在生成后立即设为只读。所有场景共享同一输入。

## 2. 完整执行路径

正常收敛、最高阶非收敛和过拟合回退场景均实际执行：

```text
冻结候选池
  -> 一阶初始设计与拟合
  -> 全局增广与重拟合
  -> 升阶
  -> 全局增广与重拟合
  -> DoI 构造
  -> DoI 局部选点映射为全局 ID
  -> DoI 增广与重拟合
  -> 阶内停止
  -> 阶间判断
```

运行失败场景使用返回非有限响应的确定性模型，在第一次真实模型评估处进入 `runtime_failure`，用于验证错误 manifest 和无部分观测提交行为。

## 3. 四类场景结果

| 场景 | 状态 | `stop_reason` | 调用数 | trace 行数 | 全局重拟合 | DoI 重拟合 | 最终阶次 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `converged` | `converged` | `outer_qoi_converged` | 12 | 15 | 2 | 1 | 2 |
| `max_order` | `nonconverged` | `max_order_reached` | 20 | 23 | 3 | 2 | 3 |
| `overfit_fallback` | `overfit_fallback` | `overfit_rebuild_not_converged` | 20 | 25 | 3 | 2 | 2 |
| `runtime_failure` | `runtime_failure` | `runtime_failure` | 0 | 2 | 0 | 0 | 1 |

对应 trace 哈希：

- `converged`：`c25652b6a7c137988ef4e5196725c4e643b09672f0da25d36cc9f3cbaee4a77e`；
- `max_order`：`3292c38bc3bef255848f92bd9bf716ea1fa24242c1f70a53750dd9cd05c04614`；
- `overfit_fallback`：`ff53db846dbdc95f69a1999e507ac4edace9f9ab35f660025b346126e2688767`；
- `runtime_failure`：`916fbf79ac99da25a110bd0b2a9065791ad1fa9bc727af4329039b628afd0f62`。

稳定 manifest 内容哈希：`9dfbb1edc928466c55c096fab451b14f570358446139412ee5393d6945edeeb4`。Git provenance 保留在 manifest 中，但不参与稳定内容哈希，避免仅因提交阶段 8 代码而改变 benchmark 内容身份。

## 4. 过拟合场景披露

过拟合回退场景使用真实 LARS/CV 拟合，但为了确定性覆盖三阶严格上升触发条件，benchmark harness 把每阶用于状态机判断的 CV 标签固定为：

```text
p=1 -> 1.0
p=2 -> 2.0
p=3 -> 3.0
```

该调度只用于状态机分支验证，并在 manifest 的 `trigger_disclosure` 中记录。它不是自然产生的统计实验结果，不得用于论文精度结论。

## 5. Manifest 内容

`uqra.adaptive.benchmark` 为每次运行记录：

- benchmark schema 和名称；
- RNG、随机种子、输入形状和哈希；
- Git 仓库、分支、提交及工作树状态；
- Python、NumPy、SciPy 和 scikit-learn 版本；
- 完整兼容配置；
- 状态、停止原因和触发披露；
- trace 哈希、完整 trace 和阶段计数；
- 已评估全局 ID 和坐标哈希；
- 模型调用数、最终阶次、CV 和 QoI 历史；
- 全部样本身份不变量检查；
- 固化的复现命令。

生成的 manifest 属于实验结果，不提交到 `specs`；本摘要只记录稳定身份和验收结果。

## 6. 样本身份验收

四个场景均自动验证：

1. `AdaptiveState.assert_invariants()` 全部通过；
2. 每个新增 ID 在其选择前未出现于累计集合；
3. trace 中观测集合只累积、不删除；
4. 所有 trace 的候选池哈希不变；
5. `model_call_count == len(evaluated_global_ids)`；
6. `model_call_count == len(evaluated_coordinate_hashes)`。

非有限模型在写入任何观测前失败，因此运行失败场景的调用和累计观测数均为零。

## 7. 可重现命令

在已安装 UQRA 及声明依赖的环境中运行：

```text
python -m uqra.adaptive.benchmark --scenario all --output artifacts/adaptive_phase8_manifest.json
```

也可以把 `--scenario all` 替换为 `converged`、`max_order`、`overfit_fallback` 或 `runtime_failure`。命令会创建输出目录并写入 UTF-8 JSON。

## 8. 自动化验证和结论

新增 `test_adaptive_benchmark.py`：

- 连续运行两次套件并比较稳定 manifest 与各场景 trace 哈希；
- 验证四类状态和停止原因；
- 验证三类完整场景均经过全局与 DoI 重拟合；
- 验证全部样本身份不变量；
- 通过 CLI 向临时目录写入并重新读取 JSON manifest。

阶段 8 专项结果：2 passed。完整兼容性测试结果：29 passed。

阶段 8 验收通过。下一步按计划进入阶段 9：恢复历史候选池、测试集、四分支输入和随机状态，并验证历史结果是否受到 IDX-01/IDX-02 影响。
