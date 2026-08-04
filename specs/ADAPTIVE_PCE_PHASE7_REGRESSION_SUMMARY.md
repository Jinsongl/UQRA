# UQRA 兼容自适应稀疏 PCE：阶段 7 行为回归摘要

## 1. 目的与边界

阶段 7 在同一组冻结二维 Hermite 输入上，对 canonical legacy 数值路径与现代 `uqra.adaptive` 实现进行逐轮比较。比较对象是中间数组、进入路径、候选分数、选点 ID、DoI 映射、累积观测和停止位置，而不是最终 QoI 的近似程度。

canonical 基线：

- Git 提交：`09c7042f8c35a262a942224e2367540b5fd2b077`；
- `polynomial_chaos_expansion.py` SHA-256：`232f6f2b65c2855989c024a9f5e9c1302c4490d2deac2eaf65da914a593a5a66`；
- `optimal_design.py` SHA-256：`c9424f5b9d4a83a3f665d1a08eec79e5a299d65506831b7554e151717488f40a`；
- `setting.py` SHA-256：`e724c7f75a468eb77b213949c68f8c7d9a85cd2bfbc0d528cc1561036e263867`。

由于现代 scikit-learn 已删除 canonical 源码使用的 `normalize` 参数，阶段 7 适配层以冻结源码公式为依据，对 canonical OLSLAR、权重缩放和最优设计私有公式做版本兼容翻译。适配层只用于行为回归，不替换投稿控制器。

## 2. 冻结共享输入

共享 fixture 由 `freeze_hermite_inputs` 生成并立即设为只读：

- 多项式：二维、三阶、probabilists Hermite；
- RNG：NumPy `Generator(PCG64)`；
- 输入种子：`90210`；
- CV 种子：`73`；
- 候选数：48；
- 测试点数：72；
- 训练点数：18；
- 候选池哈希：`d039c83d213d76005ff19fe777c453e4c98cb3b06ec11836cf1e334daf781406`；
- 测试集哈希：`2ed85b454ba489711e21b5ae60ecd27479d273fb010db7480a83e4eaf8e534db`；
- 训练 ID、响应和权重联合哈希：`3cd85228431b0f80965bb4d8133ca1d285efaaaba0570f45ecdd112753e08b10`。

重复构造必须得到相同数组和哈希；候选池及测试集均不可写。

## 3. 权重和预处理逐元素比较

canonical `_rescale_data` 的数学行为为：

```text
sqrt_weight = sqrt(w)
WX = diag(sqrt_weight) @ X
Wy = diag(sqrt_weight) @ y
```

现代 `legacy_preprocess(..., fit_intercept=False, normalize=False)` 使用同一输入。比较结果：

- 加权设计矩阵最大绝对误差：`0.0`；
- 加权响应最大绝对误差：`0.0`；
- offset：逐元素为零；
- scale：逐元素为一。

分类：数值等价，无差异。

## 4. LARS 路径、CV 和截断位置

canonical OLSLAR 适配与现代实现共享六折划分及 seed 73。冻结测试折 ID 为：

```text
[2, 11, 15]
[9, 13, 17]
[4, 5, 12]
[6, 7, 16]
[1, 3, 8]
[0, 10, 14]
```

比较结果：

- LARS 进入路径：`[0, 2, 3, 4]`；
- 选定路径前缀：`[0, 2, 3, 4]`；
- legacy 与现代 CV 路径最大绝对误差：`0.0`；
- fold 身份、进入顺序和截断长度完全一致。

分类：数值和行为等价，无差异。

## 5. RRQR 与 D/S-optimal 逐轮比较

二阶 Hermite 设计矩阵的共同 RRQR 初始全秩 ID：

```text
[16, 9, 40, 34, 24, 3]
```

canonical D/S 更新公式返回增量量。适配层将它们转换为与现代实现一致的绝对 log-criterion 后逐候选比较。

### D-optimal

- 三轮选中 ID：`[21, 29, 33]`；
- 全部候选、全部轮次最大绝对分数误差：`3.552713678800501e-15`；
- 每轮 argmax ID 一致。

### S-optimal

- 三轮选中 ID：`[6, 21, 35]`；
- 全部候选、全部轮次最大绝对分数误差：`8.881784197001252e-15`；
- 每轮 argmax ID 一致。

分类：依赖版本和矩阵运算顺序造成的机器精度差异，位于 `2e-12` 声明容差内；行为等价。

## 6. DoI 候选和局部/全局映射

冻结 DoI 测试中心 ID：

```text
[15, 36, 38, 37]
```

legacy 和现代实现得到相同的排序后 DoI 全局 ID：

```text
[1, 3, 4, 5, 6, 11, 12, 18, 22, 23, 26, 28, 30, 31, 32, 33, 37, 39, 42, 43, 44]
```

现代 `doi_local_to_global` 完整保存上述冻结全局顺序，且全部 ID 在构造 DoI 前属于未评估集合。

## 7. IDX-01/IDX-02 有意差异追踪

阶段 7 同时运行字面索引诊断：

- IDX-01：跨阶重新排列候选池后，legacy 保留的局部整数指向不同坐标；现代实现保留首次评估的稳定全局 ID 和坐标哈希；
- IDX-02：legacy 把 DoI 局部行号直接追加到全局集合；现代实现先通过 `doi_local_to_global` 映射。

诊断 trace 哈希：`da31f4bebb92e49260adfa2b393b5ed7ed322bd3808e6f40c414f0dbb9edea31`。

分类：已由 `UQRA_COMPATIBILITY_DECISIONS.md` 裁决的 legacy 源码缺陷修正。此处预期不相等，不得改为数值等价断言。

## 8. 累积观测和停止位置

使用 `literal_legacy` 兼容配置运行一至三阶：

- `order_completed` 顺序为 `[1, 2, 3]`；
- 每阶已评估 ID 集合包含上一阶全部 ID；
- 候选池哈希在所有 trace 中不变；
- 模型调用数等于唯一已评估 ID 数；
- 字面外层控制流遍历全部请求阶次；
- 最终状态：`completed`；
- 停止原因：`literal_orders_completed`；
- 最终停止阶次：3。

分类：与已声明的 literal legacy 外层不停机行为一致。

## 9. 自动化验证

新增 `test_adaptive_behavior_regression.py`，覆盖：

1. 共享 Hermite 输入冻结和哈希；
2. 权重及预处理数组逐元素比较；
3. LARS 进入路径、fold、CV 和截断位置；
4. RRQR、D/S 三轮全候选分数及选中 ID；
5. DoI 候选和局部/全局映射；
6. IDX-01/IDX-02 字面差异分类；
7. 累积观测和 literal 停止位置。

专项结果：7 passed。完整兼容性测试结果：27 passed。

## 10. 阶段结论

阶段 7 首轮行为回归通过：预处理、LARS/CV、RRQR、D/S 评分、DoI 候选和 literal 停止位置均不存在未解释差异；IDX-01/IDX-02 差异已按算法裁决分类为缺陷修正。

声明边界保持不变：尚未恢复历史 $10^5$ 候选池和 $10^6$ 测试集，因此不能宣称已复现博士论文四分支最终数值结果。下一步进入阶段 8 的确定性 benchmark 端到端集成。
