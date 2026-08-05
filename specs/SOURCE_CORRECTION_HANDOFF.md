# UQRA 源码缺陷与歧义修正：实施交接上下文

## 1. 任务定位

本文件用于直接启动 UQRA 源码修正工作。实施目录固定为：

```text
E:\HRL_XDev\UQRA
```

所有新建的开发计划、审计摘要、算法裁决、验证协议和阶段总结统一保存到：

```text
E:\HRL_XDev\UQRA\specs
```

本阶段的目标不是清理整个历史代码库，而是建立一个可验证的 UQRA-compatible 自适应稀疏 PCE 核心，并修正会破坏算法身份、LAR 路径、D/S-optimal 选点和双层自适应流程的缺陷。

## 2. 权威来源与优先级

发生不一致时按以下顺序处理：

1. 博士论文中的算法定义与数学论证；
2. canonical UQRA 源码中的实际数值公式和数据流；
3. 当前论文明确采用的实验协议和报告指标；
4. portable adaptive runner 只作为冻结对照，不是算法权威。

canonical UQRA 基准：

```text
路径：E:\HRL_XDev\UQRA
分支：master
提交：09c7042f8c35a262a942224e2367540b5fd2b077
```

详细算法裁决见：

- [UQRA_COMPATIBILITY_DECISIONS.md](UQRA_COMPATIBILITY_DECISIONS.md)

## 3. 当前已确认事实

### 3.1 canonical 算法组成

四分支自适应流程的主要证据文件为：

```text
examples/Branches_AdapSPCE.py
uqra/surrogates/polynomial_chaos_expansion.py
uqra/experiment/optimal_design.py
uqra/setting.py
```

核心流程为：

```text
外层更新多项式阶次
  -> 完整基初始全局设计
  -> 真正的 sklearn Lars 活跃路径
  -> 路径候选子模型交叉验证
  -> 选定活跃基后 OLS 重拟合
  -> 基于活跃基的全局域增量设计
  -> 重新拟合代理模型
  -> 根据代理预测构造 DoI
  -> DoI 内使用相同 D/S 准则增量选点
  -> 再次拟合并更新 QoI
  -> 检查内层和外层停止条件
```

### 3.2 当前 portable runner 不是等价实现

论文仓库中的 portable 实现存在以下简化：

- 使用前向最大相关贪婪选择，不是真正的 LAR/LARS；
- 对活跃索引排序，丢失进入路径顺序；
- S 准则使用最小奇异值，不是 UQRA S-optimal 公式；
- 初始设计使用行范数，而不是 RRQR；
- DoI 回退、预算和停止规则与 canonical 流程不同；
- 没有完整复现 legacy 候选池和样本索引更新语义。

不得继续把 portable runner 的结果描述为 UQRA 原算法结果。

## 4. 必须修正的缺陷

### IDX-01：跨阶复用失效的局部候选索引

位置：`examples/Branches_AdapSPCE.py` 第 107--110、198--199 行附近。

现象：每个多项式阶次重新随机抽取或排列候选池，但跨阶保留 `data_train.xi_index`。相同整数在新候选池中可能对应不同坐标。

正确行为：

- 每次独立重复试验只冻结一次候选池；
- 为候选点分配稳定 `global_id`；
- 跨阶保留已评估 `global_id`、坐标和响应；
- 阶次变化时只重建基函数和设计矩阵。

验收：升阶前后，每个已评估 `global_id` 的坐标哈希完全相同；真实模型不得重复评估同一坐标。

### IDX-02：DoI 局部索引直接写入全局集合

位置：`examples/Branches_AdapSPCE.py` 第 222--244 行附近；`uqra/setting.py::samples_nearby`。

现象：在 `data_cand_DoI` 上选出的局部行号被直接合并到全局 `xi_index`，未通过 `idx_data_cand_DoI` 映射回原候选池。

正确行为：

```python
chosen_global_ids = [doi_global_ids[i] for i in chosen_local_ids]
```

验收：追踪中同时保存 `chosen_local_ids`、`chosen_global_ids` 和坐标哈希；新增全局 ID 必须属于选择前的未评估集合。

### LAR-01：portable 的稀疏选择不是 LAR

权威位置：`uqra/surrogates/polynomial_chaos_expansion.py` 中 `OLSLAR` 分支。

正确行为：

1. 使用真正的 LARS 得到有序进入路径；
2. 对每个路径前缀建立 OLS 子模型；
3. 使用明确种子的交叉验证评估各前缀；
4. 选择 CV 误差最小的路径位置；
5. 在选定活跃基上 OLS 重拟合；
6. 分别保存进入路径与规范列序，不得用排序后的集合代替路径。

验收：固定 fixture 下，现代实现与 legacy 环境的活跃进入顺序、候选子模型数、CV 路径和选定前缀一致。

### OED-01：S-optimal 数学准则不一致

权威位置：`uqra/experiment/optimal_design.py`。

正确行为：

- D-optimal 使用信息矩阵行列式增量；
- S-optimal 使用行列式增量与更新后列范数乘积的组合；
- 实现欠定初始化阶段和满秩增量阶段；
- 初始满秩设计使用 RRQR；
- 并列分数按冻结候选池行序确定化处理。

验收：固定设计矩阵下，逐候选 D/S 分数、RRQR 初始行和每次贪婪选中行与基准追踪一致。

### DOI-01：DoI 规则来源不一致

canonical 源码、博士论文和 portable runner 对以下行为存在差异：

- 中心点是固定数量最近响应点，还是响应带内全部点；
- 某个中心无半径内候选时是否补最近 100 点；
- 是否强制整个 DoI 达到最小数量；
- DoI 不可形成时跳过、扩张还是回退到全局域。

处理方式：不得选一个规则后冒充所有版本。必须建立显式配置：

```text
literal_legacy
dissertation
publication
```

### STOP-01：停止规则存在字面源码与论文意图差异

已确认差异：

- 四分支脚本计算外层收敛，但终止 `break` 被注释；
- `isOverfitting` 主要打印警告，没有执行论文描述的降阶回退；
- 源码批大小为 `min(5, max(3, s))`，部分稿件写为 `min(5, s)`；
- 四分支源码预算为 `alpha * P` 且 `alpha=3`，后续投稿 guard 使用 `2P`；
- 内层稳定所需检查次数不同。

处理方式：每种配置单独声明停止规则、预算、回退和终止状态；结果清单必须记录配置名称及 `stop_reason`。

### COMPAT-01：现代 Python 兼容

需处理：

- 新版 scikit-learn 已删除 `normalize` 参数；
- NumPy 已删除 `np.float` 等别名；
- legacy `KFold(shuffle=True)` 未显式记录 `random_state`；
- `setup.py` 与锁定依赖、许可证声明不一致。

要求：

- 对旧 `normalize` 行为做数学等价预处理，不得简单删除参数；
- 兼容修改必须有针对参数组合的单元测试；
- 所有随机操作接收并记录显式 RNG；
- 许可证问题先记录，不在算法修正中擅自改写法律文本。

## 5. 推荐实施边界

优先在现有 UQRA 仓库中建立清晰的兼容层和测试，不直接破坏 legacy 入口。推荐结构需在动手前结合仓库现状最终确定，可采用：

```text
uqra/
  adaptive/
    state.py
    candidate_pool.py
    sparse_pce.py
    optimal_design.py
    doi.py
    controller.py
    profiles.py
tests/
  compatibility/
specs/
```

这是建议边界，不是已批准的目录设计。修改前应检查现有包导出方式和测试约定，避免创建重复模块。

## 6. 实施顺序

### 步骤 A：先建立追踪和测试 fixture

- 冻结小型二维 Hermite 候选池和测试集；
- 记录输入数组、排序和哈希；
- 定义逐轮 trace schema；
- 建立不依赖历史大型外部数据的内核回归 fixture。

### 步骤 B：修正样本身份

- 实现冻结候选池和稳定 `global_id`；
- 实现按全局 ID 存储的响应；
- 实现坐标级去重和真实模型调用计数；
- 修正 DoI 局部到全局映射。

### 步骤 C：现代化数值内核

- 实现 legacy-compatible normalize/weight；
- 实现真正的 LARS 路径、路径 CV 和 OLS 重拟合；
- 实现 RRQR 与 UQRA D/S 增量分数；
- 为每个内核添加确定性单元测试。

### 步骤 D：实现双层控制器

- 全局增广后重新拟合；
- 用更新后的代理构造 DoI；
- DoI 增广后再次拟合；
- 更新 QoI 和完整诊断；
- 依据具名配置执行停止规则。

### 步骤 E：行为回归

比较顺序：

1. 多指标顺序与 Vandermonde；
2. 权重和预处理；
3. LARS 进入路径；
4. 路径 CV 和截断位置；
5. RRQR 初始化；
6. D/S 全体候选分数及选中 ID；
7. DoI 中心、候选集合及索引映射；
8. 每轮累计训练点；
9. 每轮 QoI；
10. 内外循环停止位置和原因；
11. 最终 QoI、方差和唯一真实模型调用次数。

## 7. 首个最小可验证目标

在一个冻结的小型二维 Hermite fixture 上，完成：

```text
Vandermonde
  -> LARS 路径
  -> 路径 CV
  -> OLS 重拟合
  -> RRQR + 一次全局 S-optimal 加点
  -> 重拟合
  -> DoI 构造
  -> 一次 DoI S-optimal 加点及局部/全局映射
  -> 最终追踪
```

通过标准：

- 候选池和测试集哈希固定；
- LARS 进入顺序与选定路径长度一致；
- RRQR、S-optimal 选中全局 ID 一致；
- DoI 全局 ID 集合及映射一致；
- 无重复真实模型调用；
- 重复运行 trace 哈希一致。

该目标只验证算法内核和身份管理，不宣称已经复现博士论文四分支最终结果。

## 8. 开发过程中的禁止事项

- 不得修改或清理用户已有的无关改动；
- 不得用最终 Pf 接近代替中间行为一致；
- 不得把 portable forward greedy 称为 LAR；
- 不得把最小奇异值准则称为 UQRA S-optimal；
- 不得把 DoI 局部索引直接写入全局集合；
- 不得在同一次运行的不同阶次重新随机化候选池后复用旧索引；
- 不得在没有配置标记的情况下混合 legacy、博士论文和投稿停止规则；
- 不得修改 `LICENSE` 或发布声明而不先完成许可证核实；
- 不得提交或推送，除非用户明确授权。

## 9. 每轮开发交付要求

每次源码修改完成后必须报告：

1. 修改文件和职责；
2. 对应缺陷编号；
3. 保持或改变的数学行为；
4. 新增测试及其覆盖行为；
5. 实际运行的测试命令和结果；
6. 尚未解决的歧义；
7. 是否影响论文已有结果或 portable 对照。

## 10. 当前阻塞与待裁决项

### 外部数据阻塞

历史四分支运行依赖的候选池和测试数据当前未在 UQRA 仓库内找到。缺失历史数据不阻止先完成小型 fixture 的内核修正，但会阻止宣称完整复现历史四分支结果。

### 必须通过基准追踪或作者决定解决的问题

- 历史结果是否实际受到 IDX-01/IDX-02 影响；
- 博士论文 profile 的第一阶 sparsity 初始化值；
- 投稿 profile 最终采用 `2P` 还是其他阶次预算；
- DoI 最小候选数和无候选回退规则；
- 外层过拟合回退的精确定义；
- 哪些 legacy 浮点差异要求逐位一致，哪些允许声明容差。

在这些问题解决前，代码应把选择暴露为配置，不得自行猜测并固化为唯一算法。
