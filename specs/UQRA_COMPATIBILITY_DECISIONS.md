# UQRA 兼容自适应稀疏 PCE：算法裁决与实施上下文

## 文档状态

- 目的：作为修正 portable adaptive runner、建立现代 UQRA 兼容实现的权威交接文档。
- 读者：负责实现、审查和验证替代算法的开发人员与论文作者。
- 范围：候选样本身份、观测累积、LAR 路径选择、全局域/DoI 最优设计、双层循环状态和行为回归。
- 证据优先级：博士论文算法与数学论证；canonical UQRA 源码；当前论文实验协议；portable runner 仅作为冻结的对照实现。
- canonical UQRA：`E:\HRL_XDev\UQRA`，Git 提交 `09c7042f8c35a262a942224e2367540b5fd2b077`。
- 文档目录：`E:\HRL_XDev\UQRA\specs`。
- 当前状态：算法裁决已记录；源码修改和 legacy 行为追踪尚未执行。

## 1. 核心裁决

现代实现必须跨多项式阶次保留已经评估的**样本坐标和稳定候选身份**，不得保留一个被重新打乱或重新抽样的候选子池中的临时局部行号。

目标算法可概括为：

> 在一次自适应运行开始时冻结候选空间，持续累积真实模型观测；阶次变化时重建多项式空间和活跃基；仅在尚未评估的候选样本中，交替执行全局域和 DoI 限制域的最优增量设计。

由此得到以下约束：

1. 每个候选样本在一次运行内拥有不可改变的 `global_id`。
2. `global_id -> 标准空间坐标` 在整次运行中保持不变。
3. 真实模型观测跨内循环和外循环持续保留。
4. 提高多项式阶次时重建的是基函数和设计矩阵，而不是实验历史。
5. DoI 局部行号必须映射回 `global_id` 后，才能写入已评估集合。
6. 同一坐标在一次运行内不得重复调用真实模型。
7. legacy 源码的字面索引行为只用于诊断追踪，不作为投稿实现的数学规范。

## 2. 源码证据与 legacy 歧义

### 2.1 每个阶次独立抽取候选池

`E:\HRL_XDev\UQRA\examples\Branches_AdapSPCE.py` 第 107--110 行在多项式阶次循环内部重新加载候选数据，并再次调用 `random.sample(...)`。与此同时，`data_train.xi_index` 在阶次循环之前初始化（第 76--81 行），跨阶保留，随后又在第 198--199 行作为 `x0` 传入当前阶次的候选池。

因此，同一个整数索引可能在不同阶次指向不同坐标。

示例：

| 阶次 | 局部行 1 | 局部行 3 | 保存的已评估索引 |
| --- | --- | --- | --- |
| $p=2$ 的候选排列 | 样本 A | 样本 B | `[1, 3]` |
| $p=3$ 的候选排列 | 样本 E | 样本 C | 旧 `[1, 3]` 此时表示 E、C |

这会同时造成：

- 错误排除：E、C 没有运行过真实模型，却被当成已评估样本；
- 重复评估：A、B 已经评估，但在新排列中不再被旧索引识别，可能再次调用真实模型。

### 2.2 DoI 局部索引被并入全局索引集合

`samples_nearby` 返回缩减后的 DoI 候选池，以及它相对于原候选池的索引。随后，`get_samples(data_cand_DoI, ...)` 返回 DoI 子池内部的局部行号。然而 `Branches_AdapSPCE.py` 第 230--244 行把这些局部行号直接并入 `data_train.xi_index`，没有通过 `idx_data_cand_DoI` 映射回原候选池。

这是明确的索引空间混用。现代实现必须保存：

```text
DoI 局部行号 -> 冻结候选池 global_id -> 样本坐标
```

已评估集合只能使用 `global_id` 更新。

### 2.3 证据解释边界

- 已确认事实：canonical 四分支脚本包含上述两种索引行为。
- 算法裁决：根据博士论文对累积试验设计和自适应基更新的论证，跨阶保持的是已评估样本点，而不是临时局部行号。
- 未决历史问题：归档结果可能受到旧索引行为影响。在获得 legacy 逐轮追踪前，不得声称现代结果与历史结果数值等价。

## 3. 正确的自适应流程

### 3.1 冻结一次运行的输入

每次独立重复试验开始时：

1. 生成或加载一次候选池
   $\mathcal C=\{\boldsymbol\xi_i\}_{i=0}^{N-1}$；
2. 分配不可改变的 `global_id = 0, ..., N-1`；
3. 冻结候选池顺序并记录坐标数组哈希；
4. 生成或加载测试/预测样本并记录哈希；
5. 记录所有随机种子和随机数生成器类型。

不同重复试验可以使用不同的冻结候选池；同一次重复试验的阶次转换不得暗中更换候选池。

### 3.2 累积真实模型观测

运行状态至少维护：

```python
evaluated_global_ids
y_by_global_id
model_call_count
```

训练数据根据稳定 ID 重建：

```python
xi_train = candidate_xi[:, evaluated_global_ids]
y_train = np.array([y_by_global_id[i] for i in evaluated_global_ids])
```

阶次变化时，这些观测不得清空或重新编号。

### 3.3 重建当前阶次的多项式表示

对于多项式阶次 $p$：

1. 按确定化顺序构造总阶多指标集合；
2. 构造完整候选 Vandermonde 矩阵 $\Psi_p(\mathcal C)$；
3. 用已评估的稳定全局行号提取训练子矩阵；
4. 显式应用采样权重和 legacy 兼容的预处理。

同一候选行始终表示同一坐标，但其多项式特征会随 $p$ 改变。

### 3.4 达到当前阶次的初始设计目标

按照博士论文规则：

\[
n_{\mathrm{target}}
=\left\lceil\max(s_{p-1},0.8P_p)\right\rceil,
\]

其中，$P_p$ 是当前完整基数，$s_{p-1}$ 是上一阶活跃基数。第一个阶次使用明确声明的初始化约定。

只补充缺口：

\[
n_{\mathrm{add}}
=\max\left(0,n_{\mathrm{target}}-|\mathcal X_{\mathrm{eval}}|\right).
\]

已有评估样本作为最优设计的初始化状态。新样本只从未评估集合

\[
\mathcal C_u=\mathcal C\setminus\mathcal X_{\mathrm{eval}}
\]

中选择。

### 3.5 使用真正的 LAR 路径选择子模型

基于当前阶次的全部累积观测：

1. 在完整基上运行真正的 LAR/LARS；
2. 保存基函数进入模型的有序路径；
3. 沿路径构造嵌套候选子模型；
4. 使用已声明的 CV 划分和种子评估每个候选子模型；
5. 按确定化 tie 规则选择 CV 误差最小的路径位置；
6. 在选定活跃基上用 OLS 重新拟合系数。

必须分别保存：

```python
active_path          # 基函数进入 LAR 的顺序
selected_active_ids  # 被选路径前缀在完整基中的列 ID
```

不得排序 `active_path`；排序会破坏行为回归所需的进入顺序。

### 3.6 全局域增量设计

只使用当前选定的活跃列：

1. 以全部已评估全局行为 D-或 S-optimal 的初始设计；
2. 只计算未评估全局候选的增量分数；
3. 无放回地贪婪选择全局批次；
4. 分数并列时按冻结候选池行序处理；
5. 仅对新选中的 `global_id` 调用真实模型；
6. 更新累积观测，并在构造 DoI 前重新执行 LAR/CV/OLS。

实现必须复现 UQRA 的 D-和 S-optimal 数学公式，包括 RRQR 初始化以及欠定阶段到满秩阶段的转换。最大化最小奇异值不是 UQRA 的 S-optimal 准则。

### 3.7 构造兴趣域 DoI

正向可靠度分析使用冻结测试集上的代理预测，定位接近目标极限状态的中心点；逆可靠度分析使用与目标超越概率相对应的代理响应。

DoI 必须从尚未评估的全局候选中构造：

\[
\mathcal C_{\mathrm{DoI}}
=\left\{
\boldsymbol\xi_i\in\mathcal C_u:
\min_{\boldsymbol z\in\mathcal Z}
\|\boldsymbol\xi_i-\boldsymbol z\|_2\le\epsilon
\right\}.
\]

必须保存：

```python
doi_global_ids
doi_local_to_global
```

最近邻回退、最小 DoI 数量、响应带宽和半径扩展等规则在博士论文、canonical 源码和 portable runner 中并不完全一致，必须做成显式兼容配置，不能暗中混合。

### 3.8 DoI 限制域最优增量设计

在当前 DoI 中使用与全局阶段相同的活跃基和 D/S 准则。更新状态前必须把局部行号转换为全局 ID：

```python
chosen_local_ids = local_selector.select(...)
chosen_global_ids = [doi_global_ids[i] for i in chosen_local_ids]
evaluate_new(chosen_global_ids)
```

严禁把 `chosen_local_ids` 直接追加到 `evaluated_global_ids`。

### 3.9 重新拟合并检查内循环

完成一次“全局批次 + DoI 批次”后：

1. 重建 LAR 进入路径；
2. 重新执行路径 CV 截断；
3. 重新进行 OLS 拟合；
4. 更新 QoI 和诊断量；
5. 检查当前兼容配置的内循环停止规则；
6. 检查当前阶次预算。

批大小、稳定检查次数、相对误差分母和阶次预算都是兼容配置参数，不得由 portable runner 的默认值反推 canonical 算法。

### 3.10 外循环升阶

如果外层停止条件尚未满足：

1. 按所选兼容配置更新阶次 $p$；
2. 保留所有已评估全局 ID 和响应；
3. 重建基、Vandermonde 矩阵、LAR 路径和 OED 分数；
4. 只补充新阶次所需的额外样本。

不得清空观测、重新生成候选池，或把旧局部索引用于新的候选排列。

## 4. 兼容配置

实现必须明确区分以下配置：

| 配置 | 用途 | 权威来源 |
| --- | --- | --- |
| `literal_legacy` | 诊断性复现旧 Python 控制流，必要时保留已记录的源码缺陷 | canonical UQRA 源码 |
| `dissertation` | 实现博士论文论证的数学算法 | 博士论文定义和算法 |
| `publication` | 生成新投稿结果的完整、有限、可复现实验协议 | 经明确裁决后的当前论文协议 |

每份结果清单必须记录配置名称，不能把某一配置的结果描述成另一配置的结果。

以下项目必须按配置显式声明：

- 第一阶的初始样本约定；
- 批大小：统一采用已裁决的 `min(5, max(3, s))`；实际新增数量仍受剩余预算和可用候选数截断；
- 阶次预算：四分支源码 `alpha * P`（`alpha=3`）与后续 `2P` guard；
- QoI 稳定所需的检查次数；
- 外层 QoI/score 判断是否真正终止；
- 过拟合回退行为；
- DoI 回退和最小数量；
- CV 折数、shuffle 行为和随机种子。

### 4.1 外层收敛与终止裁决

三个兼容配置采用不同且不可混用的外层控制流：

| 配置 | 外层收敛与终止行为 |
| --- | --- |
| `literal_legacy` | 计算并记录阶间 Pf、score 和过拟合诊断，但保留四分支脚本中终止 `break` 被注释的字面控制流，完整遍历请求的阶次。该结果只用于诊断，不得作为投稿算法结果。 |
| `dissertation` | 只实现博士论文能够由公式、算法或文字直接支持的 QoI 稳定与精度条件。证据未明确连续稳定次数、返回阶次或阈值时，必须把相应字段标记为未决配置，不得从 portable runner 反推。 |
| `publication` | 使用下述有限、显式且可复现的终止状态机；达到条件时返回当前阶次模型，达到最高阶仍不满足时报告 `nonconverged`。 |

对 `publication`，当前阶次只有在内循环正常结束并产生有限数值后，才进入外层判断。至少完成两个相邻阶次，并计算

\[
\delta_Q^{(p)}=
\frac{\left|\hat Q^{(p)}-\hat Q^{(p-1)}\right|}
{\max\left(\left|\hat Q^{(p-1)}\right|,\epsilon_Q\right)}.
\]

外层收敛要求同时满足：

1. $\delta_Q^{(p)}\leq\tau_Q$；
2. 预先声明的 QoI-specific accuracy/validity condition 成立；
3. 当前不存在尚未处理的 overfitting trigger；
4. 当前模型及全部报告指标为有限值。

满足条件时返回当前阶次 $p$ 的模型，并记录 `stop_reason=outer_qoi_converged`。不默认返回前一阶模型。到达 $p_{\max}$ 仍不满足时，记录 `stop_reason=max_order_reached` 和状态 `nonconverged`；最后一个有限模型可以保留用于诊断，但不得标记为收敛。

QoI 稳定要求一次相邻阶次检查还是连续多次检查，必须作为预注册配置 `outer_stable_checks` 记录。若没有额外博士论文证据，`dissertation` 不得擅自增加连续检查次数；`publication` 的选定值必须在实验协议和结果清单中一致，并通过敏感性分析说明其影响。

### 4.2 过拟合检测与有限回退裁决

canonical `isOverfitting` 只打印警告且检测后仍返回 `False`。该字面行为仅属于 `literal_legacy`；现代实现不得复用其返回语义。

三个兼容配置采用以下规则：

| 配置 | 过拟合行为 |
| --- | --- |
| `literal_legacy` | 按原函数计算并记录警告，不触发降阶、重建或终止。 |
| `dissertation` | 仅实现博士论文明确给出的触发条件和回退步骤；任何缺失的 fold、阈值、返回阶次或重建细节必须显式标记为未决，不得补写为历史事实。 |
| `publication` | 使用确定性、最多一次的降阶重建规则，并以独立终止状态区分正常收敛、重建后收敛和非收敛回退。 |

`publication` 的 CV 比较必须使用相同误差定义、固定 fold 划分、固定 shuffle seed 和可比的预处理。连续三个阶次满足

\[
\epsilon_{\mathrm{cv}}^{(p-2)}
<\epsilon_{\mathrm{cv}}^{(p-1)}
<\epsilon_{\mathrm{cv}}^{(p)}
\]

时，设置 `overfit_detected=true`。随后：

1. 保留全部已评估 `global_id`、坐标和真实响应；
2. 回到 $p-1$，使用包含高阶阶段新增观测的完整累积数据；
3. 重建一次该阶的基、LARS 进入路径、路径 CV、OLS 拟合和 QoI；
4. 禁止再次降阶或形成回退循环；
5. 重建模型同时满足正常外层收敛、QoI accuracy/validity 和有限性条件时，返回状态 `converged_after_rebuild`，并记录 `stop_reason=overfit_rebuild_converged`；
6. 重建模型有限但不满足正常收敛条件时，返回状态 `overfit_fallback`，并记录 `stop_reason=overfit_rebuild_not_converged`；
7. 主路径和重建路径都不能产生有效有限模型时，返回 `runtime_failure`。

`overfit_fallback` 不是收敛，不得在论文结果中与 `converged` 或 `converged_after_rebuild` 合并统计。每个 manifest 必须记录三阶 CV 值、触发阶次、fold 身份、随机种子、是否执行重建、重建使用的样本 ID 以及最终状态。

一次阶次完成后的判断顺序固定为：

```text
完成当前阶内循环
-> 检查数值有效性
-> 检查 overfitting trigger
-> 必要时执行唯一一次降阶重建
-> 检查外层 QoI 与 accuracy/validity 条件
-> converged / converged_after_rebuild / overfit_fallback / nonconverged / runtime_failure
```

## 5. 必需的状态模型

替代 runner 应显式维护与下列信息等价的状态：

```python
@dataclass
class AdaptiveState:
    candidate_xi: np.ndarray
    candidate_hash: str
    global_ids: np.ndarray

    evaluated_global_ids: list[int]
    evaluated_coordinate_hashes: set[str]
    y_by_global_id: dict[int, float]
    model_call_count: int

    polynomial_order: int
    full_basis_ids: list[int]
    active_path: list[int]
    selected_active_ids: list[int]

    global_added_ids: list[int]
    doi_global_ids: list[int]
    doi_added_ids: list[int]

    inner_qoi_history: list[float]
    outer_qoi_history: list[float]
    cv_path: list[float]
    stop_reason: str | None
```

类的拆分方式可以不同，但上述身份信息和追踪字段必须保留。

## 6. 不可违反的不变量

必须为以下条件添加运行时断言和自动化测试：

1. 一个 `global_id` 在一次运行中只能对应一个坐标。
2. 候选数组哈希在跨阶过程中不变。
3. `evaluated_global_ids` 不含重复项。
4. 同一坐标哈希最多调用一次真实模型。
5. 对确定性单输出 benchmark，`model_call_count == len(evaluated_coordinate_hashes)`。
6. 每个训练响应都有对应的已评估全局 ID。
7. 每个 DoI 局部选择都通过 `doi_local_to_global` 映射。
8. 全局和 DoI 新增样本都属于选择前的未评估全局 ID。
9. 阶次提高不得删除或重新标记历史观测。
10. `active_path` 保留 LAR 进入顺序。
11. 最优设计的并列分数处理是确定性的。
12. 每次终止都记录有限枚举的 `stop_reason`。

## 7. 待修正的源码缺陷与歧义

| 编号 | 问题 | 处理要求 |
| --- | --- | --- |
| IDX-01 | 阶次循环内重新随机抽取候选子池，同时保留旧局部索引 | 每次运行冻结候选池并使用稳定全局 ID；字面行为只留在诊断适配器 |
| IDX-02 | DoI 局部索引直接并入全局已选索引 | 添加显式局部到全局映射和回归测试 |
| LAR-01 | portable `fit_lar` 是前向贪婪选择，不是 sklearn LARS | 替换为真正的 LARS 路径、逐路径 CV 和 OLS 重拟合 |
| LAR-02 | portable 对已选索引排序 | 分别保留进入路径和规范列序视图 |
| OED-01 | portable S 分数为最小奇异值，不是 UQRA S-optimal | 实现 UQRA S 公式以及欠定/满秩更新 |
| OED-02 | portable 使用行范数初始化，而不是 RRQR | 实现列主元 QR 初始化和确定化 tie 规则 |
| DOI-01 | 源码、博士论文和 portable 的 DoI 回退规则不同 | 作为兼容配置显式选择 |
| STOP-01 | 四分支源码计算外层收敛，但终止 `break` 被注释 | 按第 4.1 节执行：保留字面追踪；博士论文配置只采用有直接证据的条件；投稿配置采用有限外层状态机 |
| STOP-02 | 不同材料中的批大小和预算公式不同 | 批大小裁决为 `min(5, max(3, s))`；预算仍按配置声明并标记，禁止静默混用 |
| STOP-03 | canonical `isOverfitting` 只警告且返回 `False`，不驱动回退 | 按第 4.2 节执行：字面配置只追踪；投稿配置最多执行一次确定性降阶重建，并区分收敛与 fallback 状态 |
| RNG-01 | legacy CV shuffle 和候选抽样未完整暴露随机状态 | 为所有随机操作注入并记录 RNG |
| COMPAT-01 | 新 sklearn 已删除 `normalize` 参数 | 数学复现旧预处理，并覆盖各参数组合测试 |

## 8. 实施顺序

### 阶段 0：冻结证据

- 记录 UQRA 提交号、依赖版本、源码哈希和关键源码片段；
- 恢复历史候选/测试数据，或明确标记为不可用；
- 修改算法前先定义逐轮追踪数据结构。

### 阶段 1：稳定样本身份层

- 建立冻结候选池及不可变全局 ID；
- 建立按全局 ID 存储的已评估响应；
- 添加坐标哈希和重复评估 guard；
- 修正 DoI 局部到全局映射。

该阶段必须先于 LAR/OED 替换，因为所有后续行为比较都依赖可靠的样本身份。

### 阶段 2：兼容数值内核

- 实现 legacy 兼容的权重和标准化；
- 实现真正的 LARS 进入路径和路径 CV；
- 实现活跃基 OLS 重拟合；
- 实现 RRQR 初始化及 UQRA D/S 增量分数。

### 阶段 3：双层自适应控制器

- 连接全局增广、重拟合、DoI 构造、局部增广、再次拟合和停止判断；
- 实现具名兼容配置；
- 输出完整逐轮追踪。

### 阶段 4：行为回归

按以下顺序比较 legacy 与现代追踪：

1. 基排序和 Vandermonde 数值；
2. 权重及预处理数组；
3. LARS 进入顺序；
4. 路径 CV 和选定前缀；
5. RRQR 初始索引；
6. 每个 D/S 候选分数及选定全局 ID；
7. DoI 中心、全局候选 ID 和局部/全局映射；
8. 累积已评估 ID 与坐标；
9. QoI 序列；
10. 内外循环停止位置和原因；
11. 最终 QoI、方差估计和唯一真实模型调用数。

最终失效概率接近不能代替算法等价性验证。

## 9. 首个最小验证目标

使用冻结的小型二维 Hermite fixture，在 legacy 和现代内核中各执行一轮：

```text
完整基与 Vandermonde
  -> 真正的 LARS 进入路径
  -> 路径 CV 截断
  -> OLS 重拟合
  -> RRQR/全局 S-optimal 加点
  -> 重拟合
  -> 构造 DoI
  -> DoI 局部 S-optimal 加点及局部/全局映射
  -> 最终重拟合与追踪
```

验收条件：

- 候选池和测试集哈希固定；
- Vandermonde 与预处理数组在声明的浮点容差内逐元素等价；
- LARS 进入顺序和路径截断长度一致；
- RRQR 与 S-optimal 选定的全局 ID 一致；
- DoI 全局 ID 集合和局部到全局映射一致；
- 不发生重复真实模型调用；
- 停止原因一致；
- 重复运行产生相同的追踪哈希。

该 fixture 只验证兼容机制。在恢复并哈希历史 $10^5$ 候选池和 $10^6$ 测试集之前，不得声称已经复现博士论文四分支数值结果。

## 10. 源码修正完成标准

只有满足以下条件，才能淘汰 portable adaptive runner 或重新标记其用途：

1. 已实现稳定候选身份和累积观测；
2. 已测试真正的 LARS、路径 CV、OLS 重拟合、RRQR 和 UQRA D/S 分数；
3. 全局域与 DoI 加点使用相同且明确声明的活跃基准则；
4. 代码和追踪中都保留局部/全局索引映射；
5. 兼容配置和停止原因均为显式字段；
6. 行为回归覆盖中间状态，而非只比较最终 QoI；
7. 结果清单记录代码版本、兼容配置、依赖、随机种子、输入哈希和运行命令；
8. 论文表述明确区分 canonical UQRA、现代兼容实现和冻结的 portable 对照结果。
