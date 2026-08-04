# 自适应 PCE 阶段 11：投稿实验与合并发布摘要

## 1. 结论

阶段 11 已完成。publication 配置、二维输入、随机种子、预算和停止协议已经冻结；11 组敏感性案例已分别在现代兼容实现和 portable 对照实现上运行；`overfit_fallback` 独立统计；canonical UQRA 历史结果因阶段 9 已确认的数据缺失保持 `unavailable`，未用现代结果冒充。

最终代码审查没有遗留可执行缺陷，完整兼容性回归通过，当前分支可进入 pull request、required check 和默认分支合并流程。冻结结果见 `ADAPTIVE_PCE_PHASE11_FROZEN_MANIFEST.json`。

## 2. 冻结协议

- 输入：phase-8 二维 Hermite benchmark，candidate `(2, 96)`、test `(2, 128)`；
- RNG：NumPy `Generator(PCG64)`，输入和控制器 seed `424242`；
- 多项式阶：1--3；准则：S-optimal；
- 阶预算因子：2.0；每阶最多 3 次内循环；
- CV：4 folds、shuffle、seed `8080`；
- DoI：半径 0.55、最小 4 个候选、`expand` 回退；
- 外层稳定：相对 QoI 变化不超过 5%，连续 1 次；
- accuracy：最终 CV error 不超过 `1e-6`；
- validity：QoI/CV 有限，且稳定 ID、唯一调用、累积 trace 等全部运行时不变量成立；
- overfit：连续三阶 CV 严格上升时只允许一次降阶重建，并单独报告。

协议哈希：`83bd8d563911151e0e2e614ae93c1606fa766131efb2fa551e3cd785bffad398`。

## 3. 敏感性结果

| 案例 | 现代状态 / 停止原因 | 唯一模型调用数 | 解释 |
|---|---|---:|---|
| baseline | converged / outer_qoi_converged | 12 | 预注册基线 |
| budget_low | converged / outer_qoi_converged | 9 | 预算因子 1.5 |
| budget_high | converged / outer_qoi_converged | 25 | 预算因子 2.5 |
| doi_global | converged / outer_qoi_converged | 12 | DoI 回退改为 global |
| doi_skip | converged / outer_qoi_converged | 12 | DoI 回退改为 skip |
| doi_radius_narrow | converged / outer_qoi_converged | 12 | 半径 0.35 |
| doi_radius_wide | converged / outer_qoi_converged | 12 | 半径 0.80 |
| cv_folds_5 | converged / outer_qoi_converged | 12 | 5-fold CV |
| outer_stable_2 | converged / outer_qoi_converged | 20 | 连续稳定次数 2 |
| accuracy_strict | nonconverged / max_order_reached | 20 | accuracy 阈值 `1e-32` |
| overfit_fallback | overfit_fallback / overfit_rebuild_not_converged | 20 | 披露的固定 CV 标签 1、2、3 |

全部案例通过 validity。`overfit_fallback` 计数为：modern 1、portable 1、canonical unavailable；没有与普通收敛案例合并统计。

## 4. 三类实现边界

### canonical UQRA

结果类型为 `evidence_only`，状态为 `unavailable`。允许引用阶段 7 的 canonical kernel fixture 和阶段 9 历史清单；禁止声明复现历史最终 Pf、逐轮选点或停止位置。

### 现代兼容实现

使用 `AdaptiveSparsePCE`、稳定全局 ID、显式 DoI 映射和 canonical `uqra.Hermite`。表中数值均属于该实现。

### portable 对照

使用同一控制器，但 Vandermonde 由独立 NumPy Hermite-E 计算器生成。1--3 阶最大逐元素误差为 0；全部 11 个案例与现代实现的状态、停止原因、累积选中 ID 和 trace hash 一致，QoI 绝对差为 0。

## 5. 最终代码审查

审查范围包括 publication 新增代码及其调用的 controller、profile、Hermite、LARS/OED/DoI、状态不变量、阶段 6--10 测试和 CI。审查过程中发现并修正：

1. 冻结协议曾返回共享的可变全局字典，调用方可能污染后续 manifest；现每次返回独立副本并有回归测试。
2. 回退场景的 final QoI 曾取回退前 outer history；现从 trace 反向取得最后一次实际重拟合 QoI。
3. manifest 增加 `source_tree_hash`，使提交前生成的冻结结果仍可验证实际源码内容。

修正后无 P0--P3 可执行发现。残余限制只有已登记的历史 canonical 数据不可用，以及大规模论文生产实验尚未执行；二者均未被当前低成本投稿协议结果掩盖。

## 6. 复现与发布准备

复现命令：

```text
python -m uqra.adaptive.publication --output artifacts/adaptive_phase11_publication_manifest.json
```

冻结 manifest hash：`5257f1c12037c0388da2f33346ed6648bd99d8c8cbf60a03bc0c310bbb8d06e6`；源码树 hash：`92b469be99e216e3d7716fa1f8682d65c9594338ceff11bf8738a0d8caa18560`。

CI 的阶段必需测试已加入 `test_adaptive_publication.py`。合并前应创建 pull request，等待 `Adaptive compatibility gate` 在 Python 3.11/3.12 均通过，再由仓库管理员执行默认分支合并和发布标签操作。
