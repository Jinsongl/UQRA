# 自适应 PCE 阶段 10：环境与 CI 固化摘要

## 1. 结论

阶段 10 已完成。UQRA 不再依赖 `pyDOE2`，Python 3.12 中已删除的 `imp` 模块不再进入导入路径；运行依赖已清理，Python 3.11 和投稿目标 Python 3.12 分别具有锁文件；CI 以稳定的 `Adaptive compatibility gate` 聚合两套环境，并显式执行阶段 6--9 的必需测试。

许可证元数据未在本阶段改动。仓库 `LICENSE` 与包元数据的核实仍按开发计划作为独立事项处理。

## 2. pyDOE2 移除

根因是 `uqra.experiment.lhs` 顶层导入 `pyDOE2`，而 `pyDOE2` 初始化时加载 `doe_factorial.py`，其中使用 Python 3.12 已删除的 `imp`。UQRA 实际只使用 `pyDOE2.lhs`。

现由 `uqra.experiment._lhs` 提供内部 LHS 生成器，保留：

- `random_state` 为整数、`RandomState` 或 `None`；
- classic、center、maximin、centermaximin 和 correlation criterion；
- 每维每个 Latin stratum 恰好一个样本；
- 相同 seed 的确定性输出。

兼容测试同时断言导入 UQRA 和运行 LHS 后 `pyDOE2` 不在 `sys.modules` 中。`setup.py` 和依赖锁均不再声明 `pyDOE2` 或占位包 `sklearn`。

## 3. 依赖边界和锁

`setup.py` 规定支持 CPython `>=3.11,<3.13`，运行范围为：NumPy 1.26、SciPy 1.15、scikit-learn 1.6、pandas 2.2、statsmodels 0.14 和 tqdm 4.x。`pyproject.toml` 固化 setuptools 构建后端。

认证环境分别记录在：

- `requirements/compatibility-py311.txt`：维护基线；
- `requirements/compatibility-py312.txt`：投稿目标。

两份锁当前统一使用 NumPy 1.26.4、SciPy 1.15.2、scikit-learn 1.6.1、pandas 2.2.3、statsmodels 0.14.4、tqdm 4.67.1 和 pytest 8.3.5。`uqra/requirements.txt` 仅保留为旧安装入口，转向 Python 3.11 锁文件。

## 4. 干净 Python 3.11 验证

2026-08-04 在新建 `C:\tmp\uqra-phase10-py311` venv 中执行：

```powershell
python -m pip install -r requirements\compatibility-py311.txt
python -m pip install --no-deps -e .
python -m pytest tests\compatibility -q
```

环境为 CPython 3.11.15；最终测试结果为 `35 passed`。运行未设置 `PYTHONPATH`，UQRA 从仓库的可编辑安装导入，`pyDOE2-loaded=False`。本机未安装 Python 3.12，因此没有把本地 3.12 运行冒充为已完成；该环境由 CI 矩阵强制验证。

## 5. CI 与合并门槛

`.github/workflows/adaptive-compatibility.yml` 在 pull request 及维护分支 push 上运行 Python 3.11/3.12 矩阵。每个矩阵环境：

1. 从对应锁文件安装；
2. 使用 `pip install --no-deps -e .` 验证包安装；
3. 显式执行阶段 6 Hermite、阶段 7 行为回归、阶段 8 benchmark 和阶段 9 历史诊断；
4. 执行全部 `tests/compatibility`。

聚合作业名固定为 `Adaptive compatibility gate`，只有两个矩阵项都成功时才成功。仓库分支保护应将这一稳定 check 名设置为 required；workflow 已提供门槛，但 GitHub 分支保护策略仍属于仓库管理员配置。

## 6. 下一阶段

阶段 11 可开始准备投稿实验与合并发布：冻结 publication 配置和实验协议、预注册预算及停止规则、执行敏感性分析，并以提交哈希、输入哈希和 manifest 作为复现入口。
