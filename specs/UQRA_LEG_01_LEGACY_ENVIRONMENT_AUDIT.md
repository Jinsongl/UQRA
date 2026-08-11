# LEG-01：Legacy Python 3.8/3.9 环境审计

## 1. 结论与边界

LEG-01 在 Windows 隔离环境中复核了历史依赖、包导入、测试入口和
`examples/Branches_AdapSPCE.py`。Python 3.8.20 与 3.9.23 均能安装历史锁定依赖，
并能导入历史 UQRA 快照；因此 canonical FourBranch replay 的首要阻塞不是解释器或
依赖无法安装，而是脚本声明的历史测试集
`G:\My Drive\MUSE_UQ_DATA\UQRA_Examples\Branches\TestData\Branches_McsE6R0.npy`
不可获得。阶段 9 已同时确认候选池、逐轮输出和 RNG/CV 状态不可获得。

正式结论为 `blocked by missing historical assets`。这关闭 LEG-01 环境审计，但不代表
成功复现历史 FourBranch，也不把现代 reduced benchmark 或相关 MCS 池冒充为 canonical
输入。除非发现具有可验证来源的新历史资产，不重新打开 canonical replay。

## 2. 审计对象

- 历史源码树：Git tree `2b21054f98404da2ad102247814fa24f37a8bea6`；
- 历史包版本：`uqra==0.1.0`，`setup.py` 声明 Python `>=3.6.0`；
- canonical 入口：`examples/Branches_AdapSPCE.py`；
- 历史锁：该 tree 的 `uqra/requirements.txt`；
- 既有资产结论：
  [`ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md`](ADAPTIVE_PCE_PHASE9_HISTORY_RECOVERY_SUMMARY.md)
  和 [`ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json`](ADAPTIVE_PCE_PHASE9_HISTORY_INVENTORY.json)。

审计只在临时快照和仓库内隔离环境中运行，没有修改历史源码、外部归档或
`specs/releases/UQRA_V0.3.0_*` 冻结证据。

## 3. 环境与依赖

| 环境 | 解释器 | 历史直接依赖 | 安装与导入 |
| --- | --- | --- | --- |
| Windows / CPython 3.8 | 3.8.20 | NumPy 1.19.5、pandas 1.1.5、pyDOE2 1.3.0、python-dateutil 2.8.1、scikit-learn 0.24.1、SciPy 1.6.0、`sklearn` 0.0、statsmodels 0.12.1、tqdm 4.56.0 | 成功 |
| Windows / CPython 3.9 | 3.9.23 | 同上 | 成功 |

两套环境均使用 `pip install --no-deps -e .` 安装历史快照并成功导入 `uqra`。审计额外
安装 pytest 7.4.4 作为测试驱动器；它不是历史运行依赖。`pyDOE2` 在两环境均可导入，
但产生 `imp` 弃用警告；这与阶段 10 已记录的 Python 3.12 不兼容边界一致。

## 4. 入口与失败分类

| 入口 | Python 3.8 | Python 3.9 | 分类 |
| --- | --- | --- | --- |
| 仓库根目录 `import uqra` | 成功 | 成功 | 环境可用 |
| 未安装包时直接运行示例 | `ModuleNotFoundError: uqra` | 同左 | 安装前置条件，不是算法失败 |
| editable install 后运行 `Branches_AdapSPCE.py` | 缺少 `Branches_McsE6R0.npy` | 同左 | canonical 历史资产缺失 |
| 旧测试（排除现代 compatibility 与缺失的 `tests.context`） | 18 passed、42 failed | 18 passed、42 failed | 混合型 legacy 测试债务 |

42 项旧测试失败不能整体解释为算法回归。可重复观察到的类型包括：

- 硬编码的 macOS、Google Drive 或外部数据路径导致 `FileNotFoundError`；
- 测试尝试写入未创建的 `Data` 目录；
- 未导入的 `cp`、`doe` 等名称导致 `NameError`；
- 旧 NumPy/SciPy/scikit-learn 接口假设引起 `TypeError`、`AttributeError`、
  `AxisError` 或数值接口错误；
- 个别 solver/fixture 自身状态不完整。

完整测试收集还会遇到两项独立问题：`tests/test_advanced.py` 引用不存在的
`tests.context`；历史 tree 中后加入的现代 `uqra.adaptive` compatibility 测试使用
Python 3.10 才支持的 `int | None` 注解，在 3.8/3.9 收集时抛出 `TypeError`。两者均不
构成 canonical FourBranch 的运行结果。

## 5. Canonical FourBranch 阻塞链

脚本先构造 FourBranch solver 和参数，然后从配置派生测试文件路径，并在进入自适应
逐轮循环前加载 `Branches_McsE6R0.npy`。两环境均在该加载点以相同
`FileNotFoundError` 停止。因此本次不能获得任何新的逐轮选点、QoI、Pf、停止位置或
RNG/CV 状态。

阶段 9 的只读 inventory 已进一步确认：

- canonical 约 10^5 候选池：`unavailable`；
- canonical 约 10^6 FourBranch 测试集：`unavailable`；
- canonical 输入 bundle 和逐轮结果：`unavailable`；
- Python、NumPy 和 CV shuffle 状态：`unavailable`；
- 已恢复正态 MCS 池仅为 `recovered-related`，不能替代 canonical 输入。

## 6. 验收与后续

LEG-01 的完成门已满足：依赖、可运行入口和失败类型形成可审计记录，并以现有
provenance 支持 `blocked by missing historical assets` 结论。下一任务为 REG-01：把
已验证行为、容差、证据位置和 `unavailable` 项汇总为统一结案矩阵。LEG-02/REG-02
继续阻塞，除非恢复带可验证来源的原始历史资产。
