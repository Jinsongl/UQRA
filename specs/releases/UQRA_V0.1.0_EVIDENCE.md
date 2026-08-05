# UQRA v0.1.0 runner 交付证据包

## 1. 目的与版本边界

本证据包供下游项目确认 UQRA runner 的身份、环境、契约、验证结果和允许声明。
首个软件发布基线为 `v0.1.0`，指向合并提交
`6479f1803ae66583616fc3cd65d0c90cefe9053e`。M1 在提交
`c6fea079fa742fca160cdc2b90ab10bcdda364c3` 中为该基线补充版本化配置、
runner manifest、trace schema、配置驱动入口和示例配置；它没有改变
`AdaptiveSparsePCE` 的核心算法契约。

本文件将“发布时算法基线”和“发布后补齐的交付接口”分开记录，不宣称后补文件
已经包含在历史 tag `v0.1.0` 中。后续正式发布应包含这些交付接口并使用新版本号。

机器可读摘要见 `UQRA_V0.1.0_EVIDENCE.json`。

## 2. 安装与支持环境

支持 CPython 3.11 和 3.12：

```text
python -m pip install -r requirements/compatibility-py311.txt
python -m pip install --no-deps -e .
python -m pytest tests/compatibility -q
```

Python 3.12 使用 `requirements/compatibility-py312.txt`。锁文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `requirements/compatibility-py311.txt` | `77d9605d0ad90356e12ed4dc4f5b4dc8287de7b5e3564130268417e69566379e` |
| `requirements/compatibility-py312.txt` | `4b04dc01a6443a8052ec22b4189a5264f75a30d30914a4065445f3f1229cce9c` |

## 3. 配置驱动入口

```text
python -m uqra.adaptive.run --config examples/configs/adaptive_reduced_smoke.json
python -m uqra.adaptive.run --config examples/configs/adaptive_reduced_full.json
```

该入口只接受：

- `purpose: software_benchmark`；
- `scale: reduced`；
- `runner.kind: deterministic_benchmark`；
- 冻结的二维 Hermite 软件 benchmark；
- 已声明的 converged、max-order、overfit fallback 和 runtime failure 场景。

它拒绝 `paper_production`，不承担论文正式规模实验。

## 4. 版本化契约

| 契约 | 文件 | SHA-256 |
| --- | --- | --- |
| runner config v1 | `schemas/adaptive-runner-config.schema.json` | `0cf0cab928ca9e6de75d123aafef18a2cbf54711a202e3e606a09c451a2617c5` |
| runner manifest v1 | `schemas/adaptive-runner-manifest.schema.json` | `f7f26e3f65a59ff141e0a66f1b17b78b73f65a7148ebb03264902672a1ef88ae` |
| trace row v1 | `schemas/adaptive-trace.schema.json` | `a0ff2757181a8fa07cc6762ecc43cc122dfd4b045fce76aec0d4f2fd149c09c0` |

示例配置：

| 文件 | SHA-256 |
| --- | --- |
| `examples/configs/adaptive_reduced_smoke.json` | `c907b688de35399e9f4034ea3cdf9d33fd31430a927bf385d6d0742c5086596c` |
| `examples/configs/adaptive_reduced_full.json` | `afbeb3f4788fb6d5b9c0543cae1dc2c344a6d53847f8615f4bfc7910c26dd486` |

配置读取器执行与 JSON Schema 相同的严格字段、枚举、用途和规模检查；生成后再次校验
config hash、scenario、trace row 和 stable manifest hash。

## 5. 验证结果

2026-08-05 实测：

| 环境 | 结果 |
| --- | --- |
| Python 3.11.15，锁定依赖 | `42 passed` |
| Python 3.12.13，锁定依赖 | `42 passed` |
| 全新克隆 Python 3.11.15 | 安装成功，`42 passed` |
| 全新克隆 smoke 配置 | manifest 生成并通过运行时契约校验 |
| 全新克隆 full 配置 | manifest 生成并通过运行时契约校验 |

全新克隆位于临时验收目录 `C:\tmp\uqra-m1-clean-20260805`，从提交
`c6fea079fa742fca160cdc2b90ab10bcdda364c3` 创建；独立 venv 按锁文件安装，
未复用开发工作区的可编辑安装，运行后 Git 工作树保持干净。

全新克隆生成的稳定摘要：

| 配置 | config hash | stable manifest hash |
| --- | --- | --- |
| smoke | `fc4cd5b8fb37f49ba0306c6d1314cd0d087852ad58521823351ea1f4abbc22f0` | `6bb6603ddfe78ff2fafc8cceeb2c02354b7700dea619c220fe2dd270d8705b53` |
| full | `1174509ba03b4f2d9faae43e9a7fb93259cf7ec1885280bee0d834b6047c1258` | `1d0701414cb8b5956ed99820927bb3cb1a17ae546fe4496458f8f5985bc1ac36` |

## 6. v0.1.0 冻结 publication 证据

- 源码树 hash：`78eb3cd3f436cd3c22c12ac02b821ce4a4ddf2f8187e9177454a13da33fdac9a`；
- stable publication manifest hash：`1061e3494f139195a0656d85b72efe9b7352ae22d4ea3b398b167823491ae415`；
- 冻结文件：`specs/ADAPTIVE_PCE_PHASE11_FROZEN_MANIFEST.json`；
- required check：`Adaptive compatibility gate`。

## 7. 允许声明

- modern UQRA-compatible 核心具有稳定候选身份和唯一模型调用保护；
- Hermite、LARS/CV/OLS、RRQR、D/S-optimal、DoI 映射和状态机具有定向回归证据；
- 缩减规模确定性软件 benchmark 可在锁定环境重复运行；
- 配置、runner manifest 和 trace row 具有版本化、机器可读契约；
- Python 3.11/3.12 兼容性套件通过。

## 8. 禁止声明与已知限制

- 不得声明 canonical 历史 FourBranch 已完整重放；
- 不得声明博士论文最终失效概率、全部表格或图件已经复现；
- 不得将缩减规模软件 benchmark 标记为 `paper_production`；
- 不得使用 simplified portable runner 生成新的正式论文结果；
- 历史候选池、测试集、逐轮输出和 RNG/CV 状态仍不可用；
- M1 的 schema 和配置入口属于 `v0.1.0` 后补交付接口，下一次正式发布应使用新版本号。

## 9. 下游缺陷反馈

论文项目反馈问题时至少提供：UQRA commit/tag、完整配置、输入身份和哈希、运行命令、
环境版本、最小候选池、trace、实际/预期状态以及 `stop_reason`。核心修复必须回到 UQRA
完成裁决、测试和新版本发布；论文仓库不得复制核心算法形成分叉实现。
