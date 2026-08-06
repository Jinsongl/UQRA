# UQRA M3 包装与 Windows/Python 3.12 验收

状态：完成  
验收日期：2026-08-06  
冻结源码提交：`dd2c50319736db56aef55d3e3e50f8a0ad4dde88`
远端 required run：[`31062594561`](https://github.com/Jinsongl/UQRA/actions/runs/31062594561)

## 验收范围

- 使用 Windows CPython 3.12.13 构建 wheel 和 sdist；
- 核对两种发行包均包含五个已发布 JSON schema；
- 在仓库外创建两个独立 venv，分别安装 wheel 和 sdist；
- 验证 `import uqra`、distribution、`uqra.__version__`、CLI 和 manifest 版本一致；
- 用 Draft 2020-12 校验 config、manifest v2 和 trace；
- 重新读取 manifest v2 evidence package 的输入、trace、结果和摘要并核对字节数与 SHA-256；
- 将 UQRA 源码的 SyntaxWarning/DeprecationWarning 作为错误执行全包编译；
- 运行完整 compatibility suite，结果为 `61 passed`。

## 冻结发行包

| 类型 | 文件 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| wheel | `uqra-0.2.0-py3-none-any.whl` | 196544 | `374f8e19d1b91cce0ad9768ac4f6a8bdbe2248556183b9d7daf7f67e69fc3a43` |
| sdist | `uqra-0.2.0.tar.gz` | 170288 | `9366e9fd2656845ae67962eccce8fd6084c1f17524ae71d65edb66c1155d0fd1` |

结构化证据：

- [`UQRA_M3_DISTRIBUTION_MANIFEST.json`](UQRA_M3_DISTRIBUTION_MANIFEST.json)
- [`UQRA_M3_CLEAN_INSTALL_EVIDENCE.json`](UQRA_M3_CLEAN_INSTALL_EVIDENCE.json)

## 声明边界

本证据只证明 UQRA 软件包装、安装、schema、runner 和 manifest 契约在 Windows/Python
3.12 正式环境下通过。它不构成 Ubuntu 软件兼容性、历史 replay、论文生产结果或
scientific reproduction 声明。Python 3.11 仍允许安装，但不属于持续验证和 M3 完成门。
