# UQRA M5 P0 Acceptance

日期：2026-08-07  
范围：`BUILD-01`、`SEC-02`、`PATH-01`

本轮只修改发布、供应链和测试工具，不修改 `uqra/adaptive/` 或数学契约。三个 reduced
benchmark 的声明边界不变。`specs/releases/UQRA_V0.3.0_*` 冻结证据未被重写。

## BUILD-01

`tools/packaging/build_reproducible.ps1` 固定以下非确定性来源：

- `SOURCE_DATE_EPOCH`：由调用者绑定到源提交时间；
- `PYTHONHASHSEED=0`；
- wheel ZIP 成员时间：由 wheel 0.47.0 遵循 `SOURCE_DATE_EPOCH`；
- sdist tar 成员及 PAX 时间：统一为 source epoch；
- sdist gzip header 时间和文件名：统一由 `normalize_sdist.py` 写入。

`run_reproducible_build.ps1` 在两个独立空目录构建，
`verify_reproducible_build.py` 比较文件名、大小和 SHA-256，并记录 source commit 与 epoch。
输出目录已存在时明确拒绝，重复执行必须使用新的空目录，以保留失败现场且避免静默删除证据。
Windows/Python 3.12 实测：

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `uqra-0.3.0-py3-none-any.whl` | 196245 | `6c7b19f18b122c6c07a2b2895b7bd73757c0d9026453d06b4878c9268a8d7dca` |
| `uqra-0.3.0.tar.gz` | 165711 | `3c4e8f6b9be3a573460266104bf163b348b572596cb9b2080c92d68d2d926d26` |

这些是 M5 工作树的验证构建，不是 v0.3.0 Release 附件，不能替换冻结发行包摘要。

## SEC-02

`tools/security/git_blob_sha256.py` 通过 `git cat-file blob REVISION:path` 读取原始 blob 字节，
不读取可能受 CRLF/LF 转换影响的工作树文件。工具可直接绑定安全审计 JSON 中的
`input.path` 和 `input.git_blob_sha256`，路径或摘要不一致时失败。
未提供期望摘要或审计证据时同样拒绝运行；输出同时记录解析后的 immutable commit 和
Git blob object ID。

当前锁文件 canonical Git blob SHA-256 为
`51a49cdfc1ea25732789b5a1f5bc474acd95ee976483375c988880cbea4a5f78`。
正确绑定及错误摘要拒绝测试均通过。

## PATH-01

`run_windows_path_regression.ps1` 在三个独立路径中复制 Git 已跟踪及未忽略源码，并对每个
案例执行版本发现、wheel/sdist 构建、全新 venv、锁定依赖安装、wheel clean-install、CLI、
schema、runner manifest 和 artifact identity smoke：

| Case | Repository path characteristic | Result |
| --- | --- | --- |
| `space` | 含空格 | pass |
| `single_quote` | 含单引号 | pass |
| `non_ascii` | 含非 ASCII 中文字符 | pass |

正式验证环境为 Windows、Python 3.12.13。Python 3.11 和 Ubuntu 的支持声明未扩展。
