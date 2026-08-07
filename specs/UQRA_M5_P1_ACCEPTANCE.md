# UQRA M5 P1 Acceptance

日期：2026-08-07  
范围：`SEC-01`、`CI-02`  
状态：本地实现完成，等待 GitHub `windows-latest` / Python 3.12 required run

本轮只扩展发布和供应链质量门，不修改 `uqra/adaptive/`、数学契约、v0.3.0 tag、
Release 附件或 `specs/releases/UQRA_V0.3.0_*` 冻结证据。

## SEC-01

正式审计工具固定为 `pip-audit==2.10.1`。`run_security_audit.ps1` 执行以下闭环：

1. 将 revision 解析为 immutable commit；
2. 从该 commit 的 `requirements/compatibility-py312.txt` Git blob 原始字节计算 SHA-256；
3. 对锁文件运行 pip-audit；
4. 将全部依赖分类为 `runtime` 或 `test_build`；
5. 将 workflow 中的 action 分类为 `github_actions`，并记录 mutable tag 风险；
6. 输出包含 commit、工具版本、blob identity、分类、发现和处置状态的 JSON；
7. 对未处置的 critical、high 或无法确定严重度的发现返回失败。

风险接受必须具有结构化 rationale；`accepted_risk` 还必须具有未过期的日期。
当前本地 Windows/Python 3.12.13 审计结果为零已知漏洞、零 blocking finding。
GitHub Actions 当前使用 major tags，证据将其记录为 `mutable_major_tag`，并标明由
Dependabot 持续监控；这不等同于 immutable SHA pinning。

## CI-02

唯一正式 `windows-python312` job 按顺序执行：

1. 安装 Python 3.12 锁定环境及固定审计工具；
2. 运行 packaging、compatibility、schema 和 warning suite；
3. 执行 SEC-02 blob 绑定及 SEC-01 安全审计；
4. 在两个独立目录执行 BUILD-01 字节一致构建；
5. 对 wheel 和 sdist 执行 clean-install；
6. 对空格、单引号和非 ASCII 三类 Windows 路径执行 PATH-01；
7. 上传 BUILD、SEC、clean-install 和 PATH 机器可读证据。

`adaptive-compatibility-gate` Ubuntu job 仍只聚合 Windows job 结果，不构成 Ubuntu
软件兼容性验证。Python 3.11 仍是允许安装但未持续验证。

## 完成条件

本文件在 required run 成功后补充 run URL、commit 及本地/CI 构建摘要对照；在此之前
`SEC-01` 和 `CI-02` 不在进度看板中标记完成。
