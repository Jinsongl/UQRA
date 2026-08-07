# UQRA REL-04 Controlled Release Design

日期：2026-08-07  
状态：实现中；发布 workflow 合并后需由仓库管理员配置并验证人工批准环境

REL-04 只自动化软件发行对象，不改变数学契约，也不管理论文参数、表图或科学复现。
`v0.3.0` tag、Release、附件和四份冻结证据不得用于演练、覆盖或回写。

## 状态机

1. 手工输入无 `v` 前缀的 `X.Y.Z` 版本、完整 master commit SHA 和 `dry_run`；
2. Windows/Python 3.12 job 验证版本源、master 包含关系和 tag 冲突；
3. 对指定 commit 执行双目录字节一致构建及 wheel/sdist clean-install；
4. 将包的文件名、大小和 SHA-256 绑定到 `release-candidate.json`；
5. `dry_run=true` 时到此结束，不获得写权限，不创建 GitHub 对象；
6. `dry_run=false` 时 publish job 等待受保护的 `uqra-release` Environment 人工批准；
7. 批准后才获得 job 级 `contents: write`，再次检查 tag 和 Release 冲突；
8. 创建精确指向批准 commit 的 annotated tag 和 draft Release，上传候选包及清单；
9. 从 draft Release 下载 wheel/sdist，按候选清单复核文件集、大小和 SHA-256；
10. 上传 readback 证据后将 draft Release 发布。

## 失败与重复执行

- 任一已存在 tag 或 Release 都在写入前拒绝，绝不覆盖；
- publish job 只清理本次创建且尚未发布的 draft Release 和 tag；
- 同一版本使用 workflow concurrency 串行化，避免两个批准流程竞争同一 tag；
- 清理前重新确认 Release 仍为 draft；状态不可确认或已发布时拒绝删除 Release/tag；
- 已发布 Release 不进入自动清理路径；
- 下载文件缺失、多余、大小变化或 SHA-256 变化均失败；
- PR 和普通 push workflow 只有 `contents: read`；仅人工批准后的 publish job 有写权限；
- candidate artifact 保留 7 天，不是正式 Release，也不是冻结发布证据。

## 仓库配置门

合并 workflow 后，仓库管理员必须在 GitHub `Settings → Environments` 创建
`uqra-release`，设置 required reviewer，并禁止管理员绕过。未完成这一配置前，不得以
`dry_run=false` 运行 workflow，也不得将 REL-04 标记为完成。publish job 在任何发布写入
前通过 GitHub API 再次验证 required reviewer 数量及管理员绕过设置，配置不合格时失败。

首次正式验证必须使用尚未发布的新版本；禁止以 `v0.3.0` 进行测试。验证记录需包含：
批准人、输入 commit、annotated tag object、Release URL、候选 artifact digest、下载回读
证据以及最终 workflow run URL。
