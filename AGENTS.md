# UQRA repository instructions

## Mandatory task closure

任何带稳定任务 ID 的工作（如 `ARCH-01`、`TEST-01`、`PERF-01`、
`OPT-01`）只有满足以下条件后才能声明完成：

1. 交付物、测试和验收证据已经形成。
2. `specs/UQRA_PROJECT_PROGRESS_BOARD.md` 已同步更新：
   - 任务状态；
   - 分支和 PR；
   - 验收测试或 required CI；
   - merge commit 或当前阻塞条件；
   - 下一动作。
3. 如果任务改变路线、优先级、里程碑状态或完成门，同时更新
   `specs/UQRA_PROJECT_DEVELOPMENT_PLAN.md`。
4. 不得只在对话回复中报告完成，而不更新仓库文档。
5. `git diff --check` 必须通过。
6. `specs/releases/UQRA_V0.3.0_*` 冻结证据不得修改。

### Two-stage status

- 任务 PR 尚在审查或 required CI 尚未通过时，只能标记为“🔄 进行中”，并填写
  分支、PR 和候选证据。
- 任务 PR 合并且 required master gate 全绿后，必须通过一个小型纯文档 closure
  PR 写入最终 merge commit 和 master run，随后才能标记为“✅ 完成”。

### Required pre-completion check

完成任务前必须检查：

- 当前分支和工作树；
- 任务在进度看板中的状态；
- PR、merge commit 和 required CI；
- 主计划是否需要同步；
- 下一任务是否已经明确。

路线、优先级、里程碑状态或完成门是否改变，需要 PR 作者明确声明并由 reviewer
复核；自动检查不尝试推断这种语义变化。
