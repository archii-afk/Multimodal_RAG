# Repository collaboration guide

These instructions apply to the entire repository and are the shared operating rules for Codex, Claude Code, and human contributors.

## Start every task

1. Read `README.md`, `docs/PROJECT_CONTEXT.md`, `docs/DECISIONS.md`, and `docs/HANDOFF.md`.
2. Run `git status --short --branch` before editing.
3. Treat `multimodal_data_management_hackathon.pdf` as the authoritative problem statement. Do not invent requirements that are not in the PDF or explicitly agreed with the user.
4. If the requested change is ambiguous in a way that materially affects the design, record the question in `docs/HANDOFF.md` and ask before committing to an architecture.

## Working conventions

- Prefer small, reviewable changes with clear names and minimal unrelated churn.
- Preserve user and other-agent changes. Never reset, discard, or overwrite work you did not create.
- Keep secrets and local credentials out of Git. Use environment variables and document required names in `.env.example` if configuration is introduced.
- Never use `git add -A` or `git add .`. Stage explicit paths, and read `git status` for unexpected files before every commit. (A stray key file was once swept into history this way and had to be purged.)
- Add or update tests with implementation changes. Run the narrowest relevant checks first, then the full available suite before handoff.
- Update documentation when behavior, setup, architecture, or a settled decision changes.
- Avoid adding dependencies unless they materially simplify the solution; document why each non-obvious dependency is needed.

## Codex and Claude Code coordination

- Do not have two agents edit the same working tree at the same time.
- All commits must use the repository owner's configured Git name and email. Do not override the author or committer identity, and do not add Codex, Claude, AI, or agent `Co-authored-by` trailers.
- For parallel work, give each agent a separate branch and Git worktree, for example:

  ```sh
  git worktree add ../Multimodal_RAG-codex -b codex/<task>
  git worktree add ../Multimodal_RAG-claude -b claude/<task>
  ```

- Keep one concern per branch. Rebase or merge the latest `main` before integration and resolve conflicts deliberately.
- Before handing work to another agent, update `docs/HANDOFF.md` with the branch, completed work, checks run, open questions, and next action.
- Record durable architecture or product decisions in `docs/DECISIONS.md`; do not rely on chat history as project memory.
- The integrating agent reviews diffs and test results before merging. Agents must not silently commit or merge another agent's unreviewed changes.

## Project structure

The codebase has not been scaffolded yet. Add runtime-specific directories only after the architecture is agreed. When structure is introduced, document it in `README.md` and keep generated data, model artifacts, indexes, and credentials untracked.
