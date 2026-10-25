# Worktrees and branches

Status as of 5:45 AM PDT, Oct 25 2026. Nothing has been pushed; every branch below is local only.
All worktrees live under `.claude/worktrees/` in the repo (untracked by git).

| Worktree | Branch | Owner | Scope | Head | Status |
|---|---|---|---|---|---|
| `tiger-data` | `tiger-data` | lead | Database copy of the memory: events, state, heartbeat; writer thread with catch-up; `recall.db` CLI | `bc45557` | Done, merged into integration |
| `sponsor-ui` | `sponsor/ui` | lead | Page redesign, copied from the user's stash `154ceda` (the stash itself is untouched) | `76c0ebd` | Done, merged |
| `integration` | `sponsor/integration` | lead + QA fixer | Merges everything; scaffold `d1de0ad` (route registry + per-integration placeholders); QA fixes; polish | `76f2f18` | **Ready to push** (265 passed, 9 skipped without credentials) |
| `sponsor-docs` | `sponsor/docs` | research-docs agent | `research/` | `6a83ead` | Done, merged |
| `sponsor-tiger` | `sponsor/tiger` | Tiger agent + reviewer | Continuous aggregate `recall_daily`, `lost_spots`, `/week` + This week panel, `tools/db_bench.py` | `47dc44c` | Done, reviewed (ship), merged |
| `sponsor-claude` | `sponsor/claude` | Claude agent + reviewer | `recall/care.py` "Ask about the day", `/care`, `/care/summary` | `2faf434` | Done, reviewed (ship), merged |
| `sponsor-moss` | `sponsor/moss` | Moss agent + reviewer | "Understood ... as keys (Moss, N ms)", events index, `/moss`, `/moss/events` | `1183f37` | Done, reviewed (ship after fixes; fixes committed), merged |
| `sponsor-cartesia` | `sponsor/cartesia` | Cartesia agent + reviewer | `recall/tts.py`, `voice.speak()`, `/tts`, page `say()` | `c8d04dc` | Done, reviewed (ship), merged |
| `no-photos-shown` | `worktree-no-photos-shown` | (pre-existing, not ours) | | `21a0a0d` | Untouched |

Merge order into `sponsor/integration`: tiger-data, sponsor/ui, scaffold, docs, tiger, claude, moss, cartesia. No conflicts.
