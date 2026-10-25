# Worktree status

Repo: `/Users/alexanderchin/code/cal-hacks-13`. Paths below are relative to it.
- Every sponsor/* branch starts from the scaffold commit d1de0ad: the route registry plus one placeholder per integration in `recall/web.py`.
- Nothing has been pushed.
- The lead updates statuses as work lands.
- `main` is still at 77be0ec. It needs the integration merge before 9:00 AM PDT, and no commits go in after 9:00.

| Worktree path | Branch | Owner agent | Scope | Status |
|---|---|---|---|---|
| .claude/worktrees/tiger-data | tiger-data | tiger-data agent (earlier) | DB copy of the memory: event hypertable, object state, heartbeat, CLI (`recall/db.py`, `tests/test_db.py`) | done (earlier), commit bc45557 |
| .claude/worktrees/sponsor-ui | sponsor/ui | UI agent | Page redesign from the user's stash 154ceda (the stash itself is untouched) | done, commit 76c0ebd |
| .claude/worktrees/integration | sponsor/integration | lead | Merges tiger-data + sponsor/ui + the scaffold d1de0ad; the sponsor branches merge in here (order: tiger, claude, moss, cartesia), then to main | in progress |
| .claude/worktrees/sponsor-claude | sponsor/claude | Claude agent | `recall/care.py`, `tests/test_care.py`, `[care]` placeholders (`/api/care`) | in progress |
| .claude/worktrees/sponsor-tiger | sponsor/tiger | Tiger agent | Continuous aggregate, week and lost-spots queries in `recall/db.py`, `tools/db_bench.py`, `[week]` placeholders (`/api/week`) | in progress |
| .claude/worktrees/sponsor-cartesia | sponsor/cartesia | Cartesia agent | `recall/tts.py`, `voice.speak()`, `[tts]` route (`/api/tts`), page `say()` | in progress |
| .claude/worktrees/sponsor-moss | sponsor/moss | Moss agent | Latency in `recall/moss_search.py`, `recall/ask.py`, `ask()` response, `[moss]` placeholders; optional event search | in progress |
| .claude/worktrees/sponsor-docs | sponsor/docs | research-docs agent | `research/` docs only (this file, sponsor_tracks.md, track_rankings.md/.csv, implementation_plan.md, budget.md) | in progress |

Other worktrees that exist but are not part of the sponsor work: `.claude/worktrees/no-photos-shown` (branch worktree-no-photos-shown).
