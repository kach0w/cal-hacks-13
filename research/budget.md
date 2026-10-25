# Budget

**Hard cap: $200 for the whole effort.**

**Rule: no purchase, prepaid top-up, plan upgrade or pay-as-you-go switch happens without explicit approval from a human on the team.** Agents never enter payment details. Free tiers, trials with no card, and event credits are allowed.

Scope: this ledger covers paid third-party APIs and services for Recall. **The agents' own model usage (Claude Code sessions doing the research and builds) is not metered in this ledger.**

## Actuals (as of 5:50 AM PDT, Oct 25)

| Item | Spent | Notes |
|---|---|---|
| Research, builds, reviews, QA | $0 | No paid API was called: no keys existed on the build machine. Tests used fakes and local Docker TimescaleDB/Postgres containers |
| Packages installed | $0 | anthropic 2.3.0, cartesia 4.2.0, moss 1.21.0, psycopg 3.3 (open source) |
| **Total spent** | **$0 of $200** | Remaining: $200. Outstanding commitments: none |

## Ledger (as of ~5:00 AM PDT, Oct 25)

| Item | Spent | Notes |
|---|---|---|
| Research (46 tracks, 3 scorers) | $0 | Public web pages and docs only; no paid APIs |
| Builds so far (claude, tiger, cartesia, moss) | $0 | No keys on the build machine. Developed against fakes and a local Docker TimescaleDB |
| **Total spent** | **$0** | |

## Projected live-key verification costs (only if keys are provided)

| Service | Plan | Projection | Cap | Source |
|---|---|---|---|---|
| Claude (Anthropic API) | Event credits if found; otherwise prepaid usage with human approval | A few cents per question (tool-grounded Q&A over text events). Scorers estimated $2-10 for a full day of testing | **$5** | research.json (Anthropic cost) |
| Cartesia | Free plan: 20K credits/month (~27 min of Sonic TTS). Short answers mean hundreds fit | $0. Pro ($5/mo, voice cloning) is NOT planned and would need approval | **$0-5** | https://cartesia.ai/pricing |
| Tiger Cloud | Free service (2 per account) or the $1000, 30-day trial with no card | $0 | **$0** | https://www.tigerdata.com/docs/deploy/tiger-cloud/tiger-cloud-aws/pricing-and-account-management |
| Moss | Developer plan: free, local queries unmetered, 60 session-minutes/month. Pay-as-you-go is $0.05/min, only with approval | $0 | **$0** | https://docs.moss.dev/docs/pricing.md |
| GoDaddy Registry domain (stretch) | MLH free code | $0 | $0 | https://www.tech.study/ |
| **Projected maximum** | | | **$10** | |

## Contingency reserve

$200 - $10 projected maximum = **$190, unallocated**. Spending any of it needs explicit human approval for each item.

Things the reserve must NOT go to:
- a second LLM provider (Gemini or Bitdeer alongside Claude)
- extra cloud stores
- voice cloning, unless the team approves it

If no Anthropic key or credits can be obtained, the plan B is Gemini's free key (cost $0), not a purchase.
