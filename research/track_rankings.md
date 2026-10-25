# Track rankings for Recall (Cal Hacks 13.0)

The machine-readable version is `research/track_rankings.csv`. Track facts and sources are in `research/sponsor_tracks.md`.

**A high score is not a probability of winning.** The score measures how well a track fits Recall and the next 4 hours:
fit, judging edge, differentiation, feasibility, overlap with other tracks and cost. It says nothing about the size or
strength of the field. The p_finish and p_meets columns are the scorers' estimates that we finish the work and meet the
track's stated requirements. They are not win odds either.

## Method

- **Inputs.** Three independent scorers (S0, S1, S2) scored every track in `research.json`. The lead aggregated them in `rank.json`.
- **Dimensions.** Each scorer gave 0-10 on six dimensions. Weights:
  - product_fit 30
  - judging_advantage 25
  - differentiation 15
  - feasibility 15
  - multi_track 10
  - cost_efficiency 5
- **Score.** Each scorer's weighted sum is scaled to 0-100. The track score is the mean over scorers. p_finish_on_time and p_meets_requirements are scorer means.
- **needs_pivot.** "yes" if every scorer said a product pivot is needed, "split (k/3)" if only some did.
- **Duplicates merged** (different names for the same track). After merging, every track has 3 scores. The merged tracks:
  - Visa
  - Elastic
  - Apify Best Actor
  - SpaceXSI
  - Hyphenate
  - MLH Gen AI (three spellings)

  These rows were recomputed from the raw scores, so they differ from the split rows in `rank.json`. For example, Elastic is 40.5 here, versus 41.0 and 40.2 there. All other rows match `rank.json` exactly.
- **Gen AI.** [MLH] Best Use of Gen AI is on Hive only. **It is unverified on Devpost.**
- **Ties** share a rank (First, Second and Third Overall).

## Ranked table

| Rank | Track | Score | Fit | Judging | Diff | Feas | Multi | Cost | p_finish | p_meets | Pivot | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cal Hacks Best People-First Social Impact Hack | 88.2 | 10.0 | 7.7 | 7.0 | 9.7 | 9.0 | 10.0 | 0.99 | 0.98 | no | SECONDARY (no-build) |
| 2 | Cal Hacks Most Technically Complex Hack | 85.2 | 9.0 | 7.7 | 7.7 | 9.0 | 9.0 | 10.0 | 0.95 | 0.98 | no | SECONDARY (no-build) |
| 3 | [MLH] Best Use of Tiger Data | 82.2 | 9.0 | 7.3 | 7.3 | 9.0 | 7.3 | 10.0 | 0.90 | 0.92 | no | SECONDARY (build) |
| 4 | Cal Hacks First Overall | 81.8 | 8.7 | 6.7 | 7.7 | 8.7 | 9.7 | 10.0 | 0.97 | 0.98 | no | SECONDARY (no-build) |
| 4 | Cal Hacks Second Overall | 81.8 | 8.7 | 6.7 | 7.7 | 8.7 | 9.7 | 10.0 | 0.97 | 0.98 | no | SECONDARY (no-build) |
| 4 | Cal Hacks Third Overall | 81.8 | 8.7 | 6.7 | 7.7 | 8.7 | 9.7 | 10.0 | 0.97 | 0.98 | no | SECONDARY (no-build) |
| 7 | Cal Hacks Most Creative Hack | 80.8 | 8.0 | 6.7 | 8.0 | 9.7 | 8.7 | 10.0 | 0.99 | 0.98 | no | SECONDARY (no-build) |
| 8 | Cal Hacks Best Hardware Hack | 80.5 | 8.7 | 7.3 | 7.0 | 8.0 | 8.7 | 10.0 | 0.92 | 0.97 | no | SECONDARY (no-build) |
| 9 | **[Anthropic] Claude for Public Benefit** | 76.8 | 8.7 | 7.7 | 7.0 | 6.3 | 8.0 | 7.3 | 0.70 | 0.72 | no | **PRIMARY (build)** |
| 10 | [Moss] Best Use of Moss | 73.5 | 8.0 | 6.7 | 6.3 | 9.0 | 5.3 | 9.0 | 0.90 | 0.87 | no | SECONDARY (build) |
| 11 | [Cartesia] Best Use of Cartesia | 73.2 | 8.0 | 6.7 | 6.7 | 8.0 | 6.0 | 9.0 | 0.85 | 0.87 | no | SECONDARY (build) |
| 12 | [Microsoft] Built for the Real World / Customer Impact | 72.8 | 7.3 | 5.7 | 6.0 | 10.0 | 7.7 | 10.0 | 0.98 | 0.80 | no | SECONDARY (framing opt-in) |
| 13 | Cal Hacks Hacker's Choice | 68.0 | 6.7 | 5.0 | 6.7 | 9.0 | 7.0 | 10.0 | 0.99 | 0.95 | no | SECONDARY (no-build) |
| 14 | [MLH] Commit Fellowship Prize Category | 66.2 | 6.7 | 5.7 | 5.7 | 9.0 | 5.0 | 10.0 | 0.88 | 0.78 | no | SECONDARY (framing opt-in, if the team wants it) |
| 15 | [Meta] Bringing People Closer Together with AI | 64.2 | 6.7 | 4.7 | 5.7 | 8.7 | 6.0 | 10.0 | 0.95 | 0.60 | no | SECONDARY (framing opt-in) |
| 16 | [MLH] Best Use of Gen AI (Hive only; unverified on Devpost) | 57.5 | 6.0 | 4.3 | 4.3 | 7.3 | 6.3 | 9.7 | 0.72 | 0.63 | no | STRETCH (if selectable) |
| 17 | [Fetch.ai] Best Use of Fetch.ai | 53.5 | 6.0 | 5.3 | 5.7 | 3.0 | 4.7 | 9.0 | 0.35 | 0.37 | no | REJECT |
| 18 | [MLH] Best Use of Gemini API | 51.3 | 5.0 | 4.3 | 3.0 | 8.0 | 4.3 | 9.3 | 0.82 | 0.85 | no | STRETCH (Claude fallback only) |
| 19 | [Bitdeer AI] Best Use of Bitdeer AI Model Studio | 51.2 | 5.3 | 5.3 | 4.0 | 6.0 | 4.0 | 5.7 | 0.68 | 0.70 | no | REJECT |
| 20 | [Spacetime] Best Use of Spacetime | 43.3 | 4.7 | 4.3 | 5.3 | 2.0 | 3.0 | 9.0 | 0.22 | 0.28 | no | REJECT |
| 21 | [MLH] Best Domain Name from GoDaddy Registry | 41.8 | 3.0 | 3.3 | 2.3 | 9.3 | 2.0 | 10.0 | 0.93 | 0.85 | no | STRETCH |
| 22 | [Elastic] Agent Builder on a Serverless Instance | 40.5 | 4.0 | 4.0 | 3.7 | 4.0 | 3.0 | 8.0 | 0.48 | 0.48 | no | REJECT |
| 23 | [Creao AI] Best AI-First Disruptor | 36.0 | 3.0 | 3.0 | 3.0 | 5.3 | 2.7 | 8.7 | 0.80 | 0.30 | no | REJECT |
| 24 | [MLH] Best Use of Vultr | 33.5 | 3.0 | 3.0 | 2.7 | 5.0 | 2.0 | 7.0 | 0.60 | 0.60 | no | REJECT |
| 25 | [Convex] Best use of Convex | 31.3 | 3.0 | 3.3 | 3.3 | 2.0 | 2.0 | 8.0 | 0.23 | 0.28 | no | REJECT |
| 26 | Cal Hacks Best Planet-First Social Impact Hack | 29.3 | 1.3 | 1.0 | 1.3 | 9.7 | 1.3 | 10.0 | 1.00 | 0.50 | split (2/3) | REJECT (do not tick) |
| 27 | [Armadin] Best Security Hack | 27.8 | 2.0 | 2.0 | 2.0 | 4.7 | 2.0 | 9.7 | 0.67 | 0.33 | yes | REJECT |
| 27 | [Mana] Best Use of Mana | 27.8 | 2.3 | 2.7 | 2.3 | 3.3 | 2.0 | 7.3 | 0.47 | 0.28 | no | REJECT |
| 29 | [MLH] Best Use of Snowflake API | 27.7 | 2.0 | 2.7 | 2.0 | 4.0 | 2.0 | 8.0 | 0.50 | 0.50 | no | REJECT |
| 30 | [Creao AI] Best Team Empowerment | 27.5 | 2.0 | 2.0 | 2.0 | 5.0 | 1.7 | 8.7 | 0.80 | 0.20 | split (1/3) | REJECT |
| 31 | Cal Hacks Best Beginner Hack | 26.0 | 1.7 | 1.0 | 1.7 | 6.7 | 1.0 | 10.0 | 1.00 | 0.07 | no | REJECT (do not tick) |
| 32 | [Figma] Best Figma Workflow Hack | 25.7 | 2.3 | 2.0 | 2.0 | 3.0 | 2.0 | 8.3 | 0.38 | 0.30 | no | REJECT |
| 33 | [Notability] Best Use of Notability | 17.8 | 1.0 | 1.0 | 1.0 | 3.3 | 1.0 | 9.7 | 0.67 | 0.12 | no | REJECT |
| 34 | [Factory] Best Use of the Factory Droid SDK | 17.5 | 1.0 | 1.3 | 1.0 | 3.7 | 1.0 | 6.3 | 0.50 | 0.20 | yes | REJECT |
| 35 | [Creao AI] Best One-Person Company | 17.3 | 1.0 | 1.0 | 1.0 | 3.3 | 1.0 | 8.7 | 0.80 | 0.07 | yes | REJECT |
| 36 | [Apify] Best Use of Apify | 16.8 | 1.0 | 1.0 | 1.0 | 3.3 | 1.0 | 7.7 | 0.47 | 0.17 | yes | REJECT |
| 36 | [Apify] Best Actor sponsored by Apify | 16.8 | 1.0 | 1.0 | 1.0 | 3.3 | 1.0 | 7.7 | 0.47 | 0.17 | yes | REJECT |
| 36 | [Capital One] Best Use of Nessie | 16.8 | 1.0 | 1.0 | 1.3 | 2.7 | 1.0 | 8.7 | 0.37 | 0.17 | yes | REJECT |
| 39 | [MLH] Best Use of Solana | 16.7 | 1.0 | 1.0 | 1.0 | 3.0 | 1.0 | 8.3 | 0.40 | 0.30 | yes | REJECT |
| 40 | [CRS] Best Use of CRS API | 16.3 | 1.0 | 1.0 | 1.3 | 2.3 | 1.0 | 8.7 | 0.33 | 0.12 | yes | REJECT |
| 41 | [Atlassian] Best Agent / Best Security Solution for Agents | 15.3 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 8.7 | 0.27 | 0.05 | yes | REJECT |
| 41 | [Coinbase] Best Use of x402 by Coinbase | 15.3 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 8.7 | 0.27 | 0.08 | yes | REJECT |
| 43 | [Bright Data] Best Agent/App Using Public Web Data | 14.7 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 7.3 | 0.27 | 0.05 | yes | REJECT |
| 44 | [Visa] Best use of Reimagining Shopping | 13.5 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 5.0 | 0.20 | 0.08 | yes | REJECT |
| 45 | Cal Hacks Best Solo Hack | 13.2 | 0.3 | 0.3 | 0.3 | 3.7 | 0.3 | 10.0 | 1.00 | 0.00 | no | REJECT (do not tick) |
| 46 | [Hyphenate] The Hyphenated Enterprise: Autonomous Systems for Finance | 12.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5.0 | 0.10 | 0.05 | yes | REJECT |
| 47 | [SpaceXSI] Make it Legendary with SpaceXSI | 11.5 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 9.3 | 0.10 | 0.02 | yes | REJECT |

## Shortlist

**PRIMARY: [Anthropic] Claude for Public Benefit.** This is the highest-value new build. It ranks 9th, below the
no-build Cal Hacks categories, but those are entered anyway at no build cost, so it is the best use of new hours.
- The feature: a grounded Claude care-partner assistant over Recall's memory and the Tiger Data history.
- The precondition: an ANTHROPIC_API_KEY or event credits, obtained early (by about 5:15 AM).
- If no key turns up, the same tool layer moves to Gemini, and the team enters [MLH] Gemini instead of Anthropic.

**SECONDARY, builds** (one agent each):
- [MLH] Best Use of Tiger Data: continuous aggregate, "This week" panel, lost-spots query, benchmark.
- [Cartesia] Best Use of Cartesia: spoken answers and directions, with a fallback.
- [Moss] Best Use of Moss: latency visible on the page, optional event search.

**SECONDARY, no build:** these are entered on the existing product plus pitch and demo reliability:
- Cal Hacks People-First Social Impact
- Most Technically Complex
- First/Second/Third Overall
- Most Creative
- Best Hardware
- Hacker's Choice

**SECONDARY, framing-only opt-ins** (write-up text, no code):
- [Microsoft] Built for the Real World / Customer Impact
- [Meta] Bringing People Closer Together with AI
- [MLH] Commit Fellowship, only if the whole team wants the Nov 2-20 live sprint and is in the US, Canada or Mexico. It needs the supplemental form before 9.

**STRETCH:**
- [MLH] GoDaddy Registry domain: about 20 min for a non-engineer.
- [MLH] Best Use of Gen AI: only if it is selectable on Devpost and the Claude feature shipped. Unverified on Devpost.
- [MLH] Gemini API: only as the Claude fallback, never alongside Claude.

## Per-track reasoning (condensed from the three scorers)

**Shortlisted**
- **People-First Social Impact (88.2).** All three scorers call it Recall's best first-party fit: assistive tech for people with dementia and their care partners, local video, spoken answers. The edge comes from a working end-to-end hardware demo. Claude, Cartesia and Tiger all strengthen this pitch directly.
- **Most Technically Complex (85.2).** The stack is unusually deep (LiDAR + ArUco, ray casting into 3D, open-vocab detection, pose, container rules, local STT/VAD, Moss, BLE band, TimescaleDB), but judges must see the live 3D localization and an architecture diagram. Everything must be on main.
- **Tiger Data (82.2).** Already built and tested; it needs merging to main. Recall's memory is honest time-series data. The scorers called it the best value per hour of any sponsor track. Today it lacks the track's own bullets (continuous aggregates, dashboards, shown compression), and the build plan adds them. The prize is small (Stream Deck Mini). The continuous aggregate also feeds the Claude tools.
- **First/Second/Third Overall (81.8).** Strong on all five criteria, but the field is the largest. A win depends on a flawless 3-minute ask-and-point demo. Integrations must strengthen the single story rather than appear as a list of sponsor logos.
- **Most Creative (80.8).** No tags on objects, any object, "probably in the blue bag" inference, and a band that points instead of a map. Contrast it with AirTag/Tile explicitly.
- **Best Hardware (80.5).** The band is central, not decorative. The band itself is simple next to dedicated hardware hacks, so the closed loop (camera, 3D, pose, arrow) has to work live. Spend reliability time here.
- **Claude for Public Benefit (76.8).** Anthropic's four published criteria match Recall almost exactly, except criterion 4: there is zero Claude code today. A tool-grounded agent (no invented locations, text events only, the patient path stays templated, no medical advice) turns "know where it could go wrong" into a strength. Feasibility (6.3) and cost efficiency (7.3) are low only because no key or credits are confirmed. It is the largest prize Recall can credibly win: $5k in credits plus Max, one winner.
- **Moss (73.5).** Real and on-message: on-device semantic search for vague speech is Moss's own pitch. Today it is a thin fallback, and "moss" is missing from Built With. Showing latency and indexing memory events makes it a visible "best use". Criteria are unpublished.
- **Cartesia (73.2).** All laptop speech goes through `voice.speak()`, and the page through `say()`, so the swap is contained. A calm, slow voice is a real dementia-care improvement and audible in the video. TTS is the obvious use, so the edge comes from the care framing. Criteria and prize are unclear ("Meta glasses TBD").
- **Microsoft (72.8).** A theme award with no stated technology. The FDE framing rewards a named customer, deployability and graceful failure. Don't add Azure. It is unknown whether judges expect it.
- **Hacker's Choice (68.0).** The pointing band is a crowd-pleaser. The voting mechanism is unknown, so keep the demo running 1-2 PM.
- **Commit Fellowship (66.2).** No tech is needed, and the founder story is credible. The follow-through obligation (Nov 2-20 live sessions) is real, and the form URL is missing.
- **Meta (64.2).** The theme fits (less friction between the person and the care partner). Judges may expect Llama (unstated). A Llama build would duplicate Claude. The prize is a Round 2 invite with no cash and an unknown date.
- **Gen AI (57.5, Hive only).** A free tick only if it appears on Devpost and Claude ships. Low value, high competition.
- **Gemini (51.3).** The same care-partner idea as Claude, for swag. Running two LLMs for one feature reads as prize-farming. Use it as plan B only.
- **GoDaddy (41.8).** Cosmetic but nearly free. Point the domain at the Devpost or video page, and don't build a landing page.

## Separate assessments (scorer means; ranges where scorers differed)

| Track | p_finish | p_meets | Demo-failure risk (scorers) | Marginal cost | Extra hours | Pivot |
|---|---|---|---|---|---|---|
| Claude for Public Benefit | 0.70 | 0.72 | Medium: venue wifi/API latency (a tool loop takes 2-6 s), no key yet, hallucinated locations. Mitigation: templated local fallback, Claude only on the care-partner side | $0 with event credits; $2-10 if paid | 2-3 h + video | no |
| Tiger Data | 0.90 | 0.92 | Low: the background writer never blocks. The cloud DB may be unreachable, which only hides the panel. Free-tier policy jobs may not run (refresh manually) | $0 | 0.25 merge + 1.5-2 h | no |
| Moss | 0.90 | 0.87 | Low-medium: the 60 session-minutes/month free cap may be used up. On failure it falls back silently to exact match (hiding Moss) | $0 (cents if pay-as-you-go) | 0.25 + 1-1.5 h | no |
| Cartesia | 0.85 | 0.87 | Medium-low to medium: venue network latency. `say` and speechSynthesis are automatic fallbacks; cache the demo phrases | $0 (free tier); cloning would be $5 and is not recommended | 1-2 h | no |
| People-First / Overall / Creative / Tech Complex | 0.95-0.99 | 0.98 | Medium: the live chain (ask, detect, speak, band points) has many parts. Bring a backup clip | $0 | 0-1 h | no |
| Best Hardware | 0.92 | 0.97 | Medium-high: BLE drops, battery, servo calibration in a crowded RF venue | $0 | 0.5-1 h | no |
| Microsoft / Meta (framing) | 0.98 / 0.95 | 0.80 / 0.60 | Low as framing only. Unstated tech expectations (Azure / Llama) | $0 | 0-0.5 h | no |
| Commit Fellowship | 0.88 | 0.78 | None (judged after the event). The risk is the missing form URL | $0 | 0.25-0.5 h | no |
| Hacker's Choice | 0.99 | 0.95 | Medium: the demo must run unattended 1-2 PM | $0 | 0 h | no |
| Gen AI (Hive only) | 0.72 | 0.63 | Inherits Claude's risk | $0 extra | 0 h extra | no |
| Gemini (fallback) | 0.82 | 0.85 | Medium: network, free-tier rate limits | $0 | 0.3 h as a provider swap; 1.5-2 h standalone | no |
| GoDaddy | 0.93 | 0.85 | None | $0 (MLH code) | 0.25-0.5 h | no |

## REJECTION list (do not build, do not tick)

| Track | Score | Reason |
|---|---|---|
| [Fetch.ai] Best Use of Fetch.ai | 53.5 | The biggest cash of the fitting tracks, but the mandatory checklist is long: Agentverse, chat protocol, the whole workflow inside ASI:One, README badges, a 3-5 min second video, and a submission agent that every teammate must join. p_finish is 0.35, and it competes with Claude for the same role. Scorers listed it only as a stretch for an otherwise idle person with a hard kill time. Not in the plan |
| [Bitdeer AI] Model Studio | 51.2 | A clear rubric, but it would be a second or third LLM provider for the same care-partner feature (a bolt-on). Free credits are unverified and it may need a card |
| [Spacetime] Best Use of Spacetime | 43.3 | It must be the core real-time backend. That means rewriting the server and page sync hours before judging; there is no official Python SDK (unverified), and it overlaps Tiger |
| [Elastic] Agent Builder on Serverless | 40.5 | A second event store (duplicates Tiger) plus a second agent (duplicates Claude). Criteria say "Coming soon!" |
| [Creao AI] Best AI-First Disruptor | 36.0 | Requirements unknown; the CREAO platform has no runtime role. Only reconsider if a rep confirms the track is theme-only |
| [MLH] Best Use of Vultr | 33.5 | A cloud relay contradicts the local-first privacy story, adds auth and deploy risk, and makes a weak "best use" |
| [Convex] Best use of Convex | 31.3 | "Built on Convex" means a new TypeScript backend competing with Tiger: a near-pivot of the architecture |
| Cal Hacks Best Planet-First Social Impact | 29.3 | No environmental angle. Ticking it would dilute the people-first framing |
| [Armadin] Best Security Hack | 27.8 | Local-first privacy is a feature, not a security hack |
| [Mana] Best Use of Mana | 27.8 | 30% of the score is meaningful use during the build. Recall was built without Mana, and Mana can't reach the laptop-local service |
| [MLH] Best Use of Snowflake API | 27.7 | Duplicates Tiger (storage) and Claude (LLM). Slow account setup |
| [Creao AI] Best Team Empowerment | 27.5 | A workplace-productivity theme; the "care team" angle is a stretch |
| Cal Hacks Best Beginner Hack | 26.0 | Every member must have attended 0 prior hackathons, and organizers verify. A false claim risks disqualification |
| [Figma] Best Figma Workflow Hack | 25.7 | It rewards an honest idea-to-demo Figma workflow (stage log, Slides, Edu). Recall was built code-first, so this would be retroactive |
| [Notability] Best Use of Notability | 17.8 | The criteria penalize "added after the fact"; Notability wasn't used |
| [Factory] Droid SDK | 17.5 | Recall has no runtime need for a coding agent |
| [Creao AI] Best One-Person Company | 17.3 | Recall is a 4-person team, not a one-person company |
| [Apify] Best Use of Apify | 16.8 | Public web data has no role in private in-home perception |
| [Apify] Best Actor | 16.8 | A cloud Actor is unrelated to the product |
| [Capital One] Best Use of Nessie | 16.8 | Mock banking data is unrelated to finding objects |
| [MLH] Best Use of Solana | 16.7 | A patient's home memory on a public chain contradicts the privacy story |
| [CRS] Best Use of CRS API | 16.3 | The criteria penalize bolt-ons, and credit data is irrelevant |
| [Atlassian] Best Agent / Security for Agents | 15.3 | Requires a Forge app for Jira/Confluence team work |
| [Coinbase] x402 | 15.3 | Agent payments for a vulnerable user are inappropriate and off-product |
| [Bright Data] Public Web Data | 14.7 | Fails "the product would no longer work without web data" by design |
| [Visa] Reimagining Shopping | 13.5 | No shopping flow; "reorder a lost item" would be forced |
| Cal Hacks Best Solo Hack | 13.2 | Ineligible: 4 members |
| [Hyphenate] Autonomous Systems for Finance | 12.0 | A finance product. The large cash ($16k) is a trap |
| [SpaceXSI] Make it Legendary | 11.5 | Requires space data and a project built with Cursor; Recall has neither |

Not tracks, but rejected ideas from the scorers' stretch lists:
- Moss browser type-ahead: needs a custom authenticator.
- A local Llama fallback: a third LLM path.
- Cartesia voice cloning: a $5 purchase that needs explicit approval.
