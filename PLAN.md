# Recall

**A room that remembers for you.**

Two cameras watch a room and remember where the objects that matter ended up — keys,
wallet, glasses, phone. Ask out loud and hear the answer in a second. Then a band on the
wrist buzzes faster as you get closer, so you don't have to remember the answer to use it.

Built for people living with Alzheimer's or another dementia, for whom *"where did I put
it, and I don't remember putting it anywhere"* is not an annoyance but a symptom.

Architecture: [DESIGN.md](./DESIGN.md). **Scope is a contract — §8 is the list.**

---

## 1. The problem

A woman in her seventies puts her keys down in her kitchen. Ten minutes later she needs
them and has no memory of the last ten minutes. She checks the usual places. She asks her
daughter, who doesn't know either. She asks again twenty minutes later. By the fourth
time, she's certain someone took them.

**This isn't a story about forgetfulness.** Misplacing objects and being unable to retrace
the steps that put them there is one of the recognized early signs of Alzheimer's.
Repetitive questioning is among the most common daily behaviours. And objects that can't
be found are frequently believed stolen — a suspicion that lands on whoever is nearest.

One small event, three costs:

- **Her independence.** She can't leave without the keys and can't solve it alone.
- **Her care partner's day.** Answering "where's my wallet" for the ninth time before
  lunch is the texture of the work, and it stops feeling like care.
- **The relationship.** An accusation nobody can disprove corrodes something the lost
  object never touched.

**Nothing on the market closes this.** Tags and trackers need the person to have attached a
tag, kept it charged, and to operate a phone app at the moment they're least able to.
Cameras record footage, so answering means a family member scrubbing video. And the AI
assistants that *can* answer the question deliver it as **speech or text**, and depend on
the person remembering what was said.

> The name turned out right by accident. Recall is the clinical word for retrieving a
> memory. The product is recall, performed outside the head.

---

## 2. Who it's for

**The person living with dementia — the primary user.** She asks out loud and follows a
buzz. She never installs, charges, tags, aims or operates anything except the band, and
when the band is flat the voice half still answers.

**The care partner — the second user.** Usually not in the house. They get one page: what's
where, what changed today, and the picture behind any claim. That turns "is she okay" into
something answerable without a call that feels like checking up on her.

**Who buys it.** Families first, then memory-care facilities running the same product
across many rooms. The second market is shared workplaces (§7), which is the same engine
with a different reason for the memory to be missing.

---

## 3. What Recall does

Two cameras watch a room and keep a running memory of the objects in it — not footage, but
**what's where, and what changed**.

```
keys      kitchen counter        since 2:40 PM      (observed)
glasses   in the blue bag        since 2:12 PM      (inferred, 0.8)
```

**It answers when the object is out of sight.** Finding something the camera can see is
table stakes. Things vanish into bags, drawers and coat pockets precisely when nobody was
watching. When an object disappears, Recall looks at the last picture it was in and records
what it went into:

```
glasses   last seen entering blue bag @ 2:12 PM
          → inferred: in the blue bag (0.8)
          → last directly observed: counter @ 1:55 PM
```

If the bag moves, the glasses move with it. Answers always separate **observed** from
**inferred**, with confidence, so the system never confidently lies.

Infants develop object permanence. Dementia erodes it. Recall implements it in software.

---

## 4. The band

**The problem with an answer.** "Your glasses are on the kitchen counter" only helps if you
still have it when you reach the kitchen. Thirty seconds is long enough to lose it, and
then the asking starts over. A spoken answer has to be *remembered* to be followed.

**A buzz doesn't.** The band is continuous — stronger and faster as you close in, weaker as
you drift. It isn't information to hold onto; it's a signal that's still there when you
look up having forgotten what you were doing.

**The research backs it from an unexpected direction.** A systematic review of haptic
feedback for blind and low-vision users finds haptic cues impose *less cognitive load* than
visual or auditory notification, with vibrotactile localization around 90%. That's a
**cognitive** result published about a **sensory** disability — nobody has aimed it at the
population defined by cognitive load. Haptic object-finding exists, for people who can't
see. Haptics for dementia exists, for outdoor wayfinding. Driving haptics from a room's
memory, for someone who can't remember, is the open seam.

**The band is deliberately dumb — a buzzer with a battery.** No camera, no sensors, no
localization. The room sees the surface and sees the wrist; the backend sends one number
over Bluetooth; the band turns it into a pulse rate. Three decisions keep it cheap:

- **"Getting warmer," not "turn left."** Directions need to know which way the body is
  facing, and facing is where all the cost and all the error live. A confidently wrong
  instruction given to someone who can't check it is the one failure this can't survive.
- **The band wears a coloured marker, and the camera tracks it.** That gives wrist-level
  precision instead of feet-on-the-floor, where the gradient would go flat exactly in the
  last half metre. It also makes the wearer self-identifying in a crowded room.
- **Bluetooth from the laptop, never venue wifi.** Captive portals break microcontroller
  joins, and venue networking is the top demo-ending risk.

**It needs no calibration.** Warmer/colder needs a distance signal that's *monotonic*, not
*metric* — as the hand closes on a fixed target, the pixel gap shrinks every time. Closeness
comes straight from pixel distance. The band adds no new infrastructure at all.

**The pulse also carries certainty.** An inferred answer can never buzz as hard as an
observed one. The system's promise is that it never confidently lies, and the band honours
that in the channel the person is actually using ([DESIGN.md](./DESIGN.md) §3.2).

### The honest limits

**Guidance works only in a camera's view.** Voice answers from anywhere; the band guides
where a camera can see both the wrist and the target. The real interaction is *voice gets
you to the room, the band finds it in the room* — which is how the product scales, one
camera per room.

**It has to be worn and charged, and that's a staging question.** The band serves
mild-to-moderate stages; the voice half serves later ones. One band on a bedside dock
replaces a tag on every object plus an app driven at the worst moment — one habit instead
of N — and when the band is flat the product still answers.

### Hardware

| Rung | What | Status |
|---|---|---|
| 1 | Microcontroller + coin vibration motor + battery on a wrist strap, BLE, marker on the strap | **The thing we want.** ~$10, a few hours |
| 2 | A phone in a pocket, vibrating off the same signal | **Rehearsed fallback.** Zero hardware risk |
| 3 | On-screen proximity meter that fills as the wrist approaches | **Build this first.** Proves the claim, and debugs rungs 1 and 2 |

The band receives one byte, so rungs 2 and 3 are genuine drop-ins. Falling to rung 3 loses
the magic but not the argument.

---

## 5. Why this product, for this person

**It answers with evidence, not confidence.** A system that guesses wrong at someone who
can't check it is worse than none. Every answer points at the frame that produced it, and
inferred answers say so.

**It defuses the accusation.** *"It's on the counter — here's the picture of it being set
down at 2:40."* That ends an argument nobody can win, and it's the benefit a family feels
first.

**It never tires of the question.** A query is a lookup: free, instant, identical the ninth
time as the first, with no sigh in front of it. The cost argument and the dignity argument
are the same argument here.

**The person doesn't operate it.** Voice in, voice out, nothing to attach or install. Voice
is the primary interface, not a garnish.

**And the help doesn't have to be remembered.** Every other product in this space delivers
information and then depends on retention. The band is the part that works *after* the
answer is forgotten — the failure mode the condition guarantees.

---

## 6. The ethics, stated before anyone asks

- **The camera's subject is a surface, not a person.** Objects are enrolled deliberately, a
  handful at a time, by the household.
- **No faces, no identification, no person-tracking.** The system has no concept of *who*
  moved something. It records that the keys moved.
- **Consent belongs to the person living there**, not only the family — and the
  uncomfortable part said out loud: capacity to consent erodes over exactly the timeline
  this product serves. There is no clean answer. Enrolling few objects, watching surfaces,
  and keeping her the primary user is how it stays a tool for independence rather than
  something installed *at* her.
- **Not a medical device.** Diagnoses nothing, stages nothing, predicts nothing, promises
  no safety outcome. We deliberately do **not** track medication — "did she take her pills"
  is a safety-critical claim we can't stand behind at this accuracy.
- **No pilot, no clinician.** Say it plainly. The honest version survives a follow-up
  question; an implied pilot doesn't.

---

## 7. The demo

**Setup.** Two cameras on a plain surface read as a kitchen counter. Five objects every
judge recognises and has personally lost: **keys, wallet, glasses, phone, remote**, plus a
bag. Zones generated at onboarding. One band.

| Time | Beat |
|---|---|
| 0:00–0:15 | **Frame it.** "This is a kitchen counter, and these are the five things people with dementia lose and look for all day." |
| 0:15–0:40 | **Out of sight.** Judge puts the glasses in the bag while you look away. Ask out loud: *"Where are my glasses?"* → *"In the blue bag — inferred, last seen 20 seconds ago."* Show the still. |
| 0:40–1:05 | **The band, on the judge's wrist.** They hunt; it pulses faster as they close in. Let them find it by feel, without being told where it is. "She doesn't have to remember what it said — she just follows it." |
| 1:05–1:15 | **Ask again. Twice.** Same question, same answer, instantly. "That's the ninth time before lunch, and nobody sighed." |
| 1:15–1:25 | **The care partner's page.** What moved today, with the picture behind each change. |
| 1:25–1:30 | **Generalize and close.** §9, one sentence. |

**The band beat is the demo — protect it above everything.** A judge finding a hidden
object by feel is the most memorable thing on this table and the one beat no slide conveys.
Rehearse strapping it to *someone else's* wrist in three seconds, and confirm the pulse is
unmistakable through a sleeve in a loud room.

**Back pocket, for a technical judge:** the system *declining* to commit an uncertain
observation, and a hand passing over the counter producing no phantom move. Twenty seconds,
and it keeps the technical track in play without changing the lead.

**If something breaks:** band dead or BLE drops → phone in a pocket, then the on-screen
meter. Voice fails in the noise → typed query, one click away. Inference misfires → say
what it did and why it was unsure; a system that admits uncertainty *is* the pitch.

---

## 8. Scope — the contract

The failure mode isn't running out of time. It's six things half-working instead of one
thing that's flawless, when reliability *is* the demo.

### Ship, in this order — nothing starts before the thing above it works

| # | Thing | Why it survived the cut |
|---|---|---|
| 1 | Camera loops, detection + identity, enroll 5 objects + the bag | No product without it |
| 2 | Onboarding zones → "on the counter" | The human-readable half of every answer |
| 3 | Containment guess on disappearance, observed/inferred + confidence | **The wow moment**, and the most fragile code in the project |
| 4 | Append-only log → what's where / what changed today | Cheap, and the care-partner page needs it |
| 5 | Evidence stills | Verifiability is a safety property for this user |
| 6 | Voice in and out, templated answers | The primary user's only interface |
| 7 | Wrist position from the strap's marker → on-screen proximity meter | Rung 3. No calibration, so it can start on minute one |
| 8 | Marker colour verified under demo lighting | The one part lighting can break. Two minutes early, fatal late |
| 9 | Target resolver — a question becomes a *visible* thing to home on, plus a ceiling | The glasses are in the bag, so the band homes on the bag. Without it the core beat has nothing to point at |
| 10 | The band: BLE, pulse = closeness × ceiling, double-pulse on found | The novelty. The found-pulse is also how guidance *ends* |
| 11 | One care-partner page, polling, no interactivity | Second user, cheapest possible form |

### Cut — not deferred, cut

| Cut | Why it's safe to lose |
|---|---|
| **LLM query layer** | Templated answers cover four question shapes with less latency and fewer failure modes at a loud table |
| **Evidence clips** → stills | ~90% of the value for a fraction of the work |
| **3D room model** | Nothing in the demo consumes it |
| **Time-series database** | Flat append-only JSON |
| **Metric positions and all calibration** | The band works in pixel space. Optional polish, and there won't be spare time |
| **Diff views, history scrubbing, revert flows** | Never on the demo path |
| **Alerts, rules, hooks** | Zero value at ninety seconds |
| **Body-pose estimation, turn-by-turn directions** | The marker and warmer/colder delete the need |

**Two sponsor integrations are deliberately traded for reliability.** A flawless ninety
seconds wins a main track; a half-working integration wins nothing. Sponsor track: **voice
only.** Don't quietly re-add the others at 1 AM.

**Stop building by late afternoon on the final day.** Then re-enroll every object in demo
lighting at the demo camera positions and rehearse the sequence three times.

---

## 9. Where this goes next

Underneath the care product is a general one: **a room that knows what's in it, what moved,
and when.** Nothing about that is specific to dementia.

1. **Memory-care facilities** — the same product, operated by staff across many rooms.
2. **Shared homes** — two reasons the memory is missing: nobody remembers, *and* the person
   who moved it isn't the person looking.
3. **Shared workplaces** — offices with loaner hardware, lab benches with instruments
   signed out by nobody. Purely the second reason, and a straightforward paying buyer.
4. **Stockrooms, tool cribs, warehouses** — same primitives, bigger rooms. The band gets
   *more* valuable as the space grows: a picker guided to one bin out of a thousand is the
   same dumb buzzer and the same camera doing the thinking.

One sentence for the close: **a space has no memory, and the two ways that hurts are that
you can't remember, or that someone else did — and the band is how the room tells you
instead of waiting to be asked.**

---

## 10. Why it wins

- **The index matches the question.** Everything else indexes footage by time. "Where are
  my glasses" becomes a lookup, not a search.
- **It answers what footage search structurally cannot** — including from absence. "Has
  anything moved today?" needs the lack of a change, not a ranked clip list.
- **It's verifiable, so it's trustworthy.** A record with the frame attached, not a model's
  recollection. Non-negotiable for a user who can't check it.
- **It closes the loop instead of stopping at an answer.** The camera sees, the memory
  holds, **the band acts**, the picture proves. Everything in this category stops one step
  early and leaves the acting to someone who must hold the answer in their head.
- **Hardware and software that justify each other.** The camera is why the band can be a
  $10 buzzer instead of a sensor platform; the band is why the camera's knowledge is
  actionable instead of informational. Neither is a bolt-on.
- **We say which part isn't ours.** Persistent object memory is established robotics
  research — scene graphs and spatial memory layers do it better, for robots. Ours is a
  *delivery* claim: pointing it at a person who can't hold an answer, and closing the loop
  with a signal they don't have to remember.

---

## 11. The hard questions

**Never claim object memory is new.** It's a Google demo, a shipping Google product, a
peer-reviewed study, and several hackathon projects. Claim the three seams instead:
everything out there **(1)** needs the person to wear, aim, tag or charge something,
**(2)** delivers an answer that must be *remembered*, or **(3)** can't be checked. We close
all three.

**"Have you talked to anyone with dementia, or a caregiver?"** The question that can sink
this. If the call happened, lead with what they said. If not, say no plainly and say what
you'd do first.

**"This is Project Astra."** Astra answers; it doesn't guide — and it needs a camera the
person is wearing or holding at the moment the object goes down. Ours is on the wall and
ends in a buzz on the wrist.

**"Google already ships this."** They ship natural-language *video search* and live scene
description — time- and event-indexed. Ask it where the glasses are when they're inside a
bag and no clip exists. No object memory to query, and no way to answer from absence.

**"What stops Google building this next quarter?"** Technically nothing. Practically, the
distance from security-and-monitoring to a care product isn't a model upgrade: it's a
haptic wearable, a staged clinical story, a consent posture for declining capacity, and a
care-partner relationship. **Our defensibility is the vertical and the actuator, not the
perception.**

**"Academic systems did this in a paper, with a user study."** Correct, and we cite them.
They also need the camera worn and the object placed consciously in frame, store no imagery
so nothing is verifiable, and deliver by audio only — the modality their own participants'
slurred speech struggled with, and the one our user can't retain.

**"A team built this on smart glasses."** The closest prior work, and the difference is our
most important decision: **their sensor is worn by the person with dementia. Ours is on the
wall.** The only thing we ask them to wear is a buzzer with no brain in it.

**"Haptic guidance isn't new."** It isn't, which is why we're confident it works. What's
new is what drives it: that literature guides people who **can't see**, with sensors on the
body. We guide someone who **can't remember**, from a room that already knows.

**"It's just a buzzer."** Yes, and that's the design. The intelligence is in the room,
which is why the worn part costs ten dollars, lasts on a charge, needs no pairing, and
degrades to a working voice product when it dies.

**"What about tags and trackers?"** A tag on every object, each charged, plus a phone app
driven at the moment of maximum confusion — and it finds the tag, never the untagged thing.

**"Isn't a camera in her home invasive?"** §6, volunteered before they ask.

**"Isn't the world model just a scene graph?"** Essentially yes — say it before they do.
Robotics has better ones. None deliver to a person, none consider a cognitive deficit, none
close the loop with an actuator a human wears.

---

## 12. The pitch

> **Open on the person.** "My grandmother puts her keys down, and ten minutes later has no
> memory of the last ten minutes. She checks the usual places. She asks her daughter. She
> asks again twenty minutes later. By the fourth time, she's sure someone took them."
>
> **Name it.** "That's not forgetfulness. Losing things and not being able to retrace your
> steps is one of the first signs of Alzheimer's. It costs her independence, it costs her
> daughter the whole day, and eventually it costs them trust in each other."
>
> **The gap.** "A tracker needs her to have tagged it, charged it, and opened an app. A
> camera needs her daughter to scrub an hour of video. And the AI assistants that can
> answer the question just *tell* her — which only works if she still remembers it when she
> gets to the kitchen."
>
> **Recall.** "Cameras over her counter. They don't keep footage — they remember objects.
> She asks out loud and it answers in a second."
>
> **The wow.** "And here's what no camera does: the glasses aren't in any frame. Recall
> still knows they're in the blue bag, because it saw them go in. It tells you that's an
> inference, and it shows you the picture."
>
> **Hand over the band.** "But being told only helps if you remember it. So put this on. It
> buzzes faster as you get closer. You don't have to remember anything — just follow it."
> *(then stop talking and let them find it)*
>
> **The turn.** "And it'll answer that same question forty times today, identically,
> without once sounding tired of it."
>
> **Close.** "Underneath, it's just a room that knows what's in it and what moved — which
> works anywhere nobody remembers who moved what, from a lab bench to a warehouse. We
> started in the one room where the person asking can't find out any other way."

**How to talk about it.** Lead with the person; the general version is the last sentence,
never the opener. Say "someone living with Alzheimer's," never "patients" or "sufferers."
One specific scene beats every statistic. Claim no pilot, no clinician, no medical
function, and volunteer the privacy answer unprompted. For a technical judge, switch
registers completely and go at the mechanism — the care framing is the *why*, not a
substitute for the *how*. **On the band: hand it over, don't describe it.** The sentence
that earns it is *"being told only helps if you remember it."* And say plainly that the
band knows nothing and the room does the work — people assume expensive sensors, and the
truth is better than their assumption.

---

## 13. Before anything is said on stage

- [ ] **Make the caregiver call.** Highest-value hour available to this pitch.
- [ ] **Verify the clinical claims or drop them.** The "early warning sign" attribution,
      "repetitive questioning is among the most common behaviours," and "accusations of
      theft are common" are written from memory, not a checked source. Any dementia
      statistic needs a source or it doesn't get said. Someone in this field will know.
- [ ] **Build the proximity meter before the strap.** Rung 3 first, always — it needs no
      calibration, so it can start on minute one against a stubbed marker position.
- [ ] **Run onboarding in the real room early.** If furniture detection finds nothing
      nameable, the manual-box fallback is the path and that should be known on day one.
- [ ] **Test marker tracking under demo lighting.** Colour thresholding is the one part
      lighting can break; the fix is a different marker colour and takes two minutes.
- [ ] **Confirm BLE works with the laptop on venue wifi** — or with no wifi at all.
- [ ] **Pick five visually distinct props.** Two similar dark objects produce identity
      swaps no threshold setting fixes.
- [ ] **Rehearse the voice path at volume** with the typed fallback one click away.
- [ ] **Rehearse strapping the band on someone else's wrist in three seconds**, and confirm
      the pulse is unmistakable through a sleeve.
- [ ] **Charge the band. Then charge the spare.**
- [ ] **Decide the closing sentence, out loud, in advance.**
