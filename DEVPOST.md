## Inspiration

A woman in her seventies puts her keys down in the kitchen. Ten minutes later she needs them and
has no memory of those ten minutes. She checks the usual places, asks her daughter, asks again
twenty minutes later. By the fourth time, she's certain someone took them.

That isn't forgetfulness. Misplacing objects and being unable to retrace the steps is one of the
recognized early signs of Alzheimer's, repeating the same question is among the most common
daily behaviours, and things that can't be found are often believed stolen, a suspicion that lands
on whoever is nearest. One lost set of keys costs her independence, her care partner's patience,
and a little of the relationship.

Nothing on the market closes this. Tags need to be attached, charged and tracked in a phone app
at the moment the person is least able to use one. Cameras record footage that someone has to
scrub through. AI assistants can answer the question, but they answer in words, and words have
to be remembered to be followed. We wanted something that remembers *for* her and then shows the
way, so the answer doesn't have to be held in her head at all.

## What it does

A webcam in the corner of a room keeps a running memory of the things that matter: keys, wallet,
glasses, phone, the TV remote. It doesn't keep footage. It keeps *what's where, and what changed*.

- **Ask, and hear the answer.** Say *"where are my glasses?"* by pressing the button on the band,
  the "Ask out loud" button on the web page, or by typing. Recall answers out loud in about a second.
- **It answers even when the thing is out of sight.** When an object disappears, Recall looks at
  the last picture it was in and works out what it went into: *"Your glasses are probably in the
  blue bag, on the table."* If the bag gets carried somewhere else, the glasses go with it.
- **It never confidently lies.** Every answer says whether it was seen or is a best guess. Someone
  who can't check the answer deserves to know how sure it is.
- **Then it points the way.** A small band on the chest has an arrow on a servo. Recall finds the
  person, works out where they are standing in the room and which way they're facing, and turns
  the arrow toward the item (or the bag it's in). The arrow keeps pointing after the answer is
  forgotten, which is exactly the failure the condition guarantees.
- **The care partner gets a page.** On any phone or laptop on the wifi: where each thing is, how
  sure Recall is, whether the camera is live, a timeline of everything that moved today, and a
  week of how often each thing went out of sight and where. The family can also ask in their own
  words ("did Mum lose her glasses again today?") and Claude answers from the memory. That turns
  "is she okay?" into something answerable without a phone call that feels like checking up.

And it never gets tired of the question. The ninth "where's my wallet" before lunch gets the same
calm answer as the first.

## How we built it

**Setup, once.** The care partner puts a few printed ArUco markers on the floor and a wall, then
scans the room with a LiDAR iPhone in Polycam. Our setup tools find the markers in the scan and in
the webcam's view, which puts the camera and the 3D map in one set of coordinates. The surfaces
(the kitchen counter, the sofa) get named on a top-down plan of the scan.

**The live loop.** Every second a detection thread takes the newest frame. A change gate skips work
when nothing changed; otherwise an open-vocabulary detector (YOLO-World, or YOLOE) finds each
enrolled item from a plain-language prompt like "keys on a red lanyard", so tracking something new
needs no training. Each box's base is cast as a ray into the scanned room with open3d, giving a
real position in metres and the surface it's on. Memory rules then decide what actually happened:
a move needs two looks in a row, a disappearance needs three, and an object that vanishes mostly
inside a container's box is recorded as *inferred* inside it, with a confidence. If the camera
stops sending frames, the memory freezes instead of deciding everything vanished, and answers say
"as of 2:40 PM".

**The database.** Every change is an event in a time-series database, alongside the current state
of every object and the camera's heartbeat. That's what makes "what changed today?" and "how often
did the glasses go missing this week?" one query, and it means the history is there for the care
partner even when they're not in the house. Photos never leave the laptop: only the facts are
stored. Events go into a TimescaleDB hypertable, and a continuous aggregate (`recall_daily`) keeps
per-day counts for the page's "This week" panel. A benchmark over about 900,000 synthetic events
(TimescaleDB in Docker on a laptop) showed the columnstore 83% smaller and a month-long summary
across 200 households about 10x faster from the aggregate.

**Asking.** Speech-to-text runs on the laptop with faster-whisper and Silero VAD. Exact item names
are matched instantly; vague wording ("the thing I open the front door with") goes to Moss
semantic search, which picks the item it means. The answers spoken to the person are templated,
not generated, so they're fast and can't make anything up. With a Cartesia key they're spoken in a
calm, slightly slow Sonic voice, cached by sentence; without one, or if it's slow, the laptop's and
browser's own voices take over.

**Ask about the day.** The care partner's questions are different, so they get a different tool.
A card on the page sends them to Claude, which reads the memory through five tools that wrap our
existing query code (where something is, what changed since a time, one thing's history, how
often things went missing). Every tool result keeps the memory's own "seen" and "probably", and
the page shows what Claude looked at under each answer. These answers are shown to the family,
never spoken to the person. Claude gets text facts only, never a picture.

Honestly: the Claude and Cartesia parts are tested against fakes and local stand-in servers, not
yet with live keys. Moss's item matching was checked against real Moss; its newer parts (the
"Understood" line, "when" questions) only against a fake. Without a key each one switches off
cleanly and the rest of Recall works the same.

**Guiding.** A YOLO11 pose model finds the person's shoulders. One camera can't see depth, but a
point at a known height can be placed: we put the shoulders at the person's shoulder height in the
scanned room, which gives where they're standing and, from the line between the shoulders, which
way their chest faces. The direction from there to the item goes to the band over Bluetooth LE as
a tiny `dir,dist` packet.

**The band.** An ESP32, one servo and a button. It decides nothing about the world: it points the
arrow, wiggles when you've arrived, and centres itself if packets stop, because a frozen arrow is
a lie. The button asks a question through the laptop's mic.

**The page.** A dependency-free web page served by the laptop: an ask box, a live compass dial
while guiding, the live camera, the "where things are" board in plain words, the timeline, the
"This week" panel and the "Ask about the day" card.

## Challenges we ran into

- **Depth from one camera.** A single webcam sees directions, not distances. Placing things where
  a pixel's ray meets the scan, and the person where their shoulders' rays meet shoulder height,
  is what made real positions possible without a second camera.
- **Which way is someone facing?** From behind, the pose model happily swaps left and right
  shoulders. We let the face decide: if the camera sees the face, they're facing it.
- **A servo that only turns ±90°.** For things behind you, the arrow would just sit at its limit,
  which on the left is also where it rests. It now waves at that edge, meaning "turn around, this
  way". Straight behind, a few degrees of wobble flipped the turn between +179° and -179° and
  swung the arrow edge to edge, so the band keeps to the side it was on until the item is clearly
  on the other.
- **Phantom moves.** A hand passing over the counter blanks objects for a moment. Requiring
  repeated looks before committing a move or a disappearance, and detecting people separately,
  stopped the memory from inventing events.
- **An unplugged camera looks like an empty room.** Without a guard, every object "disappeared".
  A heartbeat now freezes the memory and stamps answers with when it last really saw the room.
- **Voice in a loud room.** A hackathon floor is all speech. We only count speech that's clearly
  louder than the room's usual level (the person asking is close to the mic), and drop the
  phrases Whisper invents from noise.
- **Small things are small.** Keys across a room are a handful of pixels. We measured what the
  camera could actually see at demo distances before choosing where to mount it and which props
  to use.

## Accomplishments that we're proud of

- **Object permanence, in software.** Infants develop it; dementia erodes it. Recall knows the
  glasses are in the bag when nobody can see them, and says honestly that it's a guess.
- **It closes the loop.** Everything in this space stops at an answer. The camera sees, the memory
  holds, the band acts, and the person never has to hold the answer in their head.
- **Real 3D positions from one webcam and a phone scan**, with the camera, the room and the person
  all in the same coordinates.
- **Honest by construction.** Seen and inferred are separate all the way from the memory rules to
  the spoken answer, the page and the band.
- **It runs on a laptop.** Detection, speech-to-text and pose all run locally, and the room scan
  and the photos never leave the house. The optional services get text only: events for the
  database, the family's questions and short facts for Claude, the spoken sentences for Cartesia.

## What we learned

- The hardest part of helping someone with memory loss isn't finding the answer, it's that the
  answer itself has to be remembered. Designing for "the help still works after they forget what
  it said" changed the whole product.
- Confidence is a feature. A system that's wrong at someone who can't check it is worse than no
  system, so every layer carries how sure it is.
- One well-placed camera and a little geometry go a long way. Most of our accuracy came from
  better rules and better placement, not bigger models.
- Hardware should be dumb. Keeping all the thinking on the laptop let the band be a $10 servo and
  a button, and made every bug fixable in Python.

## What's next for Recall

- **A vibration motor instead of a servo**, and a smaller band that's easier to wear and charge.
- **Memory-care facilities:** the same product run by staff across many rooms, one camera each,
  with every room's history in one database.
- **Shared homes and workplaces:** loaner hardware, lab instruments, tools. A room that knows what's
  in it and what moved isn't specific to dementia.
- **A trial with families**, with the person living there consenting and in control. Recall is not
  a medical device, and we'd want real users and clinicians to shape it before it is anything more
  than a tool for independence.
