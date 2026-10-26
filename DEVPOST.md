## Inspiration

We’ve all lost our keys and found them a minute later. For the 7.4 million Americans with Alzheimer’s, that minute doesn’t come: once something is forgotten, it stays forgotten. And over 4.3 million people with memory loss live alone, with nobody to ask.

Trackers need tags, cameras need someone to scrub footage, and AI assistants give you an answer you have to remember. Being told only helps if you remember it. So we built Recall: a room that remembers for you and guides you there with a buzz.

## What it does

A webcam in the corner of a room keeps a running memory of the things that matter: keys, wallet,
glasses, phone, the TV remote. It doesn't keep footage. It keeps *what's where, and what changed*.

- **Ask, and hear the answer.** Say *"where are my glasses?"* by pressing the button on the device. Recall answers out loud in about a second.
- **It answers even when the thing is out of sight.** When an object disappears, Recall looks at
  the state of the object into: *"Your glasses were last seen on the table, next to the blue bag, and the phone"* 
- **It never confidently lies.** Every answer says whether it was seen or where it last was. Someone
  who can't check the answer deserves to know how sure it is.
- **Then it points the way.** A small band on the device has an arrow on a servo. Recall finds the
  person, works out where they are standing in the room and which way they're facing, and turns
  the arrow toward the item. 

And it never gets tired of the question. The ninth "where's my wallet" before lunch gets the same
calm answer as the first.

## How we built it

**Setup, once.** The care partner puts a few printed ArUco markers on the floor and a wall, then
scans the room with a LiDAR iPhone in Polycam. Our setup tools find the markers in the scan and in
the webcam's view, which puts the camera and the 3D map in one set of coordinates. The surfaces
(the kitchen counter, the sofa) get named on a top-down plan of the scan.

**The live loop.** Every second a detection thread takes the newest frame. A change gate skips work when nothing changed; otherwise an computer vision model (YOLO-World-V2) finds each
enrolled item. Each bounding box's base is cast as a ray into the scanned room with open3d, giving a
real position in metres and the surface it's on. If the camera stops sending frames, the memory freezes instead of deciding everything vanished, and answers say
"as of 2:40 PM".

**The database.** Every change is an event in a time-series database not a picture. The current state
of every object is stored in 3D, and a nearest reference point, that's what makes "what changed today?" and "where do I often lose my things" one query. 

**Asking.** Speech-to-text runs on the laptop with faster-whisper. Exact item names
are matched instantly; vague wording ("the thing I open the front door with") goes to Moss
semantic search, which picks the item it means. Answers are templated, not generated, so they're
fast and can't make anything up.

**Guiding.** A YOLO11 pose model detects the person's shoulders in the scanned room to estimate their position. From there, the direction and distance to the item are sent to the device over Bluetooth packet. A servo connected to an ESP32 points the arrow while a voice tells to you where to go until you've reached your target

**The page.** A dependency-free web page served by the laptop: an ask box, a live compass dial
while guiding, the live camera, the "where things are" board in plain words, and the timeline.

## Built with our sponsors

- **Tiger Data:** Recall's memory is time-series data, so every change (moved, disappeared,
  went into the bag) is an event in a TimescaleDB hypertable on Tiger Data. A real-time continuous
  aggregate keeps per-day counts behind the care partner's "This week" panel, and "where do the
  glasses usually go missing?" is one SQL query. In our local benchmark over 900,000 synthetic
  events, compression made the data 83% smaller and a month-long summary ran in 29 ms from the
  aggregate instead of 357 ms from raw events.
- **Moss:** people don't name things the way a database does. Exact item names are matched
  instantly, and vague wording like "the thing I open the front door with" goes to Moss semantic
  search, which works out it means the keys. The page shows estion,
  so a care partner can tell if it heard wrong.
- **Claude (Anthropic):** "Ask about the day" is for the carwith
  dementia. Claude answers questions like "did the glasses go missing today?" and writes a short
  daily note. It reads the memory only through tools, cites from
  "probably", and says when it doesn't know. The person living with dementia still gets the same
  short, templated, local answers.
- **Cartesia:** everything Recall tells the person is spoken, so the voice matters. Answers and
  directions like "To your left" are spoken in a calm Cartes are made
  ahead of time so they play instantly, and if Cartesia is slow or unreachable Recall falls back
  to the built-in voice.

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
- **Small things are small.** Keys across a room are a handful of pixels. We measured what the
  camera could actually see at demo distances before choosing where to mount it and which props
  to use.

## Accomplishments that we're proud of

- **Object permanence, in software.** Infants develop it; dementia erodes it. Recall knows the
  glasses are in the bag when nobody can see them, and says honestly that it's a guess.
- **It closes the loop.** Everything in this space stops at an answer. The camera sees, the memory
  holds, the band acts, and the person never has to hold the answer in their head.
- **Real 3D positions from one webcam and a phone scan**, with the camera, the room and the person all in the same coordinates.
- **It runs on a laptop.** Detection, speech-to-text and pose all run locally; the room scan and
  the photos never leave the house.

## What we learned

- The hardest part of helping someone with memory loss isn't finding the answer, it's that the
  answer itself has to be remembered. Designing for "the help still works after they forget what
  it said" changed the whole product.
- One well-placed camera and a little geometry go a long way. Most of our accuracy came from
  better rules and better placement, not bigger models.
- Hardware should be dumb. Keeping all the thinking on the laptop let the band be a $10 servo and
  a button, and made every bug fixable in Python.

## What's next for Recall

- **A vibration motor instead of a servo**, and a smaller band that's easier to wear and charge.
- **Memory-care facilities:** the same product run by staff across many rooms, one camera each,
  with every room's history in one database.
- **Shared homes and workplaces:** loaner hardware, lab instruments, tools. A room that knows what's in it and what moved isn't specific to dementia.
- **A trial with families**, with the person living there consenting and in control. Recall is not
  a medical device, and we'd want real users and clinicians to shape it before it is anything more
  than a tool for independence.
