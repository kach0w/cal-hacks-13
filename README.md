# Recall

A corner webcam remembers where things are in a scanned room, and a band on the chest points the
person to them. Design: `docs/DESIGN.md`; camera pipeline: `docs/CAMERA.md`.

## From a fresh clone

A clone has no room, no lens calibration and no memory: each setup makes its own (all under
`data/`, which git ignores).

1. **Install** (Python 3.11; open3d has no wheels past 3.12):

       uv venv --python 3.11 --seed .venv
       uv pip install --python .venv/bin/python -r requirements.txt

2. **Markers:** print `printables/markers.pdf` at 100% and put at least 3 up where the webcam will
   see them, on the floor and on walls. Print `printables/calibration_board.pdf` too and mount it
   flat.
3. **Webcam:** mount it high in a corner at its final spot (1280×720).
4. **Scan** with Polycam (Settings → Developer mode on, then LiDAR mode). Pass within about 1 m of
   each marker, facing it. Export **Raw data** and **glTF** and drop both zips into `data/room/`.
5. **Set the room up:** `uv run tools/setup_room.py --camera <webcam name>`. The first time for a
   webcam it opens the lens calibration: hold the board up until the grid fills.
6. **Name the surfaces:** `.venv/bin/python -m recall.surfaces name`.
7. **Run:**

       .venv/bin/python -m recall.live --web --band recall-band.local --set SHOULDER_HEIGHT_M=1.45

   `SHOULDER_HEIGHT_M` is the guided person's shoulder height, about 0.82 × their height. Open the
   printed link on a phone on the same wifi. The band (`firmware/band`, or `firmware/pi_band.py`)
   must be on that wifi too. Models download on the first run.

   Add `--voice` to ask out loud on the laptop: press Enter in that terminal, ask, and the answer
   is spoken and the guidance starts. On its own, without the camera:
   `.venv/bin/python -m recall.voice --memory data/sim/real` (see `recall/voice.py`).

If the webcam is moved later, re-run `uv run tools/register_camera.py --camera <webcam name>`
with the markers still up. To start the memory over, stop live and delete `data/state.json`,
`data/events.jsonl`, `data/heartbeat.json` and `data/photos/`.

Tests: `.venv/bin/python -m pytest tests/ -q` (no hardware needed).
