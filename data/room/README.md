# data/room

This room's setup. It lives only on this machine: git ignores everything here but this file, so a
fresh clone starts blank.

To set a room up, put the markers up, scan it with Polycam (Developer mode on, LiDAR mode), export
**Raw data** and **glTF**, drop both zips here, then from the repo root:

    uv run tools/setup_room.py --camera <webcam name>
    .venv/bin/python -m recall.surfaces name

That writes room.glb, markers.json, plan.png, scan_report.json, camera.json, camera_ref.jpg,
camera_overlay.jpg and surfaces.json here (formats in docs/handoffAB.md), and the webcam's lens
file to data/cameras/.
