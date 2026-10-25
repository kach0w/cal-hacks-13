"""Open a webcam by name or index on macOS. Shared by scripts/range_test.py and the room tools."""

import json
import subprocess
import sys

import cv2

# handoffAB.md: calibration and registration are only valid at this resolution.
WIDTH, HEIGHT = 1280, 720


def list_cameras():
    """Camera names in OpenCV's index order.

    OpenCV's AVFoundation backend sorts devices by uniqueID before indexing them,
    so sorting system_profiler's list the same way gives matching indices.
    """
    out = subprocess.run(
        ["system_profiler", "-json", "SPCameraDataType"], capture_output=True, text=True
    ).stdout
    cams = json.loads(out or "{}").get("SPCameraDataType", [])
    cams.sort(key=lambda c: c.get("spcamera_unique-id", ""))
    return [c["_name"] for c in cams]


def resolve_camera(camera):
    """A name (substring is fine) or index -> (index, device name)."""
    names = list_cameras()
    if camera.isdigit():
        index = int(camera)
    else:
        matches = [i for i, n in enumerate(names) if camera.lower() in n.lower()]
        if len(matches) != 1:
            listing = "\n".join(f"  [{i}] {n}" for i, n in enumerate(names))
            sys.exit(f"'{camera}' matched {len(matches)} cameras; pass a name or index from:\n{listing}")
        index = matches[0]
    return index, names[index] if index < len(names) else f"camera {index}"


def open_camera(camera, width=WIDTH, height=HEIGHT, exact=False):
    """exact=True exits unless the camera delivers exactly width x height."""
    index, name = resolve_camera(camera)
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"couldn't open camera {index} (check the terminal's camera permission)")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"using {name} at {w}x{h}")
    if exact and (w, h) != (width, height):
        sys.exit(f"camera gave {w}x{h}, need exactly {width}x{height}")
    return cap
