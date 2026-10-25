"""Setup and session tools. Part E setup (scan_to_room, calibrate_intrinsics, register_camera,
check_room) are uv scripts run as `uv run tools/<name>.py`; the rest run as `python -m tools.<name>`."""
from recall.store import read_json, write_json_atomic as write_json  # noqa: F401  (shared by the tools)
