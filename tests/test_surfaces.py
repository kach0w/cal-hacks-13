"""Surfaces (DESIGN §6.1) on Person A's real scan and plan, and places in 3D through the memory."""
import json
import logging
import shutil
import sys

import cv2
import numpy as np
import open3d as o3d
import pytest

from types import SimpleNamespace

from fakes import FRAME, ITEMS, KEYS_COUNTER, S, ZONES, kinds, world
from recall.config import ROOT, Settings
from recall.memory import Memory
from recall.models import Surface, Zone
from recall.overlay import draw_surfaces, surface_outline
from recall.places import Places, surface_at
from recall.room import Room, read_mesh
from recall.store import Store
from recall import surfaces as sf

REAL = ROOT / "data" / "room"
SOFA = Surface("sofa", [3.9, 0.3, 0.7], [4.6, 1.0, 2.3])
TABLE = Surface("coffee table", [1.8, 0.2, 0.9], [2.8, 0.9, 1.5])
SINK = Surface("sink", [0.0, 0.8, 3.0], [0.6, 1.2, 3.6], "in")


@pytest.fixture(scope="module")
def scan():
    v, f = read_mesh(REAL / "room.glb")
    plan = cv2.imread(str(REAL / "plan.png"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    return v, plan, scene


def test_plan_mapping_matches_scan_to_room(scan):
    """Every marker corner lands on the magenta outline scan_to_room drew around it."""
    v, plan, _ = scan
    lo = sf.plan_origin(v, plan.shape)
    assert sf.to_plan(lo, 0, 0) == (105, 131)                       # the yellow origin dot
    for c in json.loads((REAL / "markers.json").read_text())["markers"].values():
        for x, _, z in c:
            u, v_ = sf.to_plan(lo, x, z)
            assert plan[v_, u].tolist() == [255, 0, 255]


def test_plan_from_another_scan_is_refused(scan):
    v, plan, _ = scan
    with pytest.raises(SystemExit, match="different scans"):
        sf.plan_origin(v, (plan.shape[0] + 10, plan.shape[1]))


def test_heights_under_the_plan(scan):
    _, _, scene = scan
    assert abs(sf.top_height(scene, 3.14, 3.64, 1.48, 1.98)) < 0.03         # floor by marker 4
    assert 0.35 < sf.top_height(scene, 3.94, 4.54, 0.68, 2.28) < 0.55        # the sofa's seat
    assert sf.top_height(scene, 20, 21, 20, 21) is None                       # outside the scan


def test_a_loose_box_keeps_the_table_top(scan):
    """Boxes dragged around the glass coffee table on plan.png, tight to 17 cm loose a side: more and
    more floor shows around the table (59-66% of the samples in the loosest two), still its top."""
    v, plan, scene = scan
    lo = sf.plan_origin(v, plan.shape)
    tops = []
    for x, y, w, h in ([270, 220, 128, 72], [264, 214, 140, 84], [258, 208, 152, 96], [250, 200, 170, 110]):
        (x0, z0), (x1, z1) = lo + np.array([x, y]) / sf.PLAN_PPM, lo + np.array([x + w, y + h]) / sf.PLAN_PPM
        tops.append(sf.top_height(scene, x0, x1, z0, z1))
    assert all(0.3 < t < 0.4 for t in tops) and max(tops) - min(tops) < 0.02, tops


def test_surface_at():
    surfaces = [SOFA, TABLE, Surface("tray", [2.0, 0.2, 1.0], [2.3, 0.9, 1.2])]
    assert surface_at(surfaces, [2.1, 0.33, 1.1]) == "tray"                 # smallest wins
    assert surface_at(surfaces, [2.6, 0.33, 1.4]) == "coffee table"
    assert surface_at(surfaces, [2.83, 0.33, 1.4]) == "coffee table"        # within the 5 cm margin
    assert surface_at(surfaces, [2.9, 0.33, 1.4]) is None
    assert surface_at(surfaces, None) is None


def test_overlapping_boxes_go_to_the_highest_surface():
    """A floor box dragged 10 cm into the sofa: its 0.6 m headroom holds the seat's front edge too, and
    it's the smaller box. The seat is higher, so things on it are on the sofa; a tray a little lower
    than the table it stands on (scan noise) is still the tray."""
    sofa = Surface("sofa", [3.824, 0.351, 0.434], [4.724, 1.051, 2.534], top=0.451)
    floor = Surface("floor by the sofa", [3.344, -0.117, 1.684], [3.924, 0.583, 2.484], top=-0.017)
    for seat in ([3.90, 0.31, 2.0], [3.92, 0.365, 2.0], [3.96, 0.40, 2.0]):  # the real seat's front edge
        assert surface_at([sofa, floor], seat) == "sofa", seat
    assert surface_at([sofa, floor], [3.90, 0.0, 2.0]) == "floor by the sofa"
    table = Surface("coffee table", [1.6, 0.25, 0.8], [2.9, 0.95, 1.7], top=0.35)
    tray = Surface("tray", [2.0, 0.22, 1.0], [2.3, 0.92, 1.2], top=0.32)
    assert surface_at([table, tray], [2.1, 0.33, 1.1]) == "tray"


def test_handoff_zones3d_format_is_read():
    s = Surface.from_dict({"name": "kitchen counter", "min": [0, 0.8, 0], "max": [1, 1.2, 2], "prep": "on"})
    assert (s.box_min, s.box_max, s.prep, s.top) == ([0, 0.8, 0], [1, 1.2, 2], "on", 0.8)   # top: its bottom
    assert list(s.to_dict()) == ["name", "box_min", "box_max", "prep", "top"]


def test_a_hand_made_box_is_listed_and_drawn_at_its_bottom(tmp_path, monkeypatch, capsys):
    """handoffAB.md §6's zones3d.json has no top: min y 0.30 around a 0.32 m table is 0.30, not 0.40."""
    (tmp_path / "room").mkdir()
    (tmp_path / "room" / "zones3d.json").write_text(json.dumps(
        [{"name": "coffee table", "min": [1.64, 0.30, 0.88], "max": [2.92, 0.60, 1.60]}]))
    monkeypatch.setattr(sys, "argv", ["surfaces", "list", "--set", f"DATA_DIR={tmp_path}"])
    sf.main()
    assert "top 0.30" in capsys.readouterr().out


@pytest.fixture
def data(tmp_path):
    (tmp_path / "room").mkdir()
    for name in ("room.glb", "plan.png", "markers.json"):
        shutil.copy(REAL / name, tmp_path / "room" / name)
    return tmp_path


def test_naming_on_the_plan(data, monkeypatch):
    """Drag two boxes on plan.png, name them: world boxes with heights from the scan."""
    rois = np.array([[500, 200, 60, 160], [420, 280, 50, 50], [10, 10, 3, 3]])   # sofa, floor, a slip
    answers = iter(["sofa", "on", "floor by the window", "on"])
    monkeypatch.setattr(cv2, "selectROIs", lambda *a, **k: rois)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("recall.overlay.preview", lambda path: None)
    monkeypatch.setattr("recall.overlay.close_windows", lambda: None)
    cfg = Settings(data_dir=data)
    sf.name(cfg)
    got = {s.name: s for s in sf.load_surfaces(cfg.surfaces_path)}
    assert list(got) == ["sofa", "floor by the window"]
    sofa = got["sofa"]
    assert np.allclose([sofa.box_min[0], sofa.box_min[2], sofa.box_max[0], sofa.box_max[2]],
                       [3.944, 0.684, 4.544, 2.284], atol=0.002)
    assert 0.35 < sofa.top < 0.55 and sofa.box_max[1] - sofa.box_min[1] == pytest.approx(0.7)
    assert sofa.box_min[1] == pytest.approx(sofa.top - sf.BELOW_M)
    assert abs(got["floor by the window"].top) < 0.03
    assert (data / "room" / "surfaces_drawn.jpg").exists()


def test_show_draws_on_the_plan_and_the_camera(data, monkeypatch):
    from test_room import camera_json
    monkeypatch.setattr("recall.overlay.preview", lambda path: None)
    cfg = Settings(data_dir=data)
    sf.save_surfaces([SOFA, TABLE], cfg.surfaces_path)
    sf.show(cfg)
    assert (data / "room" / "surfaces.png").exists() and not (data / "room" / "surfaces_camera.jpg").exists()
    (data / "room" / "camera.json").write_text(json.dumps(camera_json()))
    cv2.imwrite(str(data / "room" / "camera_ref.jpg"), np.zeros((720, 1280, 3), np.uint8))
    sf.show(cfg)
    drawn = cv2.imread(str(data / "room" / "surfaces_camera.jpg"))
    assert drawn.sum() > 0                                                # outlines drawn on the black frame


def test_names_sit_inside_their_outlines():
    """The name goes in the middle of what's seen of the outline, not at a corner of its bounding box
    (which, the view being oblique, lies outside it)."""
    from test_room import camera_json
    room = Room(camera_json())
    checked = 0
    for s in (SOFA, TABLE, Surface("floor", [1.0, -0.1, 0.5], [3.5, 0.6, 3.0], top=0.0)):
        runs, anchor = surface_outline(s, room, (1280, 720))
        if anchor is not None:
            outline = np.concatenate(runs).astype(np.float32)
            assert cv2.pointPolygonTest(outline, anchor, False) > 0, s.name
            checked += 1
    assert checked == 3


def test_outlines_only_where_the_lens_model_holds():
    """data/room/intrinsics.json's dist folds back ~51 degrees off-axis: projected through it, a rug
    beside and below the camera (out of view) lands inside the image. It must not be drawn."""
    from test_room import look_at
    R, t = look_at((0.15, 2.3, 0.15), (3.2, 0.3, 2.6))
    dist = json.loads((REAL / "intrinsics.json").read_text())["dist"]
    room = Room({"width": 1280, "height": 720, "K": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], "dist": dist,
                 "R": R.tolist(), "t": t.tolist()})
    rug = Surface("rug by the door", [-0.05, -0.1, 1.1], [0.75, 0.6, 1.9], top=0.0)
    corners = [room.world_to_pixel(x, 0.0, z) for x in (-0.05, 0.75) for z in (1.1, 1.9)]
    assert any(0 <= u < 1280 and 0 <= v < 720 for u, v in corners)        # the fold puts it in the picture
    runs, anchor = surface_outline(rug, room, (1280, 720))
    assert anchor is None and all((r[:, 1] >= 720).all() for r in runs)  # what's kept is below the frame
    assert draw_surfaces(np.zeros((720, 1280, 3), np.uint8), [rug], room).sum() == 0
    runs, anchor = surface_outline(TABLE, room, (1280, 720))              # in view: drawn as before
    assert runs and anchor is not None


# --- places in 3D through the memory ------------------------------------------------------------
def p2w(u, v):
    """A fake room: pixel u -> x, v -> z; heights: the sink's at the bottom, the sofa's on the right,
    else the coffee table's."""
    if v < 100:
        return None                                   # nothing scanned behind the top of the image
    return (u / 300, 1.0 if v >= 1200 else 0.45 if u >= 1180 else 0.33, v / 400)


def three_d(tmp_path):
    cfg, store, _, q = world(tmp_path)
    sf.save_surfaces([SOFA, TABLE, SINK], cfg.surfaces_path)
    (cfg.room_dir / "camera.json").write_text("{}")                   # registered: answers word surfaces
    return cfg, store, Memory(ITEMS, Places([], [SOFA, TABLE, SINK]), store, cfg, p2w), q


ON_TABLE = [600, 400, 660, 440]          # base (630, 440) -> (2.1, 0.33, 1.1): the coffee table
ON_SOFA = [1200, 400, 1260, 600]         # base (1230, 600) -> (4.1, 0.45, 1.5): the sofa
IN_AIR = [600, 20, 660, 60]              # base (630, 60): nothing behind it -> no place


def test_places_are_surfaces_with_xyz(tmp_path):
    cfg, store, mem, q = three_d(tmp_path)
    [ev] = mem.update([S("keys", ON_TABLE)], FRAME, 0)
    assert (ev.kind, ev.to_surface, ev.to_xyz) == ("appeared", "coffee table", [2.1, 0.33, 1.1])
    assert mem.state["keys"].xyz == [2.1, 0.33, 1.1]
    assert q.where_is("keys", now=1).text == "Your keys are on the coffee table — seen just now."


def test_a_move_between_surfaces_is_confirmed_and_carries_both_positions(tmp_path):
    cfg, store, mem, q = three_d(tmp_path)
    mem.update([S("keys", ON_TABLE)], FRAME, 0)
    assert mem.update([S("keys", ON_SOFA)], FRAME, 1) == []
    assert mem.state["keys"].xyz == [2.1, 0.33, 1.1]                     # not moved until confirmed
    [ev] = mem.update([S("keys", ON_SOFA)], FRAME, 2)
    assert (ev.kind, ev.from_surface, ev.to_surface) == ("moved", "coffee table", "sofa")
    assert (ev.from_xyz, ev.to_xyz) == ([2.1, 0.33, 1.1], [4.1, 0.45, 1.5])
    assert q.describe(ev) == "keys moved from on the coffee table to on the sofa"


def test_no_map_point_reads_as_mid_air(tmp_path):
    cfg, store, mem, q = three_d(tmp_path)
    mem.update([S("keys", ON_TABLE)], FRAME, 0)
    log = [mem.update([S("keys", IN_AIR)], FRAME, t) for t in (1, 2, 3)]
    assert [kinds(x) for x in log] == [[], [], ["moved:keys"]]           # None needs one look more
    assert (log[2][0].to_surface, log[2][0].to_xyz) == (None, None)


def test_surface_prepositions_in_answers(tmp_path):
    cfg, store, mem, q = three_d(tmp_path)
    [ev] = mem.update([S("keys", [60, 1300, 120, 1400])], FRAME, 0)     # base -> (0.3, 1.0, 3.5)
    assert ev.to_surface == "sink"
    assert q.where_is("keys", now=1).text == "Your keys are in the sink — seen just now."


def test_setup_places_picks_one_kind(tmp_path, caplog):
    from recall.live import setup_places
    from test_room import camera_json
    cfg = Settings(data_dir=tmp_path)
    (tmp_path / "room").mkdir()
    from recall.zones import save_zones
    save_zones([Zone("table", [0, 0, 10, 10])], cfg.zones_path)
    sf.save_surfaces([SOFA], cfg.surfaces_path)
    meshed = SimpleNamespace(source="mesh")
    with caplog.at_level(logging.WARNING):
        p = setup_places(cfg, room=None)                                  # surfaces need the room
    assert not p.in_3d and p.names() == ["table"] and "needs the registered room" in caplog.text
    assert setup_places(cfg, room=meshed).names() == ["sofa"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):                                # ...and its mesh: on the floor
        assert setup_places(cfg, room=Room(camera_json())).names() == ["table"]   # plane, all is floor
    assert "needs room.glb" in caplog.text
    cfg.surfaces_path.unlink()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert setup_places(cfg, room=meshed).names() == ["table"]
    assert "no surfaces named" in caplog.text


def test_answers_word_places_as_the_live_loop_does(tmp_path):
    """A zone and a surface both called 'sink': until the room is registered the live loop places by
    zone, so answers use the zone's 'in'; once registered, the surface's 'on'."""
    cfg, store, mem, q = world(tmp_path, zones=[Zone("sink", [0, 0, 1280, 720], "in")])
    sf.save_surfaces([Surface("sink", [0, 0, 0], [1, 1, 1], "on")], cfg.surfaces_path)
    mem.update([S("keys", ON_TABLE)], FRAME, 0)
    assert q.where_is("keys", now=1).text == "Your keys are in the sink — seen just now."
    (cfg.room_dir / "camera.json").write_text("{}")
    assert q.where_is("keys", now=1).text == "Your keys are on the sink — seen just now."


def test_switching_from_zones_to_surfaces_is_flagged(tmp_path, caplog):
    """state.json from a run on image zones, then the room registered: the old names stay (each logs a
    'moved' when next seen), so live warns at startup and says how to start clean."""
    from recall.live import warn_unknown_places
    cfg, store, mem, q = world(tmp_path)
    mem.update([S("keys", KEYS_COUNTER)], FRAME, 0)
    with caplog.at_level(logging.WARNING):
        warn_unknown_places(Memory(ITEMS, Places(ZONES), store, cfg))
    assert caplog.text == ""
    with caplog.at_level(logging.WARNING):
        warn_unknown_places(Memory(ITEMS, Places([], [SOFA, TABLE, SINK]), store, cfg, p2w))
    assert "kitchen counter" in caplog.text and "aren't today's surfaces" in caplog.text
    assert "delete state.json, events.jsonl, heartbeat.json and photos/" in caplog.text
