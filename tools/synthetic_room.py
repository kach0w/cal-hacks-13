"""Synthetic room in the handoffAB.md formats, from known ground truth, for testing Part E end to end.

A 4.0 × 2.6 × 3.5 m box room with a table, four ArUco markers (floor + two walls) and a corner camera
posed in ARKit/GL axes. Images are rendered exactly through the lens model: every sample is
undistorted to a ray and intersected with the scene analytically, so the distortion is exact.

    python -m tools.synthetic_room --out data/room_synthetic

Writes room.glb, markers.json, camera.json (true pose), camera_intrinsics.json (K/dist only),
checks.json, zones3d.json, camera_ref.jpg, truth.json, frames/ (noisy registration frames) and
boards/ (checkerboard views for calibrate_intrinsics).
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from recall.room import gl_to_opencv, project, undistort
from tools import write_json

log = logging.getLogger(__name__)

SIZE = (1280, 720)
K = np.array([[834.0, 0.0, 643.5], [0.0, 836.0, 357.2], [0.0, 0.0, 1.0]])  # ~75° HFOV
DIST = np.array([-0.10, 0.03, 0.0004, -0.0003, 0.0])
ROOM = np.array([4.0, 2.6, 3.5])  # x, y (up), z extents; origin at a floor corner
TABLE = np.array([[2.3, 0.0, 0.6], [3.3, 0.75, 1.4]])  # min, max corners
DICTIONARY = "DICT_4X4_50"
MARKER_M = 0.15
PAPER = 0.25  # white margin around a marker, fraction of its size
# id: (centre, normal of the surface it is taped to, marker "up")
MARKERS = {
    3: ((1.5, 0.0, 1.9), (0, 1, 0), (0, 0, 1)),
    7: ((2.9, 0.0, 2.6), (0, 1, 0), (1, 0, 0)),
    12: ((4.0, 0.9, 2.2), (-1, 0, 0), (0, 1, 0)),
    21: ((2.0, 0.9, 3.5), (0, 0, -1), (0, 1, 0)),
}
TEST_POINTS = {"far_corner_floor": (3.6, 0.0, 3.1), "mid_floor": (2.0, 0.0, 2.3)}
TEST_COLORS = [(40, 40, 220), (220, 90, 40)]  # BGR squares drawn at the test points
TAPE_NOISE = np.array([0.012, -0.009, 0.015])  # checks.lens_measured = true centre + this
EYE, YAW, PITCH, ROLL = (0.3, 2.25, 0.3), 42.0, -35.0, 2.0  # corner camera; degrees
BOARD, SQUARE_M = (9, 6), 0.025
# BGR per surface: walls x=0, x=4, floor, ceiling, walls z=0, z=3.5, table
COLORS = np.array([[170, 178, 186], [176, 182, 188], [112, 128, 146], [205, 205, 205],
                   [160, 172, 190], [168, 176, 186], [58, 92, 138]], float)
WHITE, BLACK = 235.0, 25.0
CHUNK = 48  # output rows rendered at a time (bounds memory)


def camera_gl(eye=EYE, yaw=YAW, pitch=PITCH, roll=ROLL):
    """4x4 camera->world pose in ARKit/GL axes (x right, y up, looks down -z). Yaw is from +X toward +Z."""
    y, p, r = np.radians([yaw, pitch, roll])
    f = np.array([np.cos(p) * np.cos(y), np.sin(p), np.cos(p) * np.sin(y)])
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    roll_m = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = np.c_[right, np.cross(right, f), -f] @ roll_m
    T[:3, 3] = eye
    return T


def marker_corners(centre, normal, up, size=MARKER_M):
    """TL, TR, BR, BL (OpenCV order) of a marker seen upright from the side its surface faces."""
    c, n, u = (np.asarray(a, float) for a in (centre, normal, up))
    r = np.cross(-n, u)  # viewer looks along -n; right = forward × up
    h = size / 2
    return np.array([c + h * u - h * r, c + h * u + h * r, c - h * u + h * r, c - h * u - h * r])


def scene_markers():
    return {i: marker_corners(*m) for i, m in MARKERS.items()}


def first_hit(origin, dirs):
    """Analytic first surface along rays from inside the room: (distance in units of dirs, surface, axis)."""
    idx = np.arange(len(dirs))
    with np.errstate(divide="ignore", invalid="ignore"):
        shell = np.where(dirs > 0, ROOM - origin, -origin) / dirs  # distance to the wall each axis runs into
        t0, t1 = (TABLE[0] - origin) / dirs, (TABLE[1] - origin) / dirs
    shell = np.where(shell > 0, shell, np.inf)
    axis = shell.argmin(1)
    s = shell[idx, axis]
    surf = 2 * axis + (dirs[idx, axis] > 0)
    near, far = np.minimum(t0, t1), np.maximum(t0, t1)  # slab test for the table box
    enter, leave = near.max(1), far.min(1)
    table = (enter <= leave) & (enter > 0) & (enter < s)
    s[table], surf[table], axis[table] = enter[table], 6, near.argmax(1)[table]
    return s, surf, axis


def visible(points, R, t):
    """Per world point: inside the frame, in front of the camera and not occluded."""
    points = np.asarray(points, float).reshape(-1, 3)
    origin = -R.T @ t
    s, _, _ = first_hit(origin, points - origin)  # the point itself is at s = 1
    uv = project(points, K, DIST, R, t)
    inside = (uv >= 0).all(1) & (uv[:, 0] <= SIZE[0] - 1) & (uv[:, 1] <= SIZE[1] - 1)
    return inside & ((points @ R.T + t)[:, 2] > 0) & (s > 1 - 1e-6)


def marker_bits(marker_id):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.generateImageMarker(d, marker_id, 6, borderBits=1)  # one pixel per cell


def marker_texture(p, corners, bits):
    """Gray level of a printed marker and its paper margin at points p; NaN off the paper."""
    tl, tr, _, bl = corners
    ex, ey = tr - tl, bl - tl
    rel = p - tl
    a, b = rel @ ex / (ex @ ex), rel @ ey / (ey @ ey)
    on = ((np.abs(rel @ np.cross(ex, ey)) < 1e-9) & (a >= -PAPER) & (a <= 1 + PAPER)
          & (b >= -PAPER) & (b <= 1 + PAPER))
    g = np.where(on, WHITE, np.nan)
    inside = on & (a >= 0) & (a < 1) & (b >= 0) & (b < 1)
    n = len(bits)
    g[inside] = np.where(bits[(b[inside] * n).astype(int), (a[inside] * n).astype(int)] > 0, WHITE, BLACK)
    return g


def room_shader(R, t):
    """shade(xy) -> BGR for normalized camera coordinates, seen from pose (R, t)."""
    origin = -R.T @ t
    markers = [(c, marker_bits(i)) for i, c in scene_markers().items()]

    def shade(xy):
        d = np.c_[xy, np.ones(len(xy))] @ R
        s, surf, axis = first_hit(origin, d)
        p = origin + s[:, None] * d
        col = COLORS[surf]
        floor = surf == 2
        col[floor & ((np.floor(p[:, 0] / 0.5) + np.floor(p[:, 2] / 0.5)) % 2 == 0)] *= 0.92  # floor tiles
        for corners, bits in markers:
            g = marker_texture(p, corners, bits)
            m = ~np.isnan(g)
            col[m] = g[m, None]
        for (x, _, z), color in zip(TEST_POINTS.values(), TEST_COLORS):
            col[floor & (np.abs(p[:, 0] - x) <= 0.025) & (np.abs(p[:, 2] - z) <= 0.025)] = color
        cos = np.abs(d[np.arange(len(d)), axis]) / np.linalg.norm(d, axis=1)
        return col * (0.6 + 0.4 * cos)[:, None]

    return shade


def board_shader(Rb, tb):
    """shade(xy) for a checkerboard (BOARD inner corners) at board->camera pose (Rb, tb)."""
    n = Rb[:, 2]
    cols, rows = BOARD

    def shade(xy):
        d = np.c_[xy, np.ones(len(xy))]
        s = (n @ tb) / (d @ n)
        local = (s[:, None] * d - tb) @ Rb
        i = np.floor(local[:, 0] / SQUARE_M).astype(int) + 1  # pattern squares are 0..cols, 0..rows
        j = np.floor(local[:, 1] / SQUARE_M).astype(int) + 1
        paper = (s > 0) & (i >= -1) & (i <= cols + 1) & (j >= -1) & (j <= rows + 1)
        black = paper & (i >= 0) & (i <= cols) & (j >= 0) & (j <= rows) & ((i + j) % 2 == 0)
        g = np.where(black, BLACK, np.where(paper, WHITE, 100.0))
        return np.repeat(g[:, None], 3, 1)

    return shade


def ray_grid(K, dist, size, ss):
    """Undistorted normalized coordinates of ss×ss samples per pixel: (H·ss, W·ss, 2) float32."""
    w, h = size
    off = (np.arange(ss) + 0.5) / ss - 0.5  # around integer pixel centres (OpenCV convention)
    u = (np.arange(w)[:, None] + off).ravel()
    v = (np.arange(h)[:, None] + off).ravel()
    grid = np.empty((h * ss, w * ss, 2), np.float32)
    for r in range(0, len(v), CHUNK * ss):
        uu, vv = np.meshgrid(u, v[r:r + CHUNK * ss])
        grid[r:r + len(uu)] = undistort(np.c_[uu.ravel(), vv.ravel()], K, dist).reshape(len(uu), -1, 2)
    return grid


def render(grid, shade, ss):
    """Box-filter the shaded samples into a float32 (H, W, 3) image."""
    h, w = grid.shape[0] // ss, grid.shape[1] // ss
    img = np.empty((h, w, 3), np.float32)
    for r in range(0, h, CHUNK):
        xy = grid[r * ss:(r + CHUNK) * ss].reshape(-1, 2).astype(np.float64)
        n = len(xy) // (w * ss * ss)
        img[r:r + n] = shade(xy).reshape(n, ss, w, ss, 3).mean((1, 3))
    return img


def noisy_frames(clean, n, rng, sigma=3.0, gain=0.04):
    """n camera frames: per-pixel Gaussian noise plus a small global brightness jitter."""
    return [np.clip(np.rint(clean * rng.uniform(1 - gain, 1 + gain) + rng.normal(0, sigma, clean.shape)),
                    0, 255).astype(np.uint8) for _ in range(n)]


def render_room(R, t, ss=3):
    return render(ray_grid(K, DIST, SIZE, ss), room_shader(R, t), ss)


def board_pose(goal_uv, Rb, dist_m):
    """Board->camera translation that puts the board as near goal_uv as fits in the frame, or None."""
    cols, rows = BOARD
    centre = np.array([(cols - 1) / 2, (rows - 1) / 2, 0.0]) * SQUARE_M
    paper = np.array([[-2, -2, 0], [cols + 1, -2, 0], [cols + 1, rows + 1, 0], [-2, rows + 1, 0]]) * SQUARE_M
    lo, hi = np.full(2, 4.0), np.array(SIZE) - 5.0
    for _ in range(10):  # push the board back into the frame, along each axis, as far as its paper sticks out
        tb = dist_m * np.r_[undistort([goal_uv], K, DIST)[0], 1.0] - Rb @ centre
        uv = project(paper @ Rb.T + tb, K, DIST, np.eye(3), np.zeros(3))
        push = np.maximum(lo - uv.min(0), 0) - np.maximum(uv.max(0) - hi, 0)
        if not push.any():
            return tb
        goal_uv = goal_uv + push + np.sign(push)
    return None


def board_poses(n, rng):
    """n board->camera poses aimed at a 5×3 grid reaching the image edges, 0.5-0.8 m away, tilted up to ~35°."""
    poses = []
    while len(poses) < n:
        k = len(poses)
        goal = np.array([k % 5 / 4, k // 5 % 3 / 2]) * SIZE
        Rb = cv2.Rodrigues(np.radians(rng.uniform([-35, -35, -15], [35, 35, 15])))[0]
        tb = board_pose(goal, Rb, rng.uniform(0.5, 0.8))
        if tb is not None:
            poses.append((Rb, tb))
    return poses


def board_views(n, rng, ss=2):
    """n rendered checkerboard views: [(uint8 BGR image, Rb, tb)]."""
    grid = ray_grid(K, DIST, SIZE, ss)
    return [(noisy_frames(render(grid, board_shader(Rb, tb), ss), 1, rng, sigma=2.0)[0], Rb, tb)
            for Rb, tb in board_poses(n, rng)]


def room_mesh():
    """The room shell (facing inward) plus the table: (vertices (V, 3), triangles (F, 3))."""
    import open3d as o3d

    shell = o3d.geometry.TriangleMesh.create_box(*ROOM)
    shell.triangles = o3d.utility.Vector3iVector(np.asarray(shell.triangles)[:, ::-1].copy())
    mesh = shell + o3d.geometry.TriangleMesh.create_box(*(TABLE[1] - TABLE[0])).translate(TABLE[0])
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def write_mesh(path, vertices, triangles):
    import open3d as o3d

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.asarray(triangles, np.int32))
    o3d.t.io.write_triangle_mesh(str(path), mesh)  # the legacy writer's .glb can't be read back by Open3D 0.20


def generate(out, frames=30, boards=0, seed=0):
    """Write a complete synthetic room dir. Returns (ground truth as in truth.json, clean float render)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    T = camera_gl()
    R, t = gl_to_opencv(T)  # the ARKit (Y-up, GL camera) -> OpenCV path
    markers = scene_markers()
    pts = np.vstack([*markers.values(), list(TEST_POINTS.values())])
    if not visible(pts, R, t).all():
        raise ValueError("synthetic camera does not see every marker corner and test point")
    log.info("rendering the corner camera's view")
    clean = render_room(R, t)

    intrinsics = {"width": SIZE[0], "height": SIZE[1], "K": K.tolist(), "dist": DIST.tolist()}
    write_json(out / "camera_intrinsics.json", intrinsics)
    write_json(out / "camera.json", {**intrinsics, "R": R.tolist(), "t": t.tolist(),
                                     "camera_center": T[:3, 3].tolist(), "rms_px": 0.0,
                                     "markers_used": sorted(markers),
                                     "registered_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    write_json(out / "markers.json", {"dictionary": DICTIONARY, "marker_size_m": MARKER_M,
                                      "markers": {str(i): c.tolist() for i, c in markers.items()}})
    write_json(out / "checks.json", {"lens_measured": (T[:3, 3] + TAPE_NOISE).tolist(),
                                     "test_points": {k: list(p) for k, p in TEST_POINTS.items()}})
    write_json(out / "zones3d.json", [{"name": "table", "min": [2.3, 0.7, 0.6], "max": [3.3, 1.1, 1.4],
                                       "prep": "on"}])
    write_mesh(out / "room.glb", *room_mesh())
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 95]
    cv2.imwrite(str(out / "camera_ref.jpg"), noisy_frames(clean, 1, rng)[0], jpg)
    if boards:
        log.info("rendering %d checkerboard views", boards)
    for sub, images in (("frames", noisy_frames(clean, frames, rng)),
                        ("boards", [img for img, _, _ in board_views(boards, rng)])):
        if images:
            (out / sub).mkdir(exist_ok=True)
            for k, img in enumerate(images):
                cv2.imwrite(str(out / sub / f"{sub[:-1]}_{k:03d}.jpg"), img, jpg)
    truth = {"R": R.tolist(), "t": t.tolist(), "camera_center": T[:3, 3].tolist(), "camera_gl": T.tolist(),
             "test_pixels": {k: project([p], K, DIST, R, t)[0].tolist() for k, p in TEST_POINTS.items()}}
    write_json(out / "truth.json", truth)
    return truth, clean


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic Part E room directory.")
    ap.add_argument("--out", default="data/room_synthetic")
    ap.add_argument("--frames", type=int, default=30, help="noisy registration frames to write")
    ap.add_argument("--boards", type=int, default=15, help="checkerboard views to write")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    truth, _ = generate(args.out, args.frames, args.boards, args.seed)
    pixels = " ".join(f"--test-pixel {k}={u:.1f},{v:.1f}" for k, (u, v) in truth["test_pixels"].items())
    print(f"wrote {args.out}; to exercise the tools on it:")
    if args.boards:
        print(f"  python -m tools.calibrate_intrinsics --room {args.out} --images {args.out}/boards")
    if args.frames:
        print(f"  python -m tools.register_camera --room {args.out} --images {args.out}/frames {pixels}")


if __name__ == "__main__":
    main()
