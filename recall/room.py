"""Part E bridge: the corner camera registered inside the scanned room (CAMERA.md §7, handoffAB.md).

Reads what the setup tools (tools/scan_to_room.py, register_camera.py) write to data/room/, with
their conventions (tools/roomlib.py): world in metres, Y up, floor at y = 0; camera.json holds
OpenCV intrinsics (K, dist) and the world->camera pose X_cam = R·X_world + t (camera x right,
y down, z forward). This is the runtime half handoffAB.md gives B: xyz_map, pixel_to_world,
floor_hit, world_to_pixel.
"""
import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

XYZ_MAP, XYZ_KEY = "xyz_map.npy", "xyz_map.sha1"  # the key: hash of the camera.json + room.glb it came from
FLOOR_TOL_M = 0.05
PLANE_BIN_M = 0.1  # flat surfaces are grouped into planes by their offset, in bins this wide
ORTHO_TOL = 1e-3  # camera.json R further than this from a rotation is a broken file, not rounding
EDGE_M = 0.05  # an xyz_map cell spanning more than this straddles a depth edge: don't blend the surfaces
MIN_RANGE_M = 0.2  # rays start this far out, past the webcam and its mount if they are in the scan
NEAR_M = 0.05  # in_lens: points closer in front of the lens than this aren't drawn
# cv2.undistortPoints' default (5 iterations) is off by pixels at the image edges with real lens
# distortion; iterate until reprojecting the result lands back on the input pixel.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
UNDISTORT_PX = 0.05  # a ray that reprojects further than this from its pixel didn't converge
# Camera axes OpenCV (x right, y down, looks down +z) <-> GL/ARKit/three.js (x right, y up, looks down -z).
CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # roomlib.FLIP_YZ


def undistort(uv, K, dist):
    """Pixels (N, 2) -> normalized image coordinates (N, 2) with lens distortion removed.

    NaN where the lens model has no inverse: a badly fitted dist polynomial can fold over before the
    image corners, and undistortPoints then quietly returns a ray that doesn't reproject to the pixel.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    if not len(uv):
        return uv.copy()
    xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist, R=None, P=None, criteria=UNDISTORT_CRITERIA)
    xy = xy.reshape(-1, 2)
    back = project(np.c_[xy, np.ones(len(xy))], K, dist, np.eye(3), np.zeros(3))
    xy[np.linalg.norm(back - uv, axis=1) > UNDISTORT_PX] = np.nan
    return xy


def pixel_rays(uv, K, dist, R, t):
    """World-frame rays through pixels: (camera centre (3,), unit directions (N, 3))."""
    xy = undistort(uv, K, dist)
    d = np.c_[xy, np.ones(len(xy))] @ R  # each row is Rᵀ·[x, y, 1]
    return -R.T @ t, d / np.linalg.norm(d, axis=1, keepdims=True)


def project(points, K, dist, R, t):
    """World points (N, 3) -> pixels (N, 2) through the lens model."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(np.asarray(R, np.float64))[0], np.asarray(t, np.float64), K, dist)
    return uv.reshape(-1, 2)


def orthonormalize(M):
    """Nearest orthonormal matrix (a rotation stays a rotation, a reflection a reflection)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, np.float64))
    return U @ Vt


def floor_points(origin, dirs):
    """Ray ∩ plane y = 0 per ray; NaN where the ray doesn't reach the floor in front of the camera."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -origin[1] / dirs[:, 1]
    p = origin + s[:, None] * dirs
    p[:, 1] = 0.0
    p[~(np.isfinite(s) & (s > 0))] = np.nan
    return p


def opencv_to_gl(R, t):
    """OpenCV world->camera (R, t) -> 4x4 camera->world pose in GL/ARKit camera axes."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).T @ CV_TO_GL
    T[:3, 3] = -np.asarray(R).T @ np.asarray(t).reshape(3)
    return T


def gl_to_opencv(T):
    """4x4 camera->world pose in GL/ARKit camera axes -> OpenCV world->camera (R, t)."""
    T = np.asarray(T, np.float64)
    R = CV_TO_GL @ T[:3, :3].T
    return R, -R @ T[:3, 3]


def read_mesh(path):
    """room.glb -> (vertices (V, 3) float64, triangles (F, 3) int), loaded the way the setup tools
    load it (roomlib.Mesh.load): trimesh, every node merged with its transform baked in."""
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if not len(getattr(mesh, "faces", ())):
        raise ValueError(f"{path}: no triangles (a point cloud, or a compressed glTF?)")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)


def planes(vertices, triangles, axis):
    """Flat surfaces facing ±axis, grouped by their offset along it: (offset (m), area (m²)) per plane,
    in offset order."""
    tri = np.asarray(vertices, np.float64)[np.asarray(triangles)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    facing = np.abs(n[:, axis]) > 0.95 * 2 * area  # unit normal's component along the axis > 0.95
    offset, area = tri[facing, :, axis].mean(1), area[facing]
    bins = np.unique(np.round(offset / PLANE_BIN_M), return_inverse=True)[1]
    total = np.bincount(bins, area)
    return np.bincount(bins, area * offset) / total, total


def mesh_frame_problems(vertices, triangles):
    """Reasons the mesh may not be in the world frame (Y-up, floor at y = 0); empty if it looks right.

    A box room looks the same along every axis, so the axis test only flags a largest horizontal
    plane far smaller than a wall; tools/scan_to_room.py checks the scan against its own LiDAR depth.
    """
    found = [planes(vertices, triangles, a) for a in range(3)]
    biggest = [area.max(initial=0.0) for _, area in found]
    problems = []
    if biggest[1] < 0.5 * max(biggest):
        a = int(np.argmax(biggest))
        problems.append(f"largest flat surface faces ±{'XYZ'[a]}, not ±Y: mesh may not be Y-up")
    y, area = found[1]
    if len(area):
        floor_y = y[np.searchsorted(np.cumsum(area), 0.05 * area.sum())]  # lowest 5% of horizontal area
        if abs(floor_y) > FLOOR_TOL_M:
            problems.append(f"floor is at y = {floor_y:.2f} m, expected 0")
    return problems


def _bilinear(xyz, u, v):
    h, w = xyz.shape[:2]
    if not (-0.5 <= u <= w - 0.5 and -0.5 <= v <= h - 0.5):
        return np.full(3, np.nan)
    u, v = min(max(u, 0.0), w - 1.0), min(max(v, 0.0), h - 1.0)
    u0, v0 = min(int(u), w - 2), min(int(v), h - 2)
    fu, fv = u - u0, v - v0
    q = xyz[v0:v0 + 2, u0:u0 + 2].astype(np.float64)
    if np.isnan(q).any() or np.ptp(q.reshape(-1, 3), 0).max() > EDGE_M:  # hit/miss or depth edge: nearest
        return xyz[int(round(v)), int(round(u))].astype(np.float64)
    wts = np.array([[(1 - fu) * (1 - fv), fu * (1 - fv)], [(1 - fu) * fv, fu * fv]])
    return (q * wts[..., None]).sum((0, 1))


def _point(p):
    return None if np.isnan(p).any() else tuple(float(c) for c in p)


class Room:
    """The corner camera inside the room mesh. All reads are thread-safe.

    source: "mesh" (exact ray casts into room.glb), "xyz_map" (room.glb unreadable here, but
    xyz_map.npy was built from it) or "floor" (no mesh: everything lands on y = 0).
    mesh_error: why room.glb couldn't be read, if it couldn't.
    """

    def __init__(self, camera, vertices=None, triangles=None, room_dir=None, mesh_error=None):
        if "R" not in camera or "t" not in camera:
            raise ValueError("camera.json has no pose (R, t): run uv run tools/register_camera.py")
        self.width, self.height = int(camera["width"]), int(camera["height"])
        self.K = np.array(camera["K"], np.float64)
        self.dist = np.array(camera.get("dist") or [0.0] * 5, np.float64)
        R = np.array(camera["R"], np.float64)
        self.R = orthonormalize(R)  # rays and projectPoints (via Rodrigues) must use the same rotation
        if np.abs(self.R - R).max() > ORTHO_TOL:
            log.warning("camera.json R is not a rotation (off by %.1e): using the nearest one; re-run "
                        "uv run tools/register_camera.py", np.abs(self.R - R).max())
        self.t = np.array(camera["t"], np.float64).reshape(3)
        self.center = -self.R.T @ self.t
        self.dir = Path(room_dir) if room_dir else None
        self.vertices, self.triangles, self.mesh_error = vertices, triangles, mesh_error
        self._scene, self._xyz = None, None
        self._cast_lock, self._map_lock = threading.Lock(), threading.Lock()
        # Computed now: a map built later must not be saved under a camera.json rewritten meanwhile.
        self._key = self._inputs_key()
        u, v = np.meshgrid(np.linspace(0, self.width - 1, 81), np.linspace(0, self.height - 1, 46))
        xy = undistort(np.c_[u.ravel(), v.ravel()], self.K, self.dist)
        lost = np.isnan(xy[:, 0]).mean()
        # How far off-axis the image reaches (tan of the angle, corners included), with some slack.
        self.max_r = 1.05 * float(np.nanmax(np.hypot(*xy.T), initial=0.0))
        if lost:
            log.warning("lens model (dist) has no inverse at %.1f%% of the image, near its corners: those pixels "
                        "map to None. Recalibrate with the board in the image corners", 100 * lost)
        if vertices is not None:
            for p in mesh_frame_problems(vertices, triangles):
                log.warning("room mesh: %s", p)
            self._scene = self._make_scene(vertices, triangles)
            self.source = "mesh"
        else:
            self._xyz = self._cached_map()
            self.source = "floor" if self._xyz is None else "xyz_map"
        log.info("room: camera at %s, source=%s", np.round(self.center, 3).tolist(), self.source)

    @classmethod
    def load(cls, room_dir) -> "Room":
        """camera.json (required, with a pose) + room.glb (optional) from room_dir."""
        room_dir = Path(room_dir)
        camera = json.loads((room_dir / "camera.json").read_text())
        glb = room_dir / "room.glb"
        vertices = triangles = error = None
        if glb.exists():
            try:
                vertices, triangles = read_mesh(glb)
            except Exception as e:  # any unreadable mesh: fall back rather than stop the demo
                error = f"{type(e).__name__}: {e}"
                log.warning("could not load %s (%s); falling back to %s or the floor plane", glb, error, XYZ_MAP)
        return cls(camera, vertices, triangles, room_dir, error)

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float, float] | None:
        """World point seen at pixel (u, v): the first mesh hit (or the source fallback), None if no hit."""
        if self.source == "mesh":
            return _point(self._cast(*pixel_rays([[u, v]], self.K, self.dist, self.R, self.t))[0])
        if self.source == "xyz_map":
            return _point(_bilinear(self._xyz, u, v))
        return self.floor_hit(u, v)

    def floor_hit(self, u: float, v: float) -> tuple[float, float, float] | None:
        """Ray through pixel (u, v) ∩ floor plane y = 0 (exact), or None if it misses in front."""
        return _point(floor_points(*pixel_rays([[u, v]], self.K, self.dist, self.R, self.t))[0])

    def world_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float]:
        """World point -> pixel through the lens, for overlays (doesn't check the point is in front)."""
        u, v = project([[x, y, z]], self.K, self.dist, self.R, self.t)[0]
        return float(u), float(v)

    def in_lens(self, points) -> np.ndarray:
        """Which world points (N, 3) project faithfully: in front of the camera and no further off-axis
        than the image reaches. Past that the dist polynomial can fold back, and world_to_pixel puts a
        point far off to the side inside the image."""
        cam = np.asarray(points, np.float64).reshape(-1, 3) @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.hypot(cam[:, 0], cam[:, 1]) / cam[:, 2]
        return (cam[:, 2] > NEAR_M) & (r <= self.max_r)

    def surface_distance(self, points) -> np.ndarray:
        """Distance (m) from each world point (N, 3) to the nearest mesh surface. Needs the mesh."""
        import open3d as o3d

        pts = o3d.core.Tensor(np.asarray(points, np.float32).reshape(-1, 3))
        with self._cast_lock:
            return self._scene.compute_distance(pts).numpy().astype(np.float64)

    @property
    def xyz_map(self) -> np.ndarray:
        """H×W×3 float32 world point per pixel centre, NaN = no hit. Cached to xyz_map.npy."""
        with self._map_lock:
            if self._xyz is None and self.source == "mesh":
                self._xyz = self._cached_map()
            if self._xyz is None:
                self._xyz = self._build_xyz_map()
                if self.source == "mesh" and self._key:  # a floor-only map is never cached as the mesh's
                    self._save_map(self._xyz)
            return self._xyz

    def _build_xyz_map(self):
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        origin, dirs = pixel_rays(np.c_[u.ravel(), v.ravel()], self.K, self.dist, self.R, self.t)
        p = self._cast(origin, dirs) if self._scene is not None else floor_points(origin, dirs)
        log.info("built xyz_map (%d of %d pixels hit)", np.isfinite(p[:, 0]).sum(), len(p))
        return p.reshape(self.height, self.width, 3).astype(np.float32)

    def _inputs_key(self):
        """Hash of camera.json + room.glb; None without both (only a mesh's map is worth caching)."""
        inputs = [self.dir / "camera.json", self.dir / "room.glb"] if self.dir else []
        if not inputs or not all(p.exists() for p in inputs):
            return None
        h = hashlib.sha1()
        for p in inputs:
            h.update(p.read_bytes())
        return h.hexdigest()

    def _cached_map(self):
        """xyz_map.npy if it was built from this camera.json and room.glb (same hash, neither newer), else None."""
        if not self._key:
            return None
        path, key = self.dir / XYZ_MAP, self.dir / XYZ_KEY
        if not (path.exists() and key.exists()):
            return None
        newer = any((self.dir / p).stat().st_mtime > path.stat().st_mtime for p in ("camera.json", "room.glb"))
        if newer or key.read_text().strip() != self._key:
            log.info("%s is stale", path)
            return None
        xyz = np.load(path)
        return xyz if xyz.shape == (self.height, self.width, 3) else None

    def _save_map(self, xyz):
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")  # a unique name: another process may save too
        with os.fdopen(fd, "wb") as f:
            np.save(f, xyz)
        os.replace(tmp, self.dir / XYZ_MAP)
        (self.dir / XYZ_KEY).write_text(self._key)  # last: a crash before this leaves the map stale, not wrong

    @staticmethod
    def _make_scene(vertices, triangles):
        import open3d as o3d

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(triangles, np.uint32)))
        return scene

    def _cast(self, origin, dirs):
        import open3d as o3d

        start = np.broadcast_to(origin, dirs.shape) + MIN_RANGE_M * dirs
        rays = np.hstack([start, dirs]).astype(np.float32)
        with self._cast_lock:
            t_hit = self._scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        p = origin + (MIN_RANGE_M + t_hit)[:, None] * dirs
        p[~np.isfinite(t_hit)] = np.nan
        return p
