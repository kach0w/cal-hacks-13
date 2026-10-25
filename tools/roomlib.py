"""Shared geometry for the room tools (handoffAB.md).

Conventions, everywhere in data/room/:
    world   metres, Y up, floor at y = 0, X and Z along the walls, room at x, z >= 0
    camera  OpenCV: x right, y down, z forward
    pose    X_cam = R @ X_world + t   (what cv2.solvePnP returns)
"""

import json
from pathlib import Path

import cv2
import numpy as np

from printed import TAG_DICT, TAG_IDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "room"
CAMERAS = ROOT / "data" / "cameras"  # one lens calibration per webcam, reused in every room

# Camera axes, ARKit/OpenGL (x right, y up, looking down -z) <-> OpenCV. It is its own inverse.
FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    """Write to a temp file, then rename, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    print(f"wrote {path}")


def marker_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICT))
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(detector, image):
    """{id: 4x2 corners} for our printed ids, in OpenCV's order: TL, TR, BR, BL."""
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in TAG_IDS}


def camera_center(R, t):
    return -np.asarray(R).T @ np.asarray(t).ravel()


def project(K, dist, R, t, pts):
    """World points -> pixels."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(np.asarray(R, np.float64))[0],
                              np.asarray(t, np.float64).ravel(), np.asarray(K, np.float64),
                              None if dist is None else np.asarray(dist, np.float64))
    return uv.reshape(-1, 2)


def pixel_rays(K, dist, R, t, uv):
    """Pixels -> world rays (origins, unit directions). Undistorts first."""
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(uv, np.asarray(K, np.float64),
                               None if dist is None else np.asarray(dist, np.float64)).reshape(-1, 2)
    d = np.column_stack([norm, np.ones(len(norm))]) @ np.asarray(R)  # rows of R^T @ d_cam
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.broadcast_to(camera_center(R, t), d.shape).copy(), d


def floor_hit(K, dist, R, t, uv):
    """Where pixel rays meet the floor plane y = 0. NaN where they never do."""
    o, d = pixel_rays(K, dist, R, t, uv)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o[:, 1] / d[:, 1]
    pts = o + s[:, None] * d
    pts[~(s > 0)] = np.nan
    return pts


def solve_pose(object_pts, image_pts, K, dist):
    """Camera pose from known 3D points and their pixels -> (R, t, rms_px, inlier indices)."""
    obj = np.asarray(object_pts, np.float64).reshape(-1, 3)
    img = np.asarray(image_pts, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)
    dist = None if dist is None else np.asarray(dist, np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=8.0, iterationsCount=500
    )
    if not ok or inliers is None:
        raise ValueError("solvePnP found no consistent pose")
    idx = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K, dist, rvec, tvec)
    R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
    err = project(K, dist, R, t, obj[idx]) - img[idx]
    return R, t, float(np.sqrt((err ** 2).sum(axis=1).mean())), idx


def plane_rms(pts):
    """RMS distance of points from their best-fit plane. ~0 means they're all on one plane."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    return float(np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(pts)))


def transform(T, pts):
    pts = np.asarray(pts, np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


class Mesh:
    """A triangle mesh to cast rays into (Open3D RaycastingScene)."""

    def __init__(self, vertices, faces):
        import open3d as o3d  # heavy, so only the tools that need a mesh pay for it

        self._o3d = o3d
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.core.Tensor(self.vertices.astype(np.float32)),
                                  o3d.core.Tensor(self.faces.astype(np.uint32)))

    @classmethod
    def load(cls, path):
        import trimesh

        m = trimesh.load(path, force="mesh")  # merges every node, baking its transform
        return cls(m.vertices, m.faces)

    def cast(self, origins, dirs):
        """Distance along each unit ray to the first hit; inf on a miss."""
        return self.cast_detail(origins, dirs)[0]

    def cast_detail(self, origins, dirs):
        """(distance, triangle index, barycentric uv) per ray; a miss has distance inf."""
        rays = np.hstack([origins, dirs]).astype(np.float32)
        hit = self._scene.cast_rays(self._o3d.core.Tensor(rays))
        return (hit["t_hit"].numpy().astype(np.float64), hit["primitive_ids"].numpy().astype(np.int64),
                hit["primitive_uvs"].numpy().astype(np.float64))

    def hit_points(self, origins, dirs):
        t = self.cast(origins, dirs)
        pts = origins + t[:, None] * dirs
        pts[~np.isfinite(t)] = np.nan
        return pts

    def visible(self, eye, pts, tol=0.03):
        """Which points the eye can actually see (nothing in the mesh in front of them)."""
        d = np.asarray(pts) - eye
        dist = np.linalg.norm(d, axis=1)
        return self.cast(np.broadcast_to(eye, d.shape), d / dist[:, None]) > dist - tol


def bring_to_front():
    """macOS opens a script's windows behind the terminal; raise them. No-op elsewhere."""
    try:
        from AppKit import NSApplication

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    except ImportError:
        pass

