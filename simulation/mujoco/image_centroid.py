"""Robot silhouette statistics in image space (centroid + area) for a MuJoCo free camera.

Coordinates are normalized: u, v in [0, 1] across width / height (v grows downward), area is
the fraction of image pixels covered by the robot. ``project_point`` uses the same pinhole as
``mjv`` free cameras, without clipping, so it stays defined when the robot leaves the frame.
"""

from __future__ import annotations

import mujoco
import numpy as np


def robot_mask(seg: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    gid, typ = seg[..., 0], seg[..., 1]
    body = np.where(gid >= 0, model.geom_bodyid[np.clip(gid, 0, model.ngeom - 1)], 0)
    return (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (body > 0)


def mask_stats(mask: np.ndarray) -> tuple[float, float, float]:
    """(u, v, area); u = v = nan when the mask is empty."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0.0
    return float((xs.mean() + 0.5) / w), float((ys.mean() + 0.5) / h), float(len(xs) / (h * w))


def camera_frame(lookat: np.ndarray, aed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    az, el, dist = np.radians(aed[0]), np.radians(aed[1]), float(aed[2])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.asarray(lookat) - dist * fwd, fwd, right, up


def project_point(p: np.ndarray, lookat: np.ndarray, aed: np.ndarray, fovy_deg: float) -> np.ndarray:
    pos, fwd, right, up = camera_frame(lookat, aed)
    v = np.asarray(p) - pos
    z = max(float(v @ fwd), 1e-3)
    f = 0.5 / np.tan(np.radians(fovy_deg) / 2)
    return np.array([0.5 + f * float(v @ right) / z, 0.5 - f * float(v @ up) / z])


def camera_from_view(cam_pose: np.ndarray, height: np.ndarray, elevation: np.ndarray, dist: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Planar camera pose (x, y, heading) + height + elevation [rad] -> free-camera (lookat, aed[deg])."""
    h, el = cam_pose[:, 2], elevation
    fwd = np.stack([np.cos(el) * np.cos(h), np.cos(el) * np.sin(h), np.sin(el)], -1)
    pos = np.stack([cam_pose[:, 0], cam_pose[:, 1], height], -1)
    aed = np.stack([np.degrees(h), np.degrees(el), np.full(len(h), dist)], -1)
    return pos + dist * fwd, aed


def set_free_camera(cam: mujoco.MjvCamera, lookat: np.ndarray, aed: np.ndarray) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = (float(x) for x in aed)
