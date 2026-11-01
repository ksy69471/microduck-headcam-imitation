"""MuJoCo virtual monitor as a real 3D textured geom.

Streams teacher-video frames into ``model.tex_data`` and re-uploads with
``mjr_uploadTexture`` so both world cameras and the robot observation camera
see the screen in 3D (perspective, occlusion, lighting).

Condition labels (PLAN §7.3):
  A = raw Blender mp4 (no MuJoCo)
  B = fixed world camera looking at the 3D screen
  C = robot observation camera looking at the 3D screen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco.joints import load_microduck_model  # noqa: E402
from simulation.mujoco.play_trajectory import load_sequence, set_state  # noqa: E402

DEFAULT_TEX_W = 256
DEFAULT_TEX_H = 192


def read_video_frames(path: Path, max_frames: int | None = None) -> np.ndarray:
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)
        if max_frames is not None and i + 1 >= max_frames:
            break
    reader.close()
    return np.stack(frames, axis=0)


def _find_body(body, name: str):
    if body.name == name:
        return body
    for child in body.bodies:
        found = _find_body(child, name)
        if found is not None:
            return found
    return None


def _stand_camera_frame(robot_xml: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (eye_xpos, face_look_dir) at STAND.

    Official ``head_camera`` look (-Z) points roughly **-X** (into the head).
    The visual face / mouth tip sits toward **+X**, so for a monitor the duck
    appears to watch we place the screen along the face-forward (+X) axis and
    aim ``obs_camera`` the same way.
    """
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if cid < 0:
        raise RuntimeError("head_camera not found")
    pos = data.cam_xpos[cid].copy()
    # Face-forward: from camera toward mouth_tip if available, else +X.
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    if mid >= 0:
        look = data.site_xpos[mid] - pos
        if np.linalg.norm(look) < 1e-4:
            look = np.array([1.0, 0.0, 0.0])
    else:
        look = np.array([1.0, 0.0, 0.0])
    # Prefer horizontal facing for a standing monitor.
    look[2] = 0.0
    look = look / (np.linalg.norm(look) + 1e-9)
    return pos, look


def _rotate_look(look: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Yaw about world Z, then pitch about camera-right, applied to a look vector."""
    look = look / (np.linalg.norm(look) + 1e-9)
    if abs(yaw_deg) > 1e-9:
        look = Rotation.from_euler("z", yaw_deg, degrees=True).apply(look)
    if abs(pitch_deg) > 1e-9:
        right = np.cross(np.array([0.0, 0.0, 1.0]), look)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        look = Rotation.from_rotvec(np.deg2rad(pitch_deg) * right).apply(look)
    return look / (np.linalg.norm(look) + 1e-9)


def build_screen_model(
    robot_xml: Path,
    screen_distance: float = 0.32,
    cam_out_offset: float = 0.02,
    tex_w: int = DEFAULT_TEX_W,
    tex_h: int = DEFAULT_TEX_H,
    screen_offset_y: float = 0.0,
    screen_offset_z: float = 0.0,
    screen_yaw_deg: float = 0.0,
    screen_pitch_deg: float = 0.0,
    cam_yaw_deg: float = 0.0,
    cam_pitch_deg: float = 0.0,
    cam_fovy: float = 55.0,
) -> tuple[mujoco.MjModel, dict]:
    """Compile scene_walk + 3D monitor texture + outward observation camera.

    The stock ``head_camera`` sits inside visual head meshes (group 2), which
    fully occlude the outside world in offscreen renders. We therefore add
    ``obs_camera`` on ``jaw_soft``, offset slightly along the look direction,
    and recommend hiding geom group 2 when rendering condition C.

    Optional offsets / angles intentionally allow non-perfect framing (lateral
    shift, tilt, FOV) so Condition C stays embodied rather than cropped-A.
    """
    scene = robot_xml.parent / "scene_walk.xml"
    xml = scene if scene.exists() else robot_xml
    cam_pos, face = _stand_camera_frame(xml)
    face = face / (np.linalg.norm(face) + 1e-9)
    # Camera aim can differ from face-forward (imperfect head pose / FOV).
    look = _rotate_look(face, yaw_deg=cam_yaw_deg, pitch_deg=cam_pitch_deg)
    obs_pos = cam_pos + cam_out_offset * face
    # Screen sits roughly ahead, with optional lateral/vertical miss.
    right = np.cross(np.array([0.0, 0.0, 1.0]), face)
    right /= np.linalg.norm(right) + 1e-9
    up = np.array([0.0, 0.0, 1.0])
    screen_pos = (
        cam_pos
        + screen_distance * face
        + screen_offset_y * right
        + screen_offset_z * up
    )
    # Panel normal aims roughly back at the robot, with optional yaw/pitch.
    normal = _rotate_look(-face, yaw_deg=screen_yaw_deg, pitch_deg=screen_pitch_deg)

    # Local offset of obs_camera in jaw_soft at STAND.
    model0 = mujoco.MjModel.from_xml_path(str(xml))
    data0 = mujoco.MjData(model0)
    kid = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model0, data0, kid)
    mujoco.mj_forward(model0, data0)
    bid = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    R = data0.xmat[bid].reshape(3, 3)
    local = R.T @ (obs_pos - data0.xpos[bid])

    spec = mujoco.MjSpec.from_file(str(xml))
    jaw = None
    for b in spec.worldbody.bodies:
        jaw = _find_body(b, "jaw_soft")
        if jaw is not None:
            break
    if jaw is None:
        raise RuntimeError("jaw_soft body not found")
    # Build obs_camera orientation so local -Z aligns with aim `look`.
    # Camera axes: X right, Y up, -Z look.
    z_axis = -look  # camera -Z = look  =>  Z = -look
    # Prefer world-up for camera up, then orthonormalize.
    up_w = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up_w, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    # Camera rotation in world, then express as local quat relative to jaw_soft.
    R_cam_world = np.column_stack([x_axis, y_axis, z_axis])
    R_jaw = data0.xmat[bid].reshape(3, 3)
    R_local = R_jaw.T @ R_cam_world
    q_local = Rotation.from_matrix(R_local).as_quat()  # xyzw
    cam = jaw.add_camera()
    cam.name = "obs_camera"
    cam.pos = local.tolist()
    cam.quat = [float(q_local[3]), float(q_local[0]), float(q_local[1]), float(q_local[2])]
    cam.fovy = float(cam_fovy)

    tex = spec.add_texture()
    tex.name = "virtual_screen_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width = int(tex_w)
    tex.height = int(tex_h)
    tex.nchannel = 3
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1[:] = [0.02, 0.02, 0.02]

    mat = spec.add_material()
    mat.name = "virtual_screen_mat"
    mat.texuniform = False
    mat.emission = 1.0  # self-lit panel so content reads in world + obs views
    mat.specular = 0.0
    mat.shininess = 0.0
    mat.rgba[:] = [1.0, 1.0, 1.0, 1.0]
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "virtual_screen_tex"

    # Body +X faces the robot camera (normal toward robot).
    x_axis = normal / (np.linalg.norm(normal) + 1e-9)
    y_axis = np.cross(np.array([0.0, 0.0, 1.0]), x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(np.array([0.0, 1.0, 0.0]), x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    quat = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()
    quat_wxyz = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]

    body = spec.worldbody.add_body(name="virtual_monitor", pos=screen_pos.tolist(), quat=quat_wxyz)
    body.add_geom(
        name="monitor_bezel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, 0.22, 0.16],
        rgba=[0.05, 0.05, 0.05, 1],
        contype=0,
        conaffinity=0,
        group=0,
    )
    # Front panel (faces robot). Box UV maps +X face; keep thin so only that face reads.
    panel = body.add_geom(
        name="virtual_screen",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0010, 0.19, 0.135],
        pos=[0.0115, 0.0, 0.0],
        contype=0,
        conaffinity=0,
        group=0,
    )
    panel.material = "virtual_screen_mat"
    # Back panel so third-person views also show the stream (mirrored along body X).
    panel_back = body.add_geom(
        name="virtual_screen_back",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0010, 0.19, 0.135],
        pos=[-0.0115, 0.0, 0.0],
        contype=0,
        conaffinity=0,
        group=0,
    )
    panel_back.material = "virtual_screen_mat"
    body.add_site(name="screen_center", pos=[0.0115, 0, 0], size=[0.005, 0, 0])

    # Ensure offscreen buffer is large enough for Renderer.
    # (scene_walk already sets some visual defaults; Renderer validates vs offwidth)

    model = spec.compile()
    meta = {
        "robot_xml": str(xml),
        "screen_distance": screen_distance,
        "cam_out_offset": cam_out_offset,
        "screen_offset_y": screen_offset_y,
        "screen_offset_z": screen_offset_z,
        "screen_yaw_deg": screen_yaw_deg,
        "screen_pitch_deg": screen_pitch_deg,
        "cam_yaw_deg": cam_yaw_deg,
        "cam_pitch_deg": cam_pitch_deg,
        "cam_fovy": cam_fovy,
        "screen_pos": screen_pos.tolist(),
        "obs_camera_pos_stand": obs_pos.tolist(),
        "face_dir_stand": face.tolist(),
        "look_dir_stand": look.tolist(),
        "tex_w": tex_w,
        "tex_h": tex_h,
        "texture": "virtual_screen_tex",
        "obs_camera": "obs_camera",
        "note": (
            "Stock head_camera is occluded by visual meshes (group 2). "
            "Condition C uses obs_camera with geomgroup[2]=0."
        ),
    }
    return model, meta


def upload_frame(model: mujoco.MjModel, renderer: mujoco.Renderer, frame_rgb: np.ndarray, tex_name: str = "virtual_screen_tex") -> None:
    """Write one RGB frame into the named texture and upload to the GPU."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name)
    if tid < 0:
        raise RuntimeError(f"texture not found: {tex_name}")
    w, h = int(model.tex_width[tid]), int(model.tex_height[tid])
    adr = int(model.tex_adr[tid])
    img = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
    if img.shape[-1] == 4:
        img = img[..., :3]
    # Pack top-down: box +X face UV already matches camera upright after emission lit.
    # (flipud made the streamed teacher video appear upside-down on obs_camera.)
    model.tex_data[adr : adr + w * h * 3] = np.asarray(img, dtype=np.uint8).reshape(-1)
    renderer._gl_context.make_current()
    mujoco.mjr_uploadTexture(model, renderer._mjr_context, tid)


def observe_video_on_screen(
    seq_dir: Path,
    video_path: Path,
    out_dir: Path,
    width: int = 256,
    height: int = 256,
    hold_pose: bool = True,
    condition: str = "C",
    screen_distance: float = 0.32,
    also_world: bool = True,
) -> dict:
    md = load_microduck_model()
    robot_xml = md.xml_path if md.xml_path.name.startswith("scene") else md.xml_path.parent / "robot_walk.xml"
    model, screen_meta = build_screen_model(robot_xml, screen_distance=screen_distance)
    data = mujoco.MjData(model)
    seq = load_sequence(seq_dir)
    frames = read_video_frames(video_path)
    n = min(len(frames), seq["q"].shape[0])

    obs_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "obs_camera")
    if obs_cam < 0:
        raise RuntimeError("obs_camera missing")

    renderer = mujoco.Renderer(model, height=height, width=width)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = np.zeros((n, height, width, 3), dtype=np.uint8)
    world = np.zeros((n, height, width, 3), dtype=np.uint8) if also_world else None

    for t in range(n):
        q = seq["q"][0] if hold_pose else seq["q"][t]
        root_pos = seq["root_pos"][0] if hold_pose else seq["root_pos"][t]
        root_quat = seq["root_quat"][0] if hold_pose else seq["root_quat"][t]
        set_state(model, data, md, q, root_pos, root_quat, seq["joint_names"])
        # Prefer STAND joint targets when holding for a stable watch pose.
        if hold_pose and "STAND" in md.keyframes:
            stand = np.array(md.keyframes["STAND"], dtype=np.float64)
            # keyframes ctrl is in actuator order == joint_names order for walk model
            if stand.shape[0] == len(seq["joint_names"]):
                set_state(model, data, md, stand, root_pos, root_quat, seq["joint_names"])
        mujoco.mj_forward(model, data)

        upload_frame(model, renderer, frames[t])

        if condition == "C":
            renderer._scene_option.geomgroup[:] = 1
            renderer._scene_option.geomgroup[2] = 0  # hide visual meshes that occlude lens
            renderer.update_scene(data, camera=obs_cam)
            obs[t] = renderer.render().copy()
        else:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = np.array(screen_meta["screen_pos"])
            camera.distance = 0.55
            camera.azimuth = 180
            camera.elevation = -8
            renderer._scene_option.geomgroup[:] = 1
            renderer.update_scene(data, camera=camera)
            obs[t] = renderer.render().copy()

        if also_world:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            sp = np.array(screen_meta["screen_pos"], dtype=np.float64)
            # Side view: robot faces +X toward the lit 3D monitor.
            camera.lookat[:] = 0.45 * sp + np.array([0.0, 0.0, 0.08])
            camera.distance = 0.9
            camera.azimuth = 85
            camera.elevation = -12
            renderer._scene_option.geomgroup[:] = 1  # show robot meshes
            renderer.update_scene(data, camera=camera)
            world[t] = renderer.render().copy()

    renderer.close()
    np.savez_compressed(out_dir / "robot_camera_obs.npz", rgb=obs, fps=np.array([seq["fps"]]))
    preview = out_dir / "robot_camera_obs.mp4"
    imageio.mimsave(preview, list(obs), fps=float(seq["fps"]))
    world_path = None
    if world is not None:
        world_path = out_dir / "world_view_with_screen.mp4"
        imageio.mimsave(world_path, list(world), fps=float(seq["fps"]))
        # side-by-side explainability clip
        combo = []
        for t in range(n):
            left = world[t]
            right = obs[t]
            panel = np.concatenate([left, right], axis=1)
            cv2.rectangle(panel, (0, 0), (left.shape[1], 28), (20, 20, 20), -1)
            cv2.rectangle(panel, (left.shape[1], 0), (panel.shape[1], 28), (20, 20, 20), -1)
            cv2.putText(panel, "World: 3D virtual monitor", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                panel,
                "Robot obs_camera (group2 hidden)",
                (left.shape[1] + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            combo.append(panel)
        combo_path = out_dir / "screen3d_world_and_obs.mp4"
        imageio.mimsave(combo_path, combo, fps=float(seq["fps"]))
    else:
        combo_path = None

    meta = {
        "condition": condition,
        "n_frames": n,
        "width": width,
        "height": height,
        "video_source": str(video_path),
        "sequence": str(seq_dir),
        "hold_pose": hold_pose,
        "method": "mujoco_tex_data_mjr_uploadTexture_3d_monitor",
        "preview": str(preview),
        "world_view": str(world_path) if world_path else None,
        "combo_view": str(combo_path) if combo_path else None,
        "screen": screen_meta,
    }
    (out_dir / "observation_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--condition", default="C", choices=["B", "C"])
    p.add_argument("--screen-distance", type=float, default=0.32)
    p.add_argument("--no-world", action="store_true")
    args = p.parse_args()
    meta = observe_video_on_screen(
        args.seq,
        args.video,
        args.out,
        condition=args.condition,
        screen_distance=args.screen_distance,
        also_world=not args.no_world,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
