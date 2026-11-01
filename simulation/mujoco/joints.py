"""Microduck joint catalog loaded from the official MJCF (no hard-coded guesses)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = (
    REPO_ROOT
    / "readonly"
    / "microduck_rl"
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
    / "scene_walk.xml"
)
FALLBACK_XML = DEFAULT_XML.parent / "robot_walk.xml"


@dataclass(frozen=True)
class JointInfo:
    name: str
    qposadr: int
    dofadr: int
    range: tuple[float, float]
    axis: tuple[float, float, float]
    body: str
    actuator_id: int | None


@dataclass(frozen=True)
class MicroduckModel:
    xml_path: Path
    model: mujoco.MjModel
    servo_joints: tuple[JointInfo, ...]
    joint_names: tuple[str, ...]
    keyframes: dict[str, list[float]]

    @property
    def n_servo(self) -> int:
        return len(self.servo_joints)


def load_microduck_model(xml_path: Path | None = None) -> MicroduckModel:
    path = Path(xml_path) if xml_path else DEFAULT_XML
    if not path.exists():
        path = FALLBACK_XML
    model = mujoco.MjModel.from_xml_path(str(path))

    name_to_act: dict[str, int] = {}
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n:
            name_to_act[n] = i

    servos: list[JointInfo] = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name or name.startswith("passive_"):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[i]))
        servos.append(
            JointInfo(
                name=name,
                qposadr=int(model.jnt_qposadr[i]),
                dofadr=int(model.jnt_dofadr[i]),
                range=(float(model.jnt_range[i][0]), float(model.jnt_range[i][1])),
                axis=tuple(float(x) for x in model.jnt_axis[i]),
                body=body or "",
                actuator_id=name_to_act.get(name),
            )
        )

    # Prefer actuator order when available (official ctrl layout).
    if model.nu == len(servos) and all(s.actuator_id is not None for s in servos):
        servos = sorted(servos, key=lambda s: s.actuator_id or 0)

    keyframes: dict[str, list[float]] = {}
    for i in range(model.nkey):
        kn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) or f"key_{i}"
        keyframes[kn] = model.key_ctrl[i].tolist()

    return MicroduckModel(
        xml_path=path,
        model=model,
        servo_joints=tuple(servos),
        joint_names=tuple(s.name for s in servos),
        keyframes=keyframes,
    )


def dump_model_info(out_path: Path, xml_path: Path | None = None) -> dict:
    md = load_microduck_model(xml_path)
    payload = {
        "xml_path": str(md.xml_path),
        "n_servo": md.n_servo,
        "joint_names": list(md.joint_names),
        "joints": [asdict(j) for j in md.servo_joints],
        "keyframes_ctrl": md.keyframes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = REPO_ROOT / "work" / "local" / "scratch" / "joint_catalog.json"
    info = dump_model_info(out)
    print(json.dumps({"n_servo": info["n_servo"], "joint_names": info["joint_names"]}, indent=2))
