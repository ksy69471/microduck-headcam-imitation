# Head-camera walk experiments (reproduction tree)

小型二足ロボットのシミュレータ上で、歩行クリップの生成、頭部カメラ映像から動作を推定する学習、速度から歩行指令への変換、報酬の項を変えた PPO トラッカー学習をやり直すためのソースです。課題提出文や図のキャプションは含めていません。手順の英語表記は下のとおりです。

Code to regenerate simulated walking clips for a small biped, train a network
that maps a head-camera video to motion, turn predicted speed into walk
commands, and train a PPO tracker while changing the reward terms.

This directory is meant to be its own git repository. It does not contain
datasets, checkpoints, videos, ONNX files, or third-party meshes.

## Licenses

| What | Terms |
|------|--------|
| Files in this tree | Apache-2.0 (`LICENSE`, `NOTICE`) |
| `microduck` / `microduck_rl` code and MJCF | Apache-2.0, upstream |
| STL meshes in `microduck_rl` | CC BY-NC-SA 4.0 (non-commercial, share-alike). Not included here. |
| ONNX policies | See the Hugging Face model card. Not included here. |
| `bam` actuator library | See the Rhoban repository. Not included here. |

Details: [third_party/NOTICES.md](third_party/NOTICES.md).

Do not commit downloaded meshes, ONNX weights, or generated datasets.

## Setup

```bash
python3.12 -m venv work/local/venv
source work/local/venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install "better-actuator-models @ git+https://github.com/Rhoban/bam.git@mjlab_frictionloss"

bash scripts/fetch_third_party.sh
bash scripts/fetch_policies.sh
export MUJOCO_GL=egl
```

`fetch_third_party.sh` clones into `readonly/`, which is gitignored.
`fetch_policies.sh` writes ONNX files under `work/local/scratch/policies/`.

CPU PyTorch is enough for a short tracker run. The video model is much faster
on a GPU.

## 1. Record walks and head-camera views

The recorder rolls out the official walking ONNX policy inside MuJoCo, saves
joint and root trajectories, and renders the same clip as seen by a camera in
the robot head while a monitor in the scene plays the teacher video.

```bash
python scripts/build_ego_dataset.py \
  --out datasets/motions/POL-WALK-EGO \
  --n 120 --seed-base 1000 --workers 8
python scripts/make_clip_lists.py --data datasets/motions/POL-WALK-EGO
python scripts/compute_image_centroids.py --data datasets/motions/POL-WALK-EGO
```

`--n 120` is a smoke size. The runs behind the longer training used on the
order of several hundred extra training clips (`--extend` on
`build_ego_dataset.py`).

## 2. Video model

`--source C` reads the head-camera frames (the monitor seen by the robot).
`--source A` reads the teacher video directly.

```bash
python training/train_v2m_ego.py \
  --data datasets/motions/POL-WALK-EGO \
  --variant hand_floor --source C --with-view \
  --out work/local/runs/v2m_hand_floor \
  --time-budget-s 9000 --eval-every 2
python evaluation/evaluate_v2m_ego.py \
  --runs work/local/runs/v2m_hand_floor \
  --out work/local/runs/v2m_test.json
```

## 3. Inverse command model and closed-loop walk

The inverse model is fit on random walks of the same ONNX walking policy
(speed to command). It is not fit on the teacher joint angles.

```bash
python evaluation/inverse_body_model.py \
  --n 3000 --steps 8000 \
  --out work/local/runs/inverse_nn
python scripts/export_ego_references.py \
  --ckpt work/local/runs/v2m_hand_floor/best.pt \
  --root est --img-target est \
  --out datasets/motions/POL-WALK-EGO-EST
python evaluation/command_imitation_ego.py \
  --out work/local/runs/commands \
  --preds work/local/runs/v2m_preds_v2m_hand_floor.npz \
  --inverse-nn work/local/runs/inverse_nn/model.pt \
  --est-targets datasets/motions/POL-WALK-EGO-EST \
  --workers 8
```

`command_imitation_ego.py` compares several command sources (mean command,
predicted command, optional search). See `--help` for the condition names.

## 4. Tracker reward sweep

The tracker is trained from scratch on the recorded joint and root references
(`--root-cond`), without the ONNX policy in the loop (`--no-onnx-teacher`).
The grid varies joint tolerance, a forward-travel bonus, and an alive bonus.

```bash
python scripts/grid_search_tracking_rewards.py --help
python evaluation/eval_tracking_reward_study.py --help
```

Checkpoints and logs go under `work/local/runs/`. Select a checkpoint on the
validation clip list and report numbers on the test list. Do not tune on the
test split.

## Layout

| Path | Role |
|------|------|
| `simulation/mujoco/capture_ego.py` | ONNX walk rollout and cameras |
| `observation/virtual_screen.py` | Monitor playback into the head camera |
| `training/train_v2m_ego.py` | Video-to-motion training |
| `evaluation/inverse_body_model.py` | Speed-to-command fit |
| `evaluation/command_imitation_ego.py` | Execute commands with the walking policy |
| `training/train_tracking_policy.py` | PPO tracker |
| `scripts/grid_search_tracking_rewards.py` | Reward grid |

## What is not in this tree

- Course write-ups, slide text, and figure captions
- Personal machine paths and local virtualenvs
- Rendered datasets, checkpoints, and comparison videos
- Upstream robot meshes and ONNX policies
