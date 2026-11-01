# Third-party components

Nothing in this folder vendors those components. `scripts/fetch_third_party.sh`
and `scripts/fetch_policies.sh` only download them into local directories that
are gitignored.

## Robot description and simulator glue

- https://github.com/pollen-robotics/microduck
- https://github.com/pollen-robotics/microduck_rl

Code and MJCF in those repositories are Apache-2.0 (see each `LICENSE`).

STL meshes under `microduck_rl` (`src/mjlab_microduck/robot/microduck/assets/*.stl`
and the XL330 test-bench meshes) are **CC BY-NC-SA 4.0**, copyright Pollen
Robotics:

- https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

That license is non-commercial and share-alike. Do not copy the meshes into
this repository. If you redistribute a modified mesh, keep the attribution,
state the changes, and use CC BY-NC-SA 4.0.

## Actuator model

`simulation/mujoco/stand_policy_demo.py` imports `bam` (package name
`better-actuator-models`):

- https://github.com/Rhoban/bam (branch `mjlab_frictionloss`, as pinned by
  `microduck_rl`)

Use the license that ships in that repository. This tree does not include it.

## ONNX policies

`scripts/fetch_policies.sh` downloads files from the Hugging Face dataset or
model repo `pollen-robotics/microduck-policies` (override with `POLICY_REPO`).
Read the model card and any license file there before use. Weights are not
stored in this tree.

## Other libraries

MuJoCo, PyTorch, ONNX Runtime, and the Python packages in `requirements.txt`
have their own licenses. Installing them does not change the license of this
repository.
