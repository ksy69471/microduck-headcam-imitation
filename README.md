# 頭部カメラから歩く実験の再現用ソース

小型二足ロボットのシミュレータ上で、次をやり直すためのコードです。

1. 公式の歩行ポリシーで歩かせ、関節・胴の軌道と頭部カメラ映像を保存する
2. その映像から、進む速さと向きを推定する
3. 推定した速さを歩行指令に直し、同じ公式ポリシーで実行する
4. 公式ポリシーを使わず、報酬の項を変えて追従方策を PPO で学習する

このディレクトリは単体の git リポジトリにする想定です。データセット、チェックポイント、動画、ONNX、第三者のメッシュは含みません。

歩き方そのものは、ここにある学習コードでは獲得しません。足の動かし方は公式の歩行ポリシー（`alpha_walking.onnx`）が担当し、映像から推定するのは行き先の指令です。お手本の歩行も同じポリシーで作っています。報酬を変える追従学習（手順 4）だけは、このポリシーを制御ループに入れません。

## ライセンス

| 対象 | 条件 |
|------|------|
| このツリーのファイル | Apache-2.0（`LICENSE`、`NOTICE`） |
| `microduck` / `microduck_rl` のコードと MJCF | 上流の Apache-2.0 |
| `microduck_rl` の STL メッシュ | CC BY-NC-SA 4.0（非商用・継承）。ここには含めない |
| ONNX ポリシー | Hugging Face のモデルカードに従う。ここには含めない |
| アクチュエータ庫 `bam` | Rhoban のリポジトリのライセンスに従う。ここには含めない |

詳細は [third_party/NOTICES.md](third_party/NOTICES.md) です。ダウンロードしたメッシュ、ONNX、生成したデータセットはコミットしないでください。

## 準備

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

`fetch_third_party.sh` は `readonly/` にクローンします。このディレクトリは gitignore です。`fetch_policies.sh` は ONNX を `work/local/scratch/policies/` に置きます。

短い追従学習なら CPU 版の PyTorch でも動きます。映像モデルの学習は GPU の方が速いです。

## 1. 歩行と頭部カメラ映像を記録する

公式の歩行 ONNX を MuJoCo 上で動かし、関節と胴の軌道を保存します。シーン内のモニタに教師映像を流し、ロボット頭部のカメラから見た画像も保存します。

```bash
python scripts/build_ego_dataset.py \
  --out datasets/motions/POL-WALK-EGO \
  --n 120 --seed-base 1000 --workers 8
python scripts/make_clip_lists.py --data datasets/motions/POL-WALK-EGO
python scripts/compute_image_centroids.py --data datasets/motions/POL-WALK-EGO
```

`--n 120` は動作確認用の本数です。長い学習では、`build_ego_dataset.py` の `--extend` で学習用クリップを数百本足しています。

## 2. 映像から動きを推定する

`--source C` は頭部カメラのフレーム（ロボットから見たモニタ）を読みます。`--source A` は教師映像を直接読みます。

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

## 3. 速度を指令に直し、公式ポリシーで歩く

逆モデルは、同じ ONNX 歩行ポリシーにランダムな指令を出して歩かせた記録から、速度を指令に直す対応を学習します。教師の関節角には合わせません。実行時も、推定した指令を公式ポリシーに渡します。

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

`command_imitation_ego.py` は、平均指令、推定指令、必要なら探索、など複数の指令の出どころを比べます。条件名は `--help` を見てください。

## 4. 報酬を変えて追従方策を学習する

記録済みの関節と胴の参照に対し、方策をゼロから学習します（`--root-cond`）。制御ループに ONNX ポリシーは入れません（`--no-onnx-teacher`）。グリッドでは、関節の許容幅、前進のご褒美、生存のご褒美を変えます。

```bash
python scripts/grid_search_tracking_rewards.py --help
python evaluation/eval_tracking_reward_study.py --help
```

チェックポイントとログは `work/local/runs/` に出ます。チェックポイントは検証用クリップで選び、報告する数値はテスト用クリップで出してください。テスト用で選ばないでください。

## ファイルの役割

| パス | 役割 |
|------|------|
| `simulation/mujoco/capture_ego.py` | ONNX 歩行の実行とカメラ |
| `observation/virtual_screen.py` | モニタ映像を頭部カメラに映す |
| `training/train_v2m_ego.py` | 映像から動作を推定する学習 |
| `evaluation/inverse_body_model.py` | 速度から指令への変換 |
| `evaluation/command_imitation_ego.py` | 公式ポリシーで指令を実行 |
| `training/train_tracking_policy.py` | PPO の追従方策 |
| `scripts/grid_search_tracking_rewards.py` | 報酬のグリッド |

## このツリーに無いもの

- 課題の文章、スライド、図のキャプション
- 個人のマシン名やローカルの仮想環境
- 描画済みデータ、チェックポイント、比較動画
- 上流のロボットメッシュと ONNX ポリシー
