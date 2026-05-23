# RC1: Shadow Stack ClearCut

Ветка: `release/shadow-stack-clearcut`

Initial release E2E commit: `afe90d3 Add Shadow Stack ClearCut release E2E`

Эта ветка содержит полностью воспроизводимый E2E-релиз кандидата
`RC1: Shadow Stack ClearCut` из
`notes/release_candidates_2026-05-23.md`.

Кандидат строится поверх strict public-best family и сохраняет fixed-row
структуру. Основная идея: взять private component stack с transition-optimized
punch-type, DINO ROI effectiveness corrections и target guard, затем удалить
восемь самых рискованных `blocked/miss` строк через candidate-token clear
signal.

## Коротко

Финальный шаг релиза:

```text
candidate-token clear: top8 blocked_miss by p_keep
```

Что меняется относительно parent submission:

```text
8 rows: clear=true -> clear=false
```

Split по видео:

```text
agn_037=1
agn_038=1
agn_047=6
```

Split по effectiveness:

```text
blocked=2
miss=6
```

Главный output:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Expected SHA-256:

```text
da285e84174f8cf9489d09f7d57c8e67c1c052ec385053af47786b79de07262a
```

`submissions/` и `data/processed/` игнорируются Git. Ветка коммитит код и
документацию, а финальные CSV/diagnostics пересобираются из локальных caches.

## Как запустить

Из корня репозитория:

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py
```

Успешный финал:

```text
Validation passed.
RC1 Shadow Stack ClearCut reproduced
submission=submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
oof_thresholds=data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv
```

Полезные опции:

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --device auto
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --device cuda
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --device cpu
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --epochs 30
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --data-root /path/to/data/raw
.venv/bin/python tools/release_build_shadow_stack_clearcut.py --skip-hash-check
```

Для release-воспроизведения используй default command без `--skip-hash-check`.
Этот режим проверяет byte-identical SHA финального submission.

## Требования к окружению

Ожидается локальное окружение `.venv` из этого проекта. Минимально нужны:

```text
python 3.12
numpy
scipy
scikit-learn
torch
```

Для `feature-mode=pose_audio` нужен доступ к audio/video path. На практике
должен быть доступен `ffmpeg`, а видеофайлы должны лежать по путям из
`data/raw/train/videos.csv` и `data/raw/test/videos.csv`.

Default runner выставляет thread env vars для воспроизводимости и стабильной
нагрузки:

```text
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
```

GPU не обязателен, но default `--device auto` использует CUDA, если она
доступна. В записанной проверке ветки использовался CUDA device, и итоговый
SHA совпал с expected value.

## Ключевые файлы ветки

```text
README.md
notes/release_shadow_stack_clearcut_e2e.md
notes/release_candidates_2026-05-23.md
tools/release_build_shadow_stack_clearcut.py
tools/evaluate_clear_precision_ranked_gate.py
tools/make_candidate_token_clear_submission.py
tools/evaluate_candidate_token_reranker.py
```

`tools/evaluate_candidate_token_reranker.py` vendored из исторической
experiment-ветки. `tools/make_candidate_token_clear_submission.py` сначала
пытается загрузить этот локальный файл, поэтому checkout старой ветки не нужен.

## Входные файлы

Raw dataset:

```text
data/raw/train/punches.csv
data/raw/train/videos.csv
data/raw/test/videos.csv
data/raw/sample_submission.csv
```

OOF private stack:

```text
data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv
```

OOF p_keep cache для strict current rows:

```text
data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv
```

Train predictions для candidate-token clear model:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Pose tracks:

```text
data/processed/pose_tracks/val_yolo26l_conf035/
data/processed/pose_tracks/test_yolo26l_conf035/
```

Parent test submission:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

## Выходные файлы

Runner перезаписывает:

```text
data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv
data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_video_20260523.csv
data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_rows_20260523.csv
data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_20260523.csv
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

## Из чего состоит решение

### 1. Fixed-row private stack

Base OOF stack:

```text
component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv
```

Его роль: сохранить row/timing/count структуру strict anchor family и заменить
часть component attributes:

1. DINO ROI effectiveness signal.
2. Transition-optimized punch-type component.
3. Target guard.

OOF stack baseline:

```text
macro=0.418474
time=0.530111
fighter=0.537953
fp_penalty=0.061715
```

### 2. OOF clear-drop audit

Команда внутри runner:

```bash
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --data-root data/raw \
  --predictions data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv \
  --no-default-pkeep-source \
  --pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --thresholds 1,2,3,4,5,6,7,8,9,10,12,15 \
  --top-k 12 \
  --output-thresholds data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_rows_20260523.csv
```

Runner требует наличие audit row:

```text
model                 pkeep_strict_current
guard                 blocked_miss
threshold_type        top_k
threshold             8
macro_score           0.420768
n_kept                1180
n_dropped             8
dropped_fp            2
dropped_tp_scorable   4
dropped_tp_time_only  2
fp_penalty            0.059184
```

Именно эта строка фиксирует локальный OOF выбор `top8 blocked_miss`.

### 3. Train-all candidate-token clear model

Команда внутри runner:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --data-root data/raw \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26l_conf035 \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 8 \
  --drop-effectiveness blocked,miss \
  --device auto \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Что делает tool:

1. Загружает train clear rows из `TRAIN_PREDICTIONS`.
2. Строит labels against `data/raw/train/punches.csv`: `fp`, `tp_scorable`,
   `tp_time_only`.
3. Загружает pose/audio features через legacy candidate-token reranker code.
4. Обучает clear model на всех train rows.
5. Инференсит `p_keep` для clear rows parent test submission.
6. Выбирает восемь минимальных `p_keep` среди rows с
   `effectiveness in {blocked, miss}`.
7. Меняет для этих rows `clear=true -> clear=false`.

### 4. Submission validation

Команда:

```bash
.venv/bin/python tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Expected:

```text
Validation passed.
```

### 5. Parent diff check

Runner сравнивает final output с parent submission:

```text
parent=submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
candidate=submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Required diff:

```text
changed_clear=8
by_video: agn_037=1, agn_038=1, agn_047=6
by_effectiveness: blocked=2, miss=6
```

The runner asserts exactly these counts.

### 6. SHA check

Command:

```bash
sha256sum submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Expected:

```text
da285e84174f8cf9489d09f7d57c8e67c1c052ec385053af47786b79de07262a
```

## Expected internal metrics

Final OOF score:

```text
macro_score     0.420768
n_pred          1180
n_tp            1091
n_fp            89
```

Component breakdown:

```text
time            0.529325
fighter         0.538783
punch_type      0.224796
effectiveness   0.311123
hand            0.521381
target          0.481339
fp_penalty      0.059184
```

Build-up:

```text
strict_base                         0.415261
private_stack transitionopt guard   0.418474
token_clear top8 blocked_miss       0.420768
```

Clear-drop component movement:

```text
macro_delta_vs_stack       +0.002293
time_delta_vs_stack        -0.000785
fighter_delta_vs_stack     +0.000830
fp_penalty_delta_vs_stack  -0.002531
```

Public LB for this exact submitted file:

```text
0.19069
```

## Ручные команды

OOF audit only:

```bash
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --data-root data/raw \
  --predictions data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv \
  --no-default-pkeep-source \
  --pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --thresholds 1,2,3,4,5,6,7,8,9,10,12,15 \
  --top-k 12 \
  --output-thresholds data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_rows_20260523.csv
```

Submission build only:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --data-root data/raw \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26l_conf035 \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 8 \
  --drop-effectiveness blocked,miss \
  --device auto \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Validate output:

```bash
.venv/bin/python tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Hash output:

```bash
sha256sum submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

## Проверки runner

Runner падает, если не выполнено любое из условий:

```text
all required raw/cached files exist
OOF audit row pkeep_strict_current/blocked_miss/top_k=8 exists
OOF audit macro_score == 0.420768
OOF audit n_kept == 1180
OOF audit n_dropped == 8
OOF audit dropped_fp == 2
OOF audit dropped_tp_scorable == 4
OOF audit dropped_tp_time_only == 2
OOF audit fp_penalty == 0.059184
tools/validate_data.py passes
parent diff has exactly 8 clear drops
drop split by video is agn_037=1, agn_038=1, agn_047=6
drop split by effectiveness is blocked=2, miss=6
final submission SHA-256 matches expected value
```

## Риски и ограничения

Это internal/private-risk candidate, не guaranteed public-safe solution.

Основные риски:

1. Метрика `0.420768` является local internal validation/OOF score на 13
   validation videos.
2. Public LB для exact submitted file был `0.19069`, сильно ниже internal
   score.
3. Clear-drop gain идёт в основном через снижение FP penalty:

```text
fp_penalty_delta=-0.002531
time_delta=-0.000785
```

4. Clear-drop удаляет не только confirmed FP по OOF audit:

```text
dropped_fp=2
dropped_tp_scorable=4
dropped_tp_time_only=2
```

5. Diff затрагивает `agn_037` и `agn_038`, а эти видео могут быть
   public-sensitive.
6. Визуальный аудит защищённого sibling-кандидата показывал, что похожие
   clear drops могут быть активными exchange rows, а не чистыми false
   positives.

Рекомендуемое использование: reproducible release candidate, offline audit,
controlled private-risk experiment. Не отправлять автоматически без отдельного
решения.

## Troubleshooting

### Missing required RC1 input files

Runner заранее проверяет входы. Если видишь:

```text
missing required RC1 input files
```

значит отсутствует raw dataset, local cache, pose tracks, parent submission или
один из release tools. Проверь раздел `Входные файлы`.

### Validation failed

Запусти вручную:

```bash
.venv/bin/python tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Частые причины:

1. Неполный `data/raw`.
2. Output CSV не совпадает со schema `data/raw/sample_submission.csv`.
3. Parent submission был заменён другим файлом.

### SHA-256 mismatch

Если validation проходит, но SHA не совпал:

1. Проверь, что checkout именно `release/shadow-stack-clearcut`.
2. Проверь parent submission SHA/содержимое.
3. Проверь, что `--epochs=30`, `--feature-mode=pose_audio`,
   `--drop-top-k=8`, `--drop-effectiveness=blocked,miss`.
4. Проверь device/версии `torch`, если запускал не default `--device auto`.
5. Для диагностики можно временно запустить `--skip-hash-check`, но это не
   считается exact release reproduction.

### no clear train rows / no clear test rows

Обычно указан неправильный `--train-predictions` или `--test-submission`.
Проверь, что входной CSV содержит строки с `clear=true`.

### ffmpeg/audio errors

Проверь:

```bash
ffmpeg -version
```

И наличие видеофайлов по путям из:

```text
data/raw/train/videos.csv
data/raw/test/videos.csv
```

## Где смотреть детали

Release summary:

```text
notes/release_candidates_2026-05-23.md
```

Полная E2E note:

```text
notes/release_shadow_stack_clearcut_e2e.md
```

Главный runner:

```text
tools/release_build_shadow_stack_clearcut.py
```

OOF clear audit:

```text
tools/evaluate_clear_precision_ranked_gate.py
```

Train-all materializer:

```text
tools/make_candidate_token_clear_submission.py
```
