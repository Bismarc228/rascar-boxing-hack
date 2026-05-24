# Release Reproducibility Bundle - 2026-05-24

Large files are intentionally not committed. The branch now carries the
reproducibility contract instead:

```text
release_artifacts/release_artifacts_manifest.tsv
tools/bootstrap_release_assets.py
tools/verify_release_artifacts.py
requirements-release.txt
```

The manifest pins each required external/generated artifact by path, byte size,
SHA-256, release scope, stage, and restore note. It includes the YOLO weights,
local HuggingFace model mirror files, required pose-track JSONL caches, terminal
release input CSVs, upstream feature caches, and generated release outputs.

## Environment

The local environment used for the manifest was:

```bash
.venv/bin/python -m pip install -r requirements-release.txt
```

For CUDA Torch wheels, use the matching PyTorch CUDA 12.4 index if the default
index cannot resolve `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`.

## External Weights

Download the non-committed pretrained weights/model mirrors:

```bash
.venv/bin/python tools/bootstrap_release_assets.py download-models --verify
```

This uses:

```text
ultralytics/assets:v8.4.0
google/vit-base-patch16-224-in21k@b4569560a39a0f1af58e3ddaf17facf20ab919b0
openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3
MCG-NJU/videomae-base-finetuned-kinetics@488eb9a0565f257b32866000305c8178965eb9f6
facebook/dino-vitb16@f205d5d8e640a89a2b8ef0369670dfc37cc07fc2
timm:vit_small_patch14_dinov2.lvd142m
timm:vit_small_patch16_224.dino
```

Verify already restored external files without downloading:

```bash
.venv/bin/python tools/verify_release_artifacts.py --stage external --strict
```

The `timm:*` weights are resolved through the timm/HuggingFace cache and do not
have stable project-local filenames. Their downstream feature caches are pinned
in the manifest under `upstream_cache`.

## Pose Tracks

Print the exact pose-track regeneration commands:

```bash
.venv/bin/python tools/bootstrap_release_assets.py print-track-commands --release all
```

Those commands use the validation-video key list explicitly. This matters
because the `val_*_conf035` directories contain the 13 validation videos, not
the whole train split.

Verify pose tracks and terminal generated inputs:

```bash
.venv/bin/python tools/verify_release_artifacts.py --release rc1 --stage generated_input --strict
.venv/bin/python tools/verify_release_artifacts.py --release rc2 --stage generated_input --strict
```

## Upstream Generated CSVs

The generated CSVs are not accepted as opaque source. Their rebuild path is
indexed in:

```text
notes/release_full_provenance_2026-05-24.md
```

That note points from each terminal CSV back to the training/gate scripts and
run notes that produced it. The short version:

```text
RC1: YOLO tracks -> CLIP/VideoMAE/DINO features -> fixed-row heads/gates -> private component stack -> candidate-token clear
RC2: YOLO tracks -> sequence TCN and fixed-row gates -> source table -> old_attr source switch
```

After regenerating a layer, verify it against the manifest before running the
final release runner.

## Final Release Checks

RC1:

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py
.venv/bin/python tools/verify_release_artifacts.py --release rc1 --stage generated_output --strict
```

RC2:

```bash
.venv/bin/python tools/release_build_old_attribute_source_switch.py
.venv/bin/python tools/verify_release_artifacts.py --release rc2 --stage generated_output --strict
```

The final release runners still enforce the expected output SHA-256 values.
The manifest checks add coverage for the intermediate files that used to be
hidden behind ignored CSV/NPZ/JSONL caches.
