# Audio Hit Timing Hypotheses

Context: current best public is `0.13275` from `yolo11s-pose` fighter+hand
grouped NMS; best local validation is `0.340888`. Audio must be treated as a
weak timing feature around pose candidates, not as a standalone detector. Use
only CPU/light checks here; do not touch GPU while `yolo11m` extraction may be
running.

## What We Know

- `tools/evaluate_audio_onsets.py` already decodes audio with `ffmpeg` to mono
  `f32le` at 16 kHz, maps samples to frames at 30 fps, and builds an onset score
  from high-pass RMS, positive energy deltas, and robust z-scored flux.
- Prior full-validation results were negative: audio-only oracle-count peaks
  reached only `0.02927` macro; hard-shifting pose candidates to audio peaks
  hurt; the best positive audio rerank was `0.23789` vs `0.23802` baseline.
- Local `ffprobe` spot checks on test videos showed AAC 44.1 kHz stereo,
  audio/video `start_time=0`, and video duration matching `frame_count / 30`.
  The audio stream can be about 10-13 ms shorter, which is below one frame.
- Existing pose candidates are stronger: use cached `val_yolo11s_conf035` and
  completed `val_yolo11m_conf035` tracks only after verifying line counts equal
  manifest frame counts.

## Hypotheses

1. A/V sync is mostly container-aligned; global frame offsets are unlikely to
   help. Decode audio on the original stream timeline and map `t -> round(t*30)`.
2. Better onset features may help only near pose-dominant events. Test
   per-frame `rms`, high-band `rms`, spectral flux, peak prominence, and nearest
   peak distance in `+/-3`, `+/-6`, and `+/-12` frame windows.
3. `ffmpeg` should remain the canonical extractor because it handles MP4/MOV
   stream timing reliably. Use `librosa` or `torchaudio` after ffmpeg-decoded
   WAV/float buffers for feature variants, not as the source of truth for PTS.
4. Audio should rerank or calibrate pose candidates, not move frames wholesale.
   Useful features: `max_onset_local`, `nearest_peak_delta`, `peak_rank`,
   `prominence`, and interaction with pose score, fighter, hand, and grouped NMS.
5. The plausible win is timing tie-breaks inside wide pose pools, especially
   when multiple same-fighter/hand candidates cluster around one exchange.
   Reject any method whose gain comes mainly from prediction count changes.

## Risks

- Crowd, coaches, commentators, bell sounds, referee movement, ropes, and camera
  handling can create strong transient peaks unrelated to punches.
- Misses, blocked punches, and light scoring touches may be visually clear but
  acoustically weak; landed impacts may be delayed or masked by crowd noise.
- Different sources (`boks`, `tournament_1`, `tournament_2`) may have different
  microphone distance, compression, and drift behavior.
- Extra false positives are expensive: audio peaks are dense, and the metric
  penalizes unmatched predictions directly.

## Fast Offline Plan

1. Sync sanity: run light `ffprobe` summaries over train/test manifests for
   audio stream presence, `start_time`, duration, sample rate, and compare
   `video_duration` with `frame_count / 30`. Flag drift above half a frame.
2. Feature extraction: cache per-frame CPU audio arrays with the current ffmpeg
   path, then compare simple scipy features with `librosa.onset_strength` and a
   torchaudio spectrogram/spectral-flux variant on a small validation subset.
3. GT diagnostics: for the 13 validation videos, report onset rank and nearest
   peak distance around each clear GT punch; stratify by `data_root`,
   `fighter`, `hand`, and landed/blocked/miss.
4. Pose fusion: augment cached pose candidates with audio features and rerun
   grouped `fighter_hand` NMS grids. Start from `yolo11s` best settings; repeat
   on `yolo11m` only when the cache is complete. Keep `alpha=0` as the control.
5. Quick smoke command shape, CPU only:
   `python3 tools/evaluate_audio_onsets.py --tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 --max-videos 2 --quiet --top-k 10 --audio-windows 3,6,12 --audio-alphas 0,0.03,0.05 --fusion-thresholds 1.0,1.15 --fusion-nms-frames 6,10`

## Submit Gates

- No audio-only, hard-shift, global-offset, or count-only submit.
- Candidate must beat `0.340888` local `yolo11s` by at least `0.010` macro, with
  no FP-penalty regression above `0.01`.
- Require at least 10 of 13 validation-video wins versus the matching no-audio
  pose baseline using the same pool/count policy.
- Timing score must improve directly; do not submit if macro gain disappears
  when prediction counts are held fixed.
- Generated CSV must pass `tools/validate_data.py --submission ...` against
  `data/raw/sample_submission.csv` before any Kaggle submit discussion.
