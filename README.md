# RASCAR Boxing Action Recognition

Этот репозиторий содержит два релиз-кандидата для решения RASCAR Boxing Action
Recognition. Каждый кандидат находится в отдельной ветке и содержит свой
`README.md`, код решения и команду воспроизведения.

## Релиз-кандидаты

| Кандидат | Ветка | Что делает |
| --- | --- | --- |
| RC1: Shadow Stack ClearCut | [release/shadow-stack-clearcut](https://github.com/Bismarc228/rascar-boxing-hack/tree/release/shadow-stack-clearcut) | Формирует файл отправки с clear-фильтром для рискованных строк `blocked` и `miss`. |
| RC2: Old-Attribute Source Switch | [release/old-attribute-source-switch](https://github.com/Bismarc228/rascar-boxing-hack/tree/release/old-attribute-source-switch) | Формирует файл отправки с выбором источника строк на уровне видео по правилу `old_attr.source_le15 >= 0.93`. |

## RC1: Shadow Stack ClearCut

Ветка: [release/shadow-stack-clearcut](https://github.com/Bismarc228/rascar-boxing-hack/tree/release/shadow-stack-clearcut)

Финальный CSV:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Воспроизведение одной командой:

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py
```

Ожидаемые внутренние показатели:

```text
macro_score  0.420768
n_pred       1180
n_tp         1091
n_fp         89
```

## RC2: Old-Attribute Source Switch

Ветка: [release/old-attribute-source-switch](https://github.com/Bismarc228/rascar-boxing-hack/tree/release/old-attribute-source-switch)

Финальные CSV:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

Воспроизведение одной командой:

```bash
.venv/bin/python tools/release_build_old_attribute_source_switch.py
```

Ожидаемые валидационные показатели:

```text
macro_score  0.410780
n_pred       1263
n_tp         1144
n_fp         119
```
