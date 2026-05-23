# RC2: Old-Attribute Source Switch

Ветка: [release/old-attribute-source-switch](https://github.com/Bismarc228/rascar-boxing-hack/tree/release/old-attribute-source-switch)

`RC2: Old-Attribute Source Switch` - релиз-кандидат для RASCAR Boxing Action
Recognition. Решение собирает файл отправки через выбор источника строк на уровне
видео: основной источник остается `seq_motion`, а источник `old_attr`
используется там, где его локальная согласованность с таймингами проходит
порог `source_le15 >= 0.93`.

## Из чего состоит решение

1. Чтение метаданных боев, разметки обучающей части и шаблона файла отправки.
2. Формирование таблицы качества источников по каждому видео.
3. Расчет локальной согласованности строк в окне 15 кадров.
4. Применение правила `old_attr.source_le15 >= 0.93`.
5. Сборка валидационных строк и тестового CSV по выбранному источнику для
   каждого видео.
6. Сортировка строк, перенумерация `id`, сохранение CSV и проверка схемы.

## Входные файлы

Сырые данные должны лежать здесь:

```text
data/raw/train/punches.csv
data/raw/train/videos.csv
data/raw/test/videos.csv
data/raw/sample_submission.csv
```

Сырые видео должны быть доступны по путям, указанным в:

```text
data/raw/train/videos.csv
data/raw/test/videos.csv
```

## Как запустить

Из корня репозитория:

```bash
.venv/bin/python tools/release_build_old_attribute_source_switch.py
```

## Выходные файлы

Валидационный CSV:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
```

Финальный CSV:

```text
submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

Ожидаемые SHA-256:

```text
валидационный_csv  1123c112665e351ea0249814a19b6d4a01a29d65273a7482fa0d0981515bc731
финальный_csv      a34c7eb6bbe11f1731a586ec8869a413a2ab1dd04c4648efeedb61847139102e
```

## Ожидаемые валидационные показатели

```text
macro_score  0.410780
n_pred       1263
n_tp         1144
n_fp         119
```

Компоненты метрики:

```text
time            0.541274
fighter         0.549713
punch_type      0.198453
effectiveness   0.260609
hand            0.526352
target          0.491679
fp_penalty      0.071576
```
