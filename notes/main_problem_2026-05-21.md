# Главная проблема - 2026-05-21

Коротко: проблема сейчас не в том, что закончились идеи или что надо еще
покрутить пару параметров. Главная проблема - слабый и нестабильный перенос
локального OOF на public leaderboard. Локально мы видим headroom, но пока не
умеем надежно понять, какая часть этого headroom перенесется на тест, а какая
является подгонкой под validation fights.

## Что уже не проблема

Формат/нулевой score уже не главный блокер:

```text
team public best: 0.16461
root artifact: submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv
latest read-only Kaggle check: 2026-05-21T14:47:18Z
visible leader: 0.32088
submissions used today: 8/30
```

То есть pipeline умеет делать валидные CSV и получать ненулевой public score.
Бенчмарк уже не убивается простым "сделать валидный файл"; нужен переносимый
сигнал.

## В чем реальный блокер

Локальный validation/OFFLINE score не является надежной функцией public score.
Ветки, которые выглядят сильнее локально, часто меняют timing/count/fighter
слишком широко и несут public risk. Public best, наоборот, держится на более
узком и осторожном root-like решении, где особенно важен `agn_038`.

Текущая картина:

```text
public root OOF anchor                         0.365177
best single local OOF source                   0.407232
best materialized local OOF source-policy      0.407417
current best fixed-row public-safe OOF probe   0.398438
team public best                               0.16461
```

Разрыв между локальными числами и public большой. Поэтому очередной локальный
`+0.001` не равен реальному прорыву. Без public probe или более сильного
transfer-сигнала это может быть просто локальный шум.

## Почему не грузил автоматически

Потому что без явного approval это трата сабмита на гипотезу. Сейчас осталось
22 попытки до reset, но они не бесплатные с точки зрения информации: плохой
upload может только подтвердить уже известное - локальный OOF переоценивает
широкие sequence/timing замены.

Сейчас валидные кандидаты есть, но они отвечают на разные вопросы:

```text
lowest-blast public probe:
submissions/hybrid_root_attr_effectiveness_rgb_eff_m0_no_miss_to_landed_agn038_only_OFFLINE_CANDIDATE.csv

strongest fixed-row full public/private probe:
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_rgb_eff_m0_no_miss_to_landed_OFFLINE_CANDIDATE.csv

broader but riskier attribute probe:
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_rgb_eff_m03_OFFLINE_CANDIDATE.csv
```

Первый вариант меняет только `effectiveness` на `agn_038` и поэтому является
самым дешевым public-тестом. Второй меняет только `effectiveness` на всех test
videos, timing/count/fighter остаются как у root. Третий сильнее локально по
старой ветке, но меняет больше атрибутов.

## Почему это не "жесткий прорыв"

Последний прогресс был не breakthrough, а снижение риска:

```text
attr_effectiveness base                      0.395018
attr_effectiveness + RGB effectiveness       0.397369
transition-gated no miss->landed             0.398438
```

Это полезно, потому что изменения fixed-row и не трогают timing/count/fighter.
Но это все еще атрибутная правка, в основном `effectiveness`. Она не решает
самую дорогую часть задачи: найти правильные удары, правильный момент и
правильного fighter на тесте.

То есть честная формулировка такая: мы нашли более чистый и безопасный probe,
но не нашли новый механизм, который уверенно переносит timing/fighter качество
на public/private.

## Где есть headroom, но нет политики

Oracle показывает, что источники содержат полезные куски:

```text
large source oracle headroom          0.430014
focused stacked-attribute oracle      0.422839
public-anchor source oracle           0.404506
```

Но learned policy не умеет это стабильно выбрать:

```text
best single source                    0.407232
ridge preserve-order policy           0.407417
delta                                 +0.000185
stump policies                        regress / overfit
public-anchor ridge                   collapses to one existing source
```

Это и есть центральная боль: hindsight-oracle знает, какой источник брать на
каком fight, а модель выбора источника либо переобучается, либо схлопывается в
один уже существующий artifact.

## Что будет настоящим прорывом

Жесткий прорыв - это не еще один threshold. Это один из следующих результатов:

1. Public-transfer signal: правило или модель, которая заранее отличает
   public-good от public-bad замен и подтверждается сабмитом.
2. Timing/count branch, который дает рост не только на OOF, но и не ломает
   public-sensitive videos вроде `agn_038`.
3. Fighter identity correction с нормальным OOF lift, а не micro-change на
   1-20 строк, и с чистым test artifact.
4. Source-selection policy, которая конвертирует oracle headroom в реальный
   testable artifact, а не только красиво выглядит в hindsight.

## Практический вывод

Если цель - получить информацию с Kaggle сейчас, самый рациональный upload:

```text
submissions/hybrid_root_attr_effectiveness_rgb_eff_m0_no_miss_to_landed_agn038_only_OFFLINE_CANDIDATE.csv
```

Он не претендует на private breakthrough, но отвечает на важный вопрос:
реагирует ли public на текущий RGB/attribute effectiveness signal на самом
public-sensitive видео.

Если цель - не тратить upload, следующий workstream должен быть не "еще
параметры", а один из двух:

1. построить более жесткий transfer audit по video/fight fingerprints;
2. вернуться к timing/fighter identity, но только с критерием: не меньше
   `+0.005` OOF на честной проверке или чистая testable policy.

До появления такого сигнала локальный рост на `+0.001..+0.003` надо считать
кандидатом для probe, а не доказанным прорывом.
