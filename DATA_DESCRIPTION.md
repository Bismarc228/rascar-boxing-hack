Dataset Description
Описание датасета — Rascar Box
По видео боксёрских поединков нужно найти моменты ударов: на каком кадре клипа и какой боец (красный / синий) нанёс удар, с атрибутами руки, цели, результата и «чистоты» удара.

Одна строка сабмита = одно событие удара. Кадры 0-based внутри каждого mp4-файла (как в train/punches.csv).

Что вам понадобится
Вопрос	Ответ
Что в датасете?	Папки train/, test/, файлы sample_submission.csv, README.txt, ноутбук RASCAR_baseline_boxing.ipynb.
Где видео?	Отдельно — три ссылки в README.txt (Яндекс.Диск): бокс, Турнир Бокс, Турнир Бокс 2. Пути в video_path — относительно корня соответствующей папки/архива с видео.
Формат данных?	CSV (UTF-8). Видео — mp4/MOV.
Что предсказываем?	Список ударов на 9 test-клипах: кадр, боец, hand, target, effectiveness, clear (см. сабмит).
Сокращения	АГН — id клипа (АГН-1 … АГН-72). agn_index — то же числом (14 для АГН-14). video_key — ASCII (agn_014). red / blue — угол камеры, не фамилии.
Структура датасета
train/
  videos.csv
  punches.csv
test/
  videos.csv
sample_submission.csv
README.txt
RASCAR_baseline_boxing.ipynb
Путь	Содержимое
train/videos.csv	Манифест обучающих клипов (63 видео)
train/punches.csv	Разметка ударов (9030 событий)
test/videos.csv	Тестовые клипы без разметки (9 видео)
sample_submission.csv	Шаблон формата ответа (1594 строки, см. ниже)
README.txt	Ссылки на скачивание видео
RASCAR_baseline_boxing.ipynb	Стартовый baseline (см. ниже)
Видео в таблицах нет — скачайте по README.txt и соберите локально папки из колонок data_root / video_path.

Стартовый baseline — RASCAR_baseline_boxing.ipynb
Ноутбук для Google Colab (GPU): YOLO Pose + трекинг людей, эвристика red / blue по цвету формы (торс, шорты, перчатки в HSV). На выходе — размеченное видео с двумя бойцами, без готового CSV с ударами.

Отправная точка для пайплайна (детекция людей → роли → дальше свои правила/модели на кадры ударов). Пути к видео в ноутбуке поправьте под свою копию данных.

Зависимости: ultralytics, opencv-python, numpy, scipy.

Идентификаторы клипа (кириллица и кодировки)
У каждого клипа три способа сослаться на одно и то же видео:

Поле	Пример	Когда использовать
video_id	АГН-14	Основной человекочитаемый id
agn_index	14	Join в pandas/SQL без кириллицы
video_key	agn_014	Пути к файлам, логи, сабмит — только ASCII
Плюс путь: video_path + source_video (в турнирах часто Раунд1.mp4 — тоже кириллица).

Если на Windows/Linux «ломаются» пути с русскими буквами — открывайте видео по agn_index / video_key из манифеста или переименуйте локальную копию по video_key. В сабмите рекомендуем указывать agn_index и video_key вместе с video_id.

Видео на диске
Train — три корня (колонка data_root):

data_root	Содержимое	Клипов в train
бокс	Тренировочные записи IMG_*.mp4 / MOV	13
Турнир Бокс	Турнир 1: папка боя → Раунд1.mp4, Раунд2.mp4, …	29
Турнир Бокс 2	Турнир 2: то же	21
Пример пути: Турнир Бокс/Бой1. Петросян-Зайнуллаев/Раунд1.mp4.

Test — три полных боя (все раунды вместе):

Турнир Бокс / Бой9. Вороной-Болотов → АГН-37, АГН-38, АГН-39 (agn_index 37–39)
Турнир Бокс 2 / Бой 2. Чемоханов и Иванов → АГН-47 … АГН-49
Турнир Бокс 2 / Бой 8. Рычков и Львов → АГН-62 … АГН-64
Описание CSV
train/videos.csv
Колонка	Описание
video_id	Уникальный id, напр. АГН-14
agn_index	Число 1…72 (= номер в АГН-N)
video_key	ASCII, напр. agn_014
video_path	Относительный путь к mp4
source_video	Имя файла (Раунд1.mp4, IMG_7390.mp4, …)
round_number	Номер раунда (пусто для бокс)
dataset_type	training, tournament_1, tournament_2
data_root	Корневая папка на диске
fight_index	Порядковый номер боя в турнире / блоке
fight_folder	Имя папки боя
frame_count	Число кадров в клипе
width, height	Разрешение
fps	Зарезервировано (в CSV пусто — берите fps из самого видео)
punch_count	Число размеченных ударов (только train)
train/punches.csv
Колонка	Описание
video_id, agn_index, video_key	Клип (три эквивалентных id)
video_path, source_video, round_number, dataset_type, data_root, fight_index, fight_folder	Дублируют манифест для join
frame	Индекс кадра в этом mp4 (0, 1, 2, …)
fighter	red или blue
punch_type	jab, cross, hook, uppercut — только в train, в сабмит не входит
hand	left, right
target	head, body
effectiveness	landed, blocked, miss
clear	true / false
width, height	Разрешение кадра
test/videos.csv
Те же колонки, что в train/videos.csv, без punch_count. Разметки ударов нет.

sample_submission.csv
Шаблон в корне датасета: 1594 строки, id = 1 … 1594. На Kaggle вам нужно сдать столько же строк.

Заголовок:

id,video_id,agn_index,video_key,frame,fighter,hand,target,effectiveness,clear
Колонка	Описание
id	Номер строки (1…n)
video_id	Test-клипы АГН-37 … АГН-64 (9 видео)
agn_index, video_key	Дублируйте из test/videos.csv
frame	Предсказанный кадр (0 … frame_count − 1)
fighter	red / blue
hand	left / right
target	head / body
effectiveness	landed / blocked / miss
clear	true / false
Files
6 files

Size
2.2 MB

Type
csv, ipynb, txt

License
MIT

