## Estimin3n — открытая мультимодальная казахская audio/text→text LLM (модель)

Estimin3n: SOTA opensource multimodal kazakh audio/text to text LLM

Этот репозиторий модели Estimin3n содержит всё, что связано с моделью:
- дообучение аудио-языковой базы Gemma 3N на собственных данных,
- инференс (ASR/response) с использованием локально сохранённой модели,
- оценку качества на аудио-датасетах (WER/CER) и KazMMLU.

Репозиторий ориентирован на русско- и казахскоязычные сценарии.

### Содержание
- Описание структуры репозитория
- Требования и установка
- Подготовка данных
- Обучение (fine-tuning)
- Инференс ASR/ответа
- Бенчмарки: WER/CER и KazMMLU
- Управление путями и конфигурацией
- FAQ / Троблшутинг

---

### Структура репозитория
```
Estimin3n/
  bench/
    kazmmlu/
      kazmmlu.py         # Бенчмарк на KazMMLU (множественный выбор)
    wer/
      benchmark.py       # Бенчмарк WER/CER по аудио-датасету
  data/
    data_save.py         # Конвертация корпуса во внутренний формат HF DatasetDict
  inference/
    test.py              # Инференс ASR/ответа для одиночного аудио
  train/
    finetune.py          # Дообучение Gemma 3N c Unsloth + TRL SFTTrainer
  utils/
    config.py            # Пути, дефолты и утилиты
    merge_lora.py        # Слияние LoRA-адаптеров с базовой моделью
    sample.wav           # Тестовый аудио-файл
  models/                # (создаётся автоматически) сохранение финальной модели
  outputs/               # (создаётся автоматически) чекпоинты/логи/отчёты
  data/datasets/         # (создаётся автоматически) сохранённые HF датасеты
  data/raw/              # (ожидаемая сырая структура корпуса)
```

---

### Требования и установка

Рекомендуемые версии (ориентир):
- Python 3.10+
- CUDA-совместимый PyTorch
- Библиотеки: transformers, datasets, soundfile, librosa, evaluate, jiwer, unsloth, peft, trl, tqdm, pandas

Пример установки (через pip):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # подберите под вашу CUDA
pip install transformers datasets soundfile librosa evaluate jiwer unsloth peft trl tqdm pandas
```

Если используете Windows, убедитесь, что установлены зависимости для `soundfile` (libsndfile) и `librosa`.

---

### Управление путями и конфигурацией
Ключевые пути и значения по умолчанию определены в `utils/config.py`:
- `MODELS_DIR` → `models/`
- `OUTPUTS_DIR` → `outputs/`
- `DATA_DIR` → `data/`, `RAW_DATA_DIR` → `data/raw/`, `DATASETS_DIR` → `data/datasets/`
- `DEFAULT_MODEL_PATH` → `models/Estimin3n`
- `DEFAULT_DATASET_DIR` → `data/datasets/audio_dataset_with_text`
- `DEFAULT_SAMPLE_WAV` → `utils/sample.wav`

Утилита `ensure_dirs()` автоматически создаёт необходимые директории при запуске скриптов.

---

### Подготовка данных
Скрипт `data/data_save.py` обходит сырые данные и формирует HF DatasetDict.

Ожидаемая структура сырого корпуса (пример под `Kazakh_Speech_Corpus_2/ISSAI_KSC2_formatted`):
```
data/raw/Kazakh_Speech_Corpus_2/ISSAI_KSC2_formatted/
  train/
    **/*.flac + соответствующие .txt с транскриптом
  validation/
    **/*.flac + .txt
  test/
    **/*.flac + .txt
```

Запуск конвертации:
```bash
python -m data.data_save
```
Итоговый датасет сохранится в `data/datasets/audio_dataset_with_text`.

---

### Обучение (fine-tuning)
Скрипт: `train/finetune.py`

Основные положения:
- Загружается базовая Gemma 3N через `unsloth.FastModel`
- Формируется тренировочный датасет с сообщениями вида system/user/assistant
- Используется `trl.SFTTrainer` для SFT
- Чекпоинты и логи в `outputs/`; финальная модель и процессор сохраняются в `models/Estimin3n`

Запуск (пример):
```bash
python -m train.finetune
```

После обучения можно объединить веса LoRA с базовой моделью:
```bash
python -m utils.merge_lora
```
Скрипт ожидает, что ваши адаптеры лежат в `outputs/checkpoint-10000` (переопределите путь при необходимости внутри файла).

---

### Инференс ASR/ответа
Скрипт: `inference/test.py`

Поддерживается два варианта запуска:
1) Без аргументов — быстрый тест на `utils/sample.wav`:
```bash
python -m inference.test
```

2) С указанием произвольного файла:
```bash
python -m inference.test \
  --audio path/to/audio.wav \
  --model-path models/Estimin3n \
  --streaming \
  --max-tokens 2048 \
  --temperature 0.8
```

Скрипт автоматически приведёт аудио к 16 кГц, моно, float32. Ответ генерируется моделью Gemma 3N через `apply_chat_template` и `generate`.

Примечание: текущая версия `inference/test.py` содержит системный промпт под сценарий ответов колл-центра, а не буквальную транскрипцию. Для чистого ASR используйте логику из `bench/wer/benchmark.py` как ориентир (см. ниже).

---

### Бенчмарк WER/CER
Скрипт: `bench/wer/benchmark.py`

Пример запуска:
```bash
python -m bench.wer.benchmark \
  --dataset_path data/datasets/audio_dataset_with_text \
  --model-path models/Estimin3n \
  --sampling_rate 16000 \
  --output_file outputs/benchmark_log.tsv \
  --detailed_output_file outputs/benchmark_detailed.tsv \
  --max-tokens 256 \
  --temperature 0.0 \
  --show-examples \
  --show-every 10
```

Скрипт:
- загружает датасет из `--dataset_path` (ожидается колонка `audio` и `text`),
- приводит аудио к 16 кГц при необходимости,
- считает промежуточные и итоговые WER/CER,
- сохраняет логи в `outputs/` и печатает примеры.

---

### Бенчмарк KazMMLU
Скрипт: `bench/kazmmlu/kazmmlu.py`

Пример запуска (все конфигурации):
```bash
python -m bench.kazmmlu.kazmmlu \
  --model-path models/Estimin3n \
  --run-all \
  --max-tokens 5 \
  --temperature 0.1 \
  --show-examples \
  --show-every 50 \
  --output_file outputs/kazmlu_results.tsv \
  --detailed_output_file outputs/kazmlu_detailed.tsv
```

Опции фильтрации:
- `--subset "Biology (High School in kaz)"`
- `--kazakh-only`
- `--russian-only`

Результаты сохраняются в `outputs/` и печатается агрегированная статистика по точности.

---

### Частые проблемы
- CUDA/torch несовместимость: установите совместимую сборку PyTorch для вашей версии CUDA.
- Ошибки `soundfile/librosa`: установите системные зависимости (libsndfile) и убедитесь, что `numpy` совместимой версии.
- Пустые ответы/токены: проверьте `max_new_tokens`, `temperature`, а также, что у процессора корректно выставлен `pad_token_id`.
- Плохая транскрипция: убедитесь, что промпт действительно просит транскрипцию (см. `bench/wer/benchmark.py`) и что аудио 16 кГц моно.

---

### Благодарности
- Google Gemma 3n
- Hugging Face `transformers`, `datasets`, `trl`, `unsloth`
- Unsloth (эффективное дообучение в 4-битном режиме)
- Сообщество KazMMLU
- tsu
