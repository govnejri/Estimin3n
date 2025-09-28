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

### Лицензия и кредиты
- Библиотеки и модели принадлежат их авторам (Google Gemma 3N, Hugging Face, Unsloth, TRL и др.).
- Этот репозиторий предоставляет вспомогательные скрипты и обвязку.

## Estimin3n — ASR/LLM пайплайн на базе Gemma 3n (Kazakh)

Проект для дообучения и инференса мультимодальной модели речи-текст (ASR) на казахском языке, а также для автоматической оценки качества распознавания (WER/CER) и проверки знаний на KazMMLU. Основа — семейство моделей Gemma 3n и стек Hugging Face.

### Возможности
- **Формирование датасета** из аудио `.flac` и парных `.txt` транскрипций и сохранение в формате `datasets` на диск.
- **Дообучение (SFT) с 4-битной загрузкой** через `unsloth.FastModel` и `trl.SFTTrainer`.
- **Инференс (CLI)** по одиночному файлу или на встроенном примере.
- **Бенчмарк WER/CER** на пользовательском датасете, сохранение логов и детальной выборки.
- **KazMMLU бенчмарк** (множественный выбор, каз/рус поднаборы).
- **Слияние LoRA-адаптеров** в базовую модель и экспорт итоговой модели (включая GGUF Q8_0).
- Единая конфигурация путей и устройств в `utils/config.py`.

---

## Структура репозитория
```
bench/
  kazmmlu/kazmmlu.py        # Бенчмарк KazMMLU (множественный выбор)
  wer/benchmark.py          # Подсчёт WER/CER по датасету
data/
  data_save.py              # Сборка датасета из .flac + .txt → save_to_disk
inference/
  test.py                   # CLI-инференс ASR по аудиофайлу или sample
train/
  finetune.py               # Дообучение Gemма 3n (SFT, unsloth, LoRA)
utils/
  config.py                 # Пути/устройства/дефолты, ensure_dirs()
  merge_lora.py             # Слияние LoRA в базовую модель и сохранение
  sample.wav                # Пример аудио для быстрого теста
```
Ключевые директории (см. `utils/config.py`):
- `models/` — итоговые модели (по умолчанию `models/Estimin3n`).
- `outputs/` — чекпойнты обучения, логи бенчмарков.
- `data/raw/` — сырьевые данные (ваш корпус).
- `data/datasets/` — датасеты, сохранённые `datasets.save_to_disk`.

---

## Требования
- Python 3.10–3.12 (рекомендовано)
- NVIDIA GPU с CUDA для обучения; инференс возможен на CPU (медленно)
- Рекомендуется актуальная `pip` и установка PyTorch с CUDA согласно официальной инструкции

Установите зависимости (Windows/PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# Установите PyTorch под вашу версию CUDA с https://pytorch.org/get-started/locally/
# Затем общие зависимости:
pip install -U transformers accelerate datasets evaluate jiwer librosa soundfile peft trl unsloth bitsandbytes huggingface_hub pandas tqdm
```
Замечания:
- На Windows пакет `bitsandbytes` может быть недоступен. Если установка не удалась, в `train/finetune.py` замените `optim="adamw_8bit"` на, например, `adamw_torch`.
- Для логирования в Weights & Biases (W&B) требуется вход. Чтобы отключить W&B:
  - PowerShell: `setx WANDB_DISABLED "true"` (новая сессия) или `$env:WANDB_DISABLED="true"` (текущая сессия)
  - Или измените `report_to = []` в конфигурации тренера.

---

## Подготовка данных
Ожидается корпус с файлами:
- `*.flac` — аудио
- `*.txt` — текстовая транскрипция с тем же именем файла

По умолчанию скрипт ищет данные здесь:
```
data/raw/Kazakh_Speech_Corpus_2/ISSAI_KSC2_formatted/
  train/
  validation/
  test/
```
Соберите датасет и сохраните на диск:
```powershell
python data/data_save.py
```
Итоговый датасет будет сохранён в `data/datasets/audio_dataset_with_text` (см. `DEFAULT_DATASET_DIR`).

---

## Дообучение (SFT)
Скрипт `train/finetune.py` использует `unsloth.FastModel` и TRL SFTTrainer над базой `unsloth/gemma-3n-E4B-it` с LoRA.

Запуск (минимум 1 GPU, желательно 24GB+ VRAM):
```powershell
$env:WANDB_DISABLED="true"   # опционально, чтобы отключить W&B
python train/finetune.py
```
Что делает скрипт:
- Загружает базовую модель в 4-битах (`load_in_4bit=True`).
- Форматирует датасет в чат-структуру c аудио.
- Запускает `SFTTrainer` и сохраняет чекпойнты в `outputs/`.
- После обучения сохраняет итоговую модель и процессор в `models/Estimin3n` и экспортирует варианты:
  - Слияние весов в float16/bfloat16 (`save_pretrained_merged`)
  - Экспорт в GGUF Q8_0 (`models/Estimin3n/gguf`)

Настройки путей и имён можно изменить в `utils/config.py` (`MODEL_NAME`, `MODELS_DIR`, `OUTPUTS_DIR`, `DEFAULT_DATASET_DIR`).

---

## Слияние LoRA и сохранение модели
Если вы обучали LoRA-адаптеры отдельно и хотите вручную слить их с базовой моделью и сохранить полный процессор:
```powershell
python utils/merge_lora.py
```
Пути по умолчанию:
- База: `google/gemma-3n-E4B-it`
- Адаптеры: `outputs/checkpoint-10000`
- Выход: `models/Estimin3n`
Отредактируйте значения в `utils/merge_lora.py` при необходимости.

Примечание: `train/finetune.py` уже сохраняет слитые варианты и GGUF. Скрипт `merge_lora.py` полезен для альтернативных сценариев.

---

## Инференс (CLI)
Быстрая проверка на `utils/sample.wav`:
```powershell
python inference/test.py --model-path models/Estimin3n --audio utils/sample.wav --streaming
```
Опции:
- `--audio` — путь к аудиофайлу
- `--model-path` — путь к локальной модели (по умолчанию `models/Estimin3n`)
- `--streaming` — потоковый вывод токенов
- `--max-tokens`, `--temperature` — параметры генерации

Если запустить без аргументов, скрипт автоматически протестирует `utils/sample.wav`.

---

## Бенчмарк WER/CER
Оценка качества распознавания на датасете, сохранённом `datasets.save_to_disk`:
```powershell
python bench/wer/benchmark.py `
  --dataset_path data/datasets/audio_dataset_with_text `
  --model-path models/Estimin3n `
  --sampling_rate 16000 `
  --show-examples --show-every 10 `
  --max-tokens 256 --temperature 0.0
```
Артефакты:
- Лог метрик по мере прогресса: `outputs/benchmark_log.tsv`
- Детальный отчёт по сэмплам: `outputs/benchmark_detailed.tsv`

---

## Бенчмарк KazMMLU
Запуск полного или частичного бенчмарка множественного выбора:
```powershell
# Полный прогон со всеми конфигурациями
python bench/kazmmlu/kazmmlu.py --model-path models/Estimin3n

# Только казахские поднаборы
python bench/kazmmlu/kazmmlu.py --model-path models/Estimin3n --kazakh-only

# Только русские поднаборы
python bench/kazmmlu/kazmmlu.py --model-path models/Estimin3n --russian-only

# Конкретный поднабор
python bench/kazmmlu/kazmmlu.py --model-path models/Estimin3n --subset "Biology (High School in kaz)"
```
Результаты:
- Сводка по конфигурациям: `outputs/kazmlu_results.tsv`
- Детальный отчёт: `outputs/kazmlu_detailed.tsv`

---

## Конфигурация и устройства
Все ключевые пути и настройки находятся в `utils/config.py`.
- `ensure_dirs()` автоматически создаёт нужные папки.
- `get_device()` и `get_dtype()` выбирают устройство (`cuda`/`cpu`) и dtype (`bfloat16` на GPU, `float32` на CPU).

Если у вас CPU-окружение:
- Обучение будет крайне медленным и ограниченным. Рекомендуется GPU.
- Инференс возможен, но скорость ниже.

---

## Известные оговорки
- В Windows возможны сложности с `bitsandbytes`. При ошибках смените оптимизатор на стандартный (`adamw_torch`).
- Для W&B логирования выполните `wandb login` или отключите, как указано выше.
- Базовая модель и процессор загружаются с `trust_remote_code=True` — необходим интернет при первом запуске.

---

## Лицензия
Добавьте `LICENSE` в корень репозитория и опишите условия распространения проекта.

## Благодарности
- Google Gemma 3n
- Hugging Face `transformers`, `datasets`, `trl`
- Unsloth (эффективное дообучение в 4-битном режиме)
- Сообщество KazMMLU
- tsu
