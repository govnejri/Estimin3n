import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm
from pathlib import Path
from utils.config import DATASETS_DIR, RAW_DATA_DIR, DEFAULT_DATASET_DIR, ensure_dirs

def create_dataset_from_files(base_dir):
    # Функция для создания датасета из файлов в директории
    audio_files = []
    transcriptions = []
    
    # Перечисляем все FLAC файлы
    for root, _, files in os.walk(base_dir):
        for file in tqdm(files):
            if file.endswith('.flac'):
                audio_path = os.path.join(root, file)
                text_path = os.path.join(root, file.replace('.flac', '.txt'))
                
                # Проверяем существование текстового файла
                if os.path.exists(text_path):
                    with open(text_path, 'r') as f:
                        text = f.read().strip()
                    
                    audio_files.append(audio_path)
                    transcriptions.append(text)
    
    # Создаем датафрейм и конвертируем в датасет
    df = pd.DataFrame({"audio": audio_files, "text": transcriptions})
    dataset = Dataset.from_pandas(df)
    
    # Преобразуем колонку 'audio' в нужный формат
    dataset = dataset.cast_column("audio", Audio())
    
    return dataset

# Создаем датасеты для каждого сплита
ensure_dirs()
base_corpus = RAW_DATA_DIR / "Kazakh_Speech_Corpus_2" / "ISSAI_KSC2_formatted"
print("Создание тренировочного датасета...")
train_ds = create_dataset_from_files(str(base_corpus / "train"))

print("Создание валидационного датасета...")
val_ds = create_dataset_from_files(str(base_corpus / "validation"))

print("Создание тестового датасета...")
test_ds = create_dataset_from_files(str(base_corpus / "test"))

# Объединяем в один DatasetDict
dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

# Проверяем структуру
print(f"\nПроверка структуры датасета:")
print(f"Ключи в датасете: {dataset['train'][0].keys()}")
print(f"Пример транскрипции: {dataset['train'][0]['text']}")

# Сохраняем датасет
output_path = DEFAULT_DATASET_DIR
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(output_path))
print(f"Датасет сохранен в {output_path}")