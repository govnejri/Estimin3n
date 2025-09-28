import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel
import os
from pathlib import Path
from utils.config import MODELS_DIR, OUTPUTS_DIR, MODEL_NAME, ensure_dirs

# --- НАСТРОЙКИ ---
# 1. Базовая модель от Google
base_model_name = "google/gemma-3n-E4B-it"

# 2. Путь к вашим LoRA адаптерам
adapter_path = str(OUTPUTS_DIR / "checkpoint-10000")

# 3. Имя папки для сохранения ПОЛНОЙ, исправленной модели
output_dir = str(MODELS_DIR / MODEL_NAME)
# -----------------

print("Шаг 1: Загрузка базовой модели и ПОЛНОГО процессора от Google...")

# Загружаем модель. Используем bfloat16 для экономии памяти и device_map для распределения по GPU
model = Gemma3nForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto", 
)

# КЛЮЧЕВОЙ ШАГ: Загружаем AutoProcessor, который включает всё (токены, аудио, изображения)
processor = AutoProcessor.from_pretrained(base_model_name)

print("✅ Базовая модель и процессор загружены.")

# 2. Применяем ваши LoRA адаптеры с помощью PeftModel
print(f"Шаг 2: Применение адаптеров из '{adapter_path}'...")
model = PeftModel.from_pretrained(model, adapter_path)
print("✅ Адаптеры загружены.")

# 3. Объединяем веса адаптеров с базовой моделью
print("Шаг 3: Объединение весов...")
model = model.merge_and_unload()
print("✅ Модель успешно объединена.")

# 4. Сохраняем объединенную модель вместе с полным процессором
print(f"Шаг 4: Сохранение модели и процессора в '{output_dir}'...")
ensure_dirs()
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"🎉 Модель и процессор успешно сохранены в {output_dir}")
print("Теперь эта папка содержит все необходимые файлы для инференса, включая 'preprocessor_config.json'.")