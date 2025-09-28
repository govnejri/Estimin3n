import torch
import soundfile as sf
from transformers import AutoProcessor, TextStreamer, Gemma3nForConditionalGeneration
import argparse
from pathlib import Path
import sys
import os
import librosa
from utils.config import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_WAV, get_device, get_dtype, MODEL_NAME, ensure_dirs
torch.set_float32_matmul_precision('high')

class Estimin3nASRInference:
    def __init__(self, model_path: str | os.PathLike | None = None):
        """
        Инициализация модели для инференса
       
        Args:
            model_path: Путь к локальной дообученной модели
        """
        ensure_dirs()
        self.model_path = os.path.expanduser(model_path) if model_path else str(DEFAULT_MODEL_PATH)
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        print(f"📊 Устройство: {self.device} | Тип данных: {self.dtype}")


        # Загружаем модель из локального пути
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map={"": str(self.device)},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )


        # Загружаем процессор из того же пути
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )


        self.model = self.model.to(self.device)
        self.model.eval()
    print(f"🚀 Модель {MODEL_NAME} загружена и готова!\n")


    def load_audio(self, audio_path):
        """Загрузка и предобработка аудио файла"""
        audio_data, sample_rate = sf.read(audio_path)
       
        # Конвертация в mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
       
        # Ресэмплинг до 16kHz
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
       
        # Конвертация в float32
        audio_data = audio_data.astype('float32')
       
        print(f"📊 Аудио загружено: {len(audio_data)/16000:.2f}с")
        return audio_data


    def transcribe(self, audio_path, max_new_tokens=1024, temperature=0.8, streaming=False):
        """Транскрипция аудио файла"""
        audio_data = self.load_audio(audio_path)
       
        # Формируем сообщения
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text", 
                    "text": (
                        "Твоя роль: Анна, высокоэффективный администратор колл-центра компании 'Автодом'. "
                        "Твоя задача — прослушать аудио и немедленно дать четкий, профессиональный ответ на казахском языке. "
                        "Неукоснительно следуй этим правилам:\n"
                        "1. НИКОГДА не транскрибируй то, что услышала. Только отвечай на запрос.\n"
                        "2. Всегда начинай свой ответ со слов: 'Здравствуйте, меня зовут Анна, компания «Автодом»'.\n"
                        "3. Твой ответ должен быть кратким, по делу и полезным для клиента.\n"
                        "Отклонение от этих правил недопустимо."
                    )
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    # Задача для пользователя теперь тоже более прямая
                    {"type": "text", "text": "Прослушай аудио и немедленно предоставь ответ, следуя системным инструкциям."}
                ]
            }
        ]
       
        print("🔄 Генерация транскрипции...")
       
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
           
            # Приведение типов
            if 'input_features' in inputs:
                inputs['input_features'] = inputs['input_features'].to(self.dtype)
            for key in ['input_ids', 'attention_mask']:
                if key in inputs:
                    inputs[key] = inputs[key].long()
           
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': True if temperature > 0 else False,
                'temperature': temperature,
                'pad_token_id': self.processor.tokenizer.eos_token_id,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
                'use_cache': True,
            }
           
            if streaming:
                streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True)
                generation_kwargs['streamer'] = streamer
           
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
           
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            transcription = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return transcription.strip()
       
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            raise


def test_sample():
    """Быстрый тест на sample.wav"""
    audio_path = str(DEFAULT_SAMPLE_WAV)
   
    if not Path(audio_path).exists():
        print(f"❌ Файл {audio_path} не найден!")
        return None
   
    print(f"🎯 Тестирование модели {MODEL_NAME} на sample.wav")
    asr_model = Estimin3nASRInference()
    transcription = asr_model.transcribe(audio_path)
    print(f"📝 Транскрипция: {transcription}")
    return transcription


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} ASR Inference")
    parser.add_argument("--audio", type=str, help="Путь к аудио файлу")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Путь к модели")
    parser.add_argument("--streaming", action="store_true", help="Потоковый вывод")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Максимальное количество токенов")
    parser.add_argument("--temperature", type=float, default=0.8, help="Температура генерации")
    parser.add_argument("--test-sample2", action="store_true", help="Тест на sample2.wav")
   
    args = parser.parse_args()
   
    if args.test_sample2:
        test_sample()
    elif args.audio:
        if not Path(args.audio).exists():
            print(f"❌ Файл {args.audio} не найден!")
            return
       
    asr_model = Estimin3nASRInference(model_path=args.model_path)
        result = asr_model.transcribe(
            args.audio,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            streaming=args.streaming
        )
        print(f"📝 Результат: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        test_sample()
    else:
        main()