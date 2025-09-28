import torch
torch._dynamo.disable()
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import argparse
import os
import librosa
import numpy as np
from tqdm import tqdm
from evaluate import load
from datasets import load_from_disk
import pandas as pd
import jiwer
from pathlib import Path
from utils.config import DEFAULT_MODEL_PATH, DEFAULT_DATASET_DIR, OUTPUTS_DIR, get_device, get_dtype, MODEL_NAME, ensure_dirs


class Estimin3nASRInference:
    def __init__(self, model_path: str | os.PathLike | None = None):
        ensure_dirs()
        self.model_path = str(Path(model_path).expanduser()) if model_path else str(DEFAULT_MODEL_PATH)
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        print(f"📊 Устройство: {self.device} | Тип данных: {self.dtype}")

        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=self.dtype,
            device_map={"": str(self.device)}, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model.eval()
    print(f"🚀 Модель {MODEL_NAME} загружена и готова!\n")

    def transcribe_from_array(self, audio_array: np.ndarray, source_sampling_rate: int, max_new_tokens=256, temperature=0.1):
        if source_sampling_rate != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=source_sampling_rate, target_sr=16000)
        
        audio_data = audio_array.astype('float32')

        messages = [
            {"role": "system", "content": [{"type": "text", "text": """Сіз қазақ тіліндегі аудионы дәл транскрипциялайтын маман көмекшісіз. 


Нәтижесінде тек таза қазақша мәтін беріңіз."""}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_data}, {"type": "text", "text": "Осы аудионы транскрипциялаңыз."}]}
        ]
        
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(self.device)
            
            if 'input_features' in inputs:
                inputs['input_features'] = inputs['input_features'].to(self.dtype)
            for key in ['input_ids', 'attention_mask']:
                if key in inputs:
                    inputs[key] = inputs[key].long()
            
            generation_kwargs = {
                'max_new_tokens': max_new_tokens, 'do_sample': True if temperature > 0 else False, 'temperature': temperature,
                'pad_token_id': self.processor.tokenizer.eos_token_id, 'eos_token_id': self.processor.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            transcription = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return transcription.strip()
        
        except Exception as e:
            print(f"❌ Ошибка при обработке аудио-массива: {e}")
            return ""


def print_comparison(sample_num, reference, hypothesis, show_metrics=True):
    """Красивый вывод сравнения текстов"""
    print(f"\n{'='*80}")
    print(f"📝 СЭМПЛ #{sample_num}")
    print(f"{'='*80}")
    print(f"🎯 РЕАЛЬНЫЙ ТЕКСТ:")
    print(f"   {reference}")
    print(f"\n🤖 ТРАНСКРИПЦИЯ:")
    print(f"   {hypothesis}")
    
    if show_metrics:
        # Вычисляем метрики для отдельного сэмпла
        sample_wer = jiwer.wer(reference, hypothesis)
        sample_cer = jiwer.cer(reference, hypothesis)
        print(f"\n📊 МЕТРИКИ ДЛЯ ЭТОГО СЭМПЛА:")
        print(f"   WER: {sample_wer:.4f} ({sample_wer*100:.2f}%)")
        print(f"   CER: {sample_cer:.4f} ({sample_cer*100:.2f}%)")
    
    print(f"{'='*80}")


def run_benchmark(asr_model, dataset_path, source_sampling_rate, max_tokens, temperature, 
                  output_file, detailed_output_file, show_examples=True, show_every=10):
    try:
        dataset = load_from_disk(dataset_path)['train']
    except FileNotFoundError:
        print(f"❌ Ошибка: папка с датасетом не найдена по пути {dataset_path}")
        return

    print(f"🔊 ВАЖНО: предполагается, что все аудио в датасете имеют частоту {source_sampling_rate} Гц.")
    print(f"👁️  Примеры будут показываться каждые {show_every} сэмплов" if show_examples else "👁️  Примеры отключены")
    
    references = []
    hypotheses = []
    results_data = [] 

    wer_metric = load("wer")
    cer_metric = load("cer")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("sample_count\tWER\tCER\n")

        for i, sample in enumerate(tqdm(dataset, desc="🎛️  Обработка аудио")):
            audio_array = sample['audio']['array']
            reference_text = sample.get('text') or sample.get('transcription')

            if audio_array is None or reference_text is None:
                continue
            
            reference_text = str(reference_text).lower()
            
            hypothesis_text = asr_model.transcribe_from_array(
                audio_array,
                source_sampling_rate=source_sampling_rate,
                max_new_tokens=max_tokens,
                temperature=temperature
            ).lower()

            references.append(reference_text)
            hypotheses.append(hypothesis_text)
            
            results_data.append({
                "sample_id": i + 1,
                "reference": reference_text,
                "hypothesis": hypothesis_text
            })
            
            # Показываем примеры согласно настройкам
            if show_examples and ((i + 1) % show_every == 0 or i == 0):
                print_comparison(i + 1, reference_text, hypothesis_text, show_metrics=True)
            
            # Сохраняем промежуточные метрики каждые 10 сэмплов
            if (i + 1) % 10 == 0:
                wer_score = wer_metric.compute(predictions=hypotheses, references=references)
                cer_score = cer_metric.compute(predictions=hypotheses, references=references)
                f.write(f"{(i + 1)}\t{wer_score:.4f}\t{cer_score:.4f}\n")
                f.flush()
                
                # Показываем промежуточную статистику
                print(f"\n📈 Промежуточная статистика после {i + 1} сэмплов:")
                print(f"   Средний WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
                print(f"   Средний CER: {cer_score:.4f} ({cer_score*100:.2f}%)")

    if not references:
        print("❌ Не было обработано ни одного файла.")
        return

    print("\n✅ Бенчмарк завершен. Расчет итоговых метрик...")
    
    final_wer_score = wer_metric.compute(predictions=hypotheses, references=references)
    final_cer_score = cer_metric.compute(predictions=hypotheses, references=references)
    
    print(f"\n{'='*50}")
    print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*50}")
    print(f"📊 Всего обработано сэмплов: {len(references)}")
    print(f"🔡 WER (Word Error Rate):     {final_wer_score:.4f} ({final_wer_score*100:.2f}%)")
    print(f"🔤 CER (Character Error Rate): {final_cer_score:.4f} ({final_cer_score*100:.2f}%)")
    print(f"{'='*50}\n")
    
    # Показываем несколько лучших и худших примеров
    if show_examples and len(results_data) > 0:
        print("\n🎯 АНАЛИЗ РЕЗУЛЬТАТОВ:")
        
        # Вычисляем WER для каждого сэмпла
        for result in results_data:
            result['sample_wer'] = jiwer.wer(result['reference'], result['hypothesis'])
        
        # Сортируем по WER
        results_sorted = sorted(results_data, key=lambda x: x['sample_wer'])
        
        print(f"\n✅ ТОП-3 ЛУЧШИХ РЕЗУЛЬТАТА (самый низкий WER):")
        for i, result in enumerate(results_sorted[:3]):
            print(f"\n🥇 #{i+1} (Сэмпл {result['sample_id']}, WER: {result['sample_wer']:.4f}):")
            print(f"   Реальный:     {result['reference']}")
            print(f"   Транскрипция: {result['hypothesis']}")
        
        print(f"\n❌ ТОП-3 ХУДШИХ РЕЗУЛЬТАТА (самый высокий WER):")
        for i, result in enumerate(results_sorted[-3:]):
            print(f"\n💔 #{i+1} (Сэмпл {result['sample_id']}, WER: {result['sample_wer']:.4f}):")
            print(f"   Реальный:     {result['reference']}")
            print(f"   Транскрипция: {result['hypothesis']}")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{len(references)}\t{final_wer_score:.4f}\t{final_cer_score:.4f}\n")
    
    print(f"💾 Промежуточные и финальные результаты сохранены в файл: {output_file}")

    print("✍️  Расчет метрик для детального отчета...")
    results_df = pd.DataFrame(results_data)
    results_df['wer'] = results_df.apply(lambda row: jiwer.wer(row['reference'], row['hypothesis']), axis=1)
    results_df['cer'] = results_df.apply(lambda row: jiwer.cer(row['reference'], row['hypothesis']), axis=1)

    results_df.to_csv(detailed_output_file, sep='\t', index=False, encoding='utf-8')
    print(f"💾 Детальный отчет по каждому сэмплу сохранен в файл: {detailed_output_file}")


def main():
    parser = argparse.ArgumentParser(description=f"Бенчмарк для модели {MODEL_NAME} ASR")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_DIR), help="Путь к папке с датасетом")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Путь к модели")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Частота дискретизации аудио в вашем датасете.")
    parser.add_argument("--output_file", type=str, default=str(OUTPUTS_DIR / "benchmark_log.tsv"), help="Файл для сохранения лога с промежуточными результатами.")
    parser.add_argument("--detailed_output_file", type=str, default=str(OUTPUTS_DIR / "benchmark_detailed.tsv"), help="Файл для сохранения детального отчета.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Макс. кол-во новых токенов")
    parser.add_argument("--temperature", type=float, default=0.0, help="Температура генерации")
    parser.add_argument("--show-examples", action="store_true", default=True, help="Показывать примеры транскрипций")
    parser.add_argument("--show-every", type=int, default=10, help="Показывать каждый N-й пример")
    parser.add_argument("--no-examples", action="store_true", help="Отключить показ примеров")
    
    args = parser.parse_args()
    
    # Если указан --no-examples, отключаем показ примеров
    show_examples = args.show_examples and not args.no_examples
    
    asr_model = Estimin3nASRInference(model_path=args.model_path)
    run_benchmark(
        asr_model=asr_model,
        dataset_path=args.dataset_path,
        source_sampling_rate=args.sampling_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_file=args.output_file,
        detailed_output_file=args.detailed_output_file,
        show_examples=show_examples,
        show_every=args.show_every
    )


if __name__ == "__main__":
    main()
