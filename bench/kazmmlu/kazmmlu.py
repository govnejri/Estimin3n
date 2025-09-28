############################################
###    RUN WITH TORCHDYNAMO_DISABLE=1    ###
############################################

import torch
torch._dynamo.disable()

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
from utils.config import DEFAULT_MODEL_PATH, OUTPUTS_DIR, get_device, get_dtype, MODEL_NAME, ensure_dirs


ensure_dirs()

class Estimin3nKazMMLUInference:
    def __init__(self, model_path: str | os.PathLike | None = None):
        self.model_path = str(Path(model_path).expanduser()) if model_path else str(DEFAULT_MODEL_PATH)
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        print(f"📊 Устройство: {self.device} | Тип данных: {self.dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
    print(f"🚀 Модель {MODEL_NAME} загружена и готова для KazMMLU!\n")

    def format_question(self, question, options, language="kazakh"):
        """Форматирует вопрос в стиле множественного выбора"""
        if language == "kazakh":
            prompt = f"""Сұрақ: {question}

A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}
E) {options['E']}

Дұрыс жауап (тек әріпті жазыңыз):"""
        else:
            prompt = f"""Вопрос: {question}

A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}
E) {options['E']}

Правильный ответ (только букву):"""
        
        return prompt

    def get_answer(self, prompt, max_new_tokens=5, temperature=0.1):
        """Получает ответ от модели"""
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': True if temperature > 0 else False,
                'temperature': temperature,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            answer_clean = self.extract_answer_letter(answer)
            return answer_clean
            
        except Exception as e:
            print(f"❌ Ошибка при генерации ответа: {e}")
            return "N/A"

    def extract_answer_letter(self, text):
        """Извлекает букву ответа из сгенерированного текста"""
        match = re.search(r'\b([ABCDE])\b', text.upper())
        if match:
            return match.group(1)
        
        clean_text = text.strip().upper()
        if clean_text and clean_text[0] in 'ABCDE':
            return clean_text[0]
            
        return "N/A"

def get_available_configs():
    """Получает список доступных конфигураций KazMMLU"""
    return [
        'Accounting and Auditing (Professional & University in rus)',
        'Biology (High School in kaz)', 'Biology (High School in rus)',
        'Biology (Professional & University in rus)', 'Chemistry (High School in kaz)',
        'Chemistry (High School in rus)', 'Culture and Art (Professional & University in rus)',
        'Economics and Entrepreneurship (Professional in rus)',
        'Education and Training (Professional & University in rus)',
        'Finance (Professional & University in rus)',
        'General Education Disciplines (Professional & University in rus)',
        'Geography (High School in kaz)', 'Geography (High School in rus)',
        'Informatics (High School in kaz)', 'Informatics (High School in rus)',
        'Jurisprudence (Professional & University in rus)',
        'Kazakh History (High School in kaz)', 'Kazakh History (High School in rus)',
        'Kazakh Language (High School in kaz)', 'Kazakh Literature (High School in kaz)',
        'Law (High School in kaz)', 'Law (High School in rus)',
        'Management and Marketing (Professional & University in rus)',
        'Math (High School in kaz)', 'Math (High School in rus)',
        'Math Literacy (High School in rus)', 'Medicine (Professional & University in rus)',
        'Philosophy and Psychology (Professional & University in rus)',
        'Physics (High School in kaz)', 'Physics (High School in rus)',
        'Reading Literacy (High School in kaz)', 'Reading Literacy (High School in rus)',
        'Russian Language (High School in rus)', 'Russian Literature (High School in rus)',
        'Social Science (Professional & University in rus)',
        'World History (High School in kaz)', 'World History (High School in rus)'
    ]

def print_sample_comparison(sample_num, question, options, correct_answer, predicted_answer, 
                          subject, level, is_correct):
    """Красивый вывод сравнения для отдельного вопроса"""
    status = "✅ ПРАВИЛЬНО" if is_correct else "❌ НЕПРАВИЛЬНО"
    print(f"\n{'='*100}")
    print(f"📝 ОБРАЗЕЦ #{sample_num} | {subject} ({level}) | {status}")
    print(f"{'='*100}")
    print(f"❓ ВОПРОС:")
    print(f"   {question}")
    print(f"\n📋 ВАРИАНТЫ ОТВЕТОВ:")
    for letter in ['A', 'B', 'C', 'D', 'E']:
        marker = "👉" if predicted_answer == letter else "  "
        correct_marker = "⭐" if correct_answer == letter else "  "
        print(f"   {marker} {correct_marker} {letter}) {options[letter]}")
    
    print(f"\n🎯 ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}")
    print(f"🤖 ОТВЕТ МОДЕЛИ:     {predicted_answer}")
    print(f"{'='*100}")

def run_kazmlu_benchmark(model, max_tokens=5, temperature=0.1, 
                        output_file=None, 
                        detailed_output_file=None,
                        show_examples=True, show_every=50, subset_names=None,
                        run_all_configs=True):
    output_file = str(OUTPUTS_DIR / "kazmlu_results.tsv") if output_file is None else output_file
    detailed_output_file = str(OUTPUTS_DIR / "kazmlu_detailed.tsv") if detailed_output_file is None else detailed_output_file
    
    # Определяем конфигурации для запуска
    if subset_names:
        configs_to_run = subset_names if isinstance(subset_names, list) else [subset_names]
    elif run_all_configs:
        configs_to_run = get_available_configs()
    else:
        print("❌ Не указаны конфигурации для запуска")
        return None

    print(f"🔄 Будет запущено {len(configs_to_run)} конфигураций:")
    for i, config in enumerate(configs_to_run[:5], 1):
        print(f"   {i}. {config}")
    if len(configs_to_run) > 5:
        print(f"   ... и ещё {len(configs_to_run) - 5} конфигураций")

    all_results_data = []
    total_correct = 0
    total_questions = 0
    config_stats = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("config\tsample_count\taccuracy\tcorrect\ttotal\n")
        
        for config_idx, config_name in enumerate(configs_to_run):
            print(f"\n{'='*100}")
            print(f"📚 ОБРАБОТКА КОНФИГУРАЦИИ {config_idx + 1}/{len(configs_to_run)}")
            print(f"📖 {config_name}")
            print(f"{'='*100}")
            
            try:
                # Загружаем конкретную конфигурацию
                dataset = load_dataset("MBZUAI/KazMMLU", config_name)['test']
                print(f"✅ Загружено {len(dataset)} вопросов")
            except Exception as e:
                print(f"❌ Ошибка загрузки конфигурации '{config_name}': {e}")
                continue

            if len(dataset) == 0:
                print(f"⚠️ Пустая конфигурация: {config_name}")
                continue

            config_correct = 0
            config_total = 0
            
            for i, sample in enumerate(tqdm(dataset, desc=f"🧠 {config_name[:30]}...")):
                try:
                    # Извлекаем данные
                    question = sample['Question']
                    options = {
                        'A': sample['Option A'],
                        'B': sample['Option B'], 
                        'C': sample['Option C'],
                        'D': sample['Option D'],
                        'E': sample['Option E']
                    }
                    correct_answer = sample['Answer Key']
                    subject = sample.get('Subject', config_name)
                    level = sample.get('Level', 'Unknown')
                    language = "kazakh" if sample.get('Language') == "Kazakh" else "russian"
                    
                    # Форматируем промпт
                    prompt = model.format_question(question, options, language)
                    
                    # Получаем ответ модели
                    predicted_answer = model.get_answer(
                        prompt, 
                        max_new_tokens=max_tokens, 
                        temperature=temperature
                    )
                    
                    # Проверяем правильность
                    is_correct = predicted_answer == correct_answer
                    if is_correct:
                        config_correct += 1
                        total_correct += 1
                    config_total += 1
                    total_questions += 1
                    
                    # Сохраняем результат
                    all_results_data.append({
                        'config': config_name,
                        'sample_id': i + 1,
                        'question': question,
                        'correct_answer': correct_answer,
                        'predicted_answer': predicted_answer,
                        'is_correct': is_correct,
                        'subject': subject,
                        'level': level,
                        'language': language
                    })
                    
                    # Показываем примеры
                    if show_examples and ((i + 1) % show_every == 0 or i == 0):
                        print_sample_comparison(
                            i + 1, question, options, correct_answer, 
                            predicted_answer, subject, level, is_correct
                        )
                    
                except Exception as e:
                    print(f"❌ Ошибка обработки вопроса {i+1} в {config_name}: {e}")
                    continue
            
            # Статистика по конфигурации
            config_accuracy = config_correct / config_total if config_total > 0 else 0
            config_stats[config_name] = {
                'correct': config_correct,
                'total': config_total,
                'accuracy': config_accuracy
            }
            
            print(f"\n📊 РЕЗУЛЬТАТ КОНФИГУРАЦИИ '{config_name}':")
            print(f"   Точность: {config_accuracy:.4f} ({config_accuracy*100:.2f}%)")
            print(f"   Правильных: {config_correct}/{config_total}")
            
            # Сохраняем результат конфигурации
            f.write(f"{config_name}\t{config_total}\t{config_accuracy:.4f}\t{config_correct}\t{config_total}\n")
            f.flush()

    # Финальные результаты
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    print(f"\n{'='*100}")
    print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ KazMMLU")
    print(f"{'='*100}")
    print(f"📊 Всего конфигураций:    {len(config_stats)}")
    print(f"📊 Всего вопросов:        {total_questions}")
    print(f"✅ Правильных ответов:    {total_correct}")
    print(f"❌ Неправильных ответов:  {total_questions - total_correct}")
    print(f"🎯 **ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)**")
    print(f"{'='*100}")
    
    # ТОП конфигураций
    print(f"\n📈 ТОП-10 ЛУЧШИХ КОНФИГУРАЦИЙ:")
    sorted_configs = sorted(config_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (config, stats) in enumerate(sorted_configs[:10], 1):
        print(f"   {i:2d}. {stats['accuracy']:.3f} ({stats['accuracy']*100:.1f}%) | {config}")
    
    print(f"\n📉 ТОП-5 ХУДШИХ КОНФИГУРАЦИЙ:")
    for i, (config, stats) in enumerate(sorted_configs[-5:], 1):
        print(f"   {i:2d}. {stats['accuracy']:.3f} ({stats['accuracy']*100:.1f}%) | {config}")
    
    # Сохранение финальных результатов
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"OVERALL\t{total_questions}\t{overall_accuracy:.4f}\t{total_correct}\t{total_questions}\n")
    
    # Детальный отчет
    results_df = pd.DataFrame(all_results_data)
    results_df.to_csv(detailed_output_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"\n💾 Результаты сохранены:")
    print(f"   Основной лог: {output_file}")
    print(f"   Детальный отчет: {detailed_output_file}")
    
    return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description=f"Бенчмарк {MODEL_NAME} на KazMMLU")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), 
                       help="Путь к модели")
    parser.add_argument("--subset", type=str, default=None, 
                       help="Конкретное подмножество для тестирования")
    parser.add_argument("--run-all", action="store_true", default=True,
                       help="Запустить все конфигурации (по умолчанию)")
    parser.add_argument("--kazakh-only", action="store_true",
                       help="Только казахские конфигурации")
    parser.add_argument("--russian-only", action="store_true", 
                       help="Только русские конфигурации")
    parser.add_argument("--output_file", type=str, default="kazmlu_benchmark.tsv", 
                       help="Файл для сохранения результатов")
    parser.add_argument("--detailed_output_file", type=str, default="kazmlu_detailed.tsv", 
                       help="Файл для детального отчета")
    parser.add_argument("--max-tokens", type=int, default=5, 
                       help="Макс. количество новых токенов")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Температура генерации")
    parser.add_argument("--show-examples", action="store_true", default=True, 
                       help="Показывать примеры вопросов и ответов")
    parser.add_argument("--show-every", type=int, default=50, 
                       help="Показывать каждый N-й пример")
    parser.add_argument("--no-examples", action="store_true", 
                       help="Отключить показ примеров")
    
    args = parser.parse_args()
    
    # Определяем конфигурации для запуска
    subset_names = None
    run_all = True
    
    if args.subset:
        subset_names = [args.subset]
        run_all = False
    elif args.kazakh_only:
        subset_names = [config for config in get_available_configs() if "in kaz)" in config]
        run_all = False
    elif args.russian_only:
        subset_names = [config for config in get_available_configs() if "in rus)" in config]
        run_all = False
    
    show_examples = args.show_examples and not args.no_examples
    
    print(f"🚀 Инициализация модели {MODEL_NAME} для KazMMLU...")
    model = Estimin3nKazMMLUInference(model_path=args.model_path)
    
    accuracy = run_kazmlu_benchmark(
        model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_file=args.output_file,
        detailed_output_file=args.detailed_output_file,
        show_examples=show_examples,
        show_every=args.show_every,
        subset_names=subset_names,
        run_all_configs=run_all
    )
    
    if accuracy is not None:
        print(f"\n🎊 Бенчмарк завершен! Итоговая точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"\n💔 Бенчмарк завершен с ошибками")

if __name__ == "__main__":
    main()
