
import os
from unsloth import FastModel
import torch
from huggingface_hub import snapshot_download
from transformers import TextStreamer
from datasets import load_dataset,Audio,concatenate_datasets
import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 256  
from utils.config import DEFAULT_DATASET_DIR, OUTPUTS_DIR, MODELS_DIR, MODEL_NAME, ensure_dirs


ensure_dirs()

model, processor = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, 
    max_seq_length = 2048, 
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, 
)

def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True, 
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        do_sample=False,
        streamer = TextStreamer(processor, skip_prompt = True),
    )


dataset = load_dataset(str(DEFAULT_DATASET_DIR))
test_audio = dataset['train'][7546]

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(test_audio['text'])
messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": test_audio['audio']['array']},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            }
        ]

do_gemma_3n_inference(messages, max_new_tokens = 256)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, 
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 8,
    lora_alpha = 16,                  # alpha == 2*r
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               
    loftq_config = None,              
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        
        # audio layers
        "post", "linear_start", "linear_end",
        "embedding_projection",
    ],
    modules_to_save=[ # must have
        "lm_head",
        "embed_tokens",
        "embed_audio",
    ],
)

def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        label = str(samples["text"][idx])

        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples

dataset = dataset.map(format_intersection_data, batched=True, batch_size=4, num_proc=4)
def collate_fn(examples):
        texts = []
        audios = []

        for example in examples:
            # apply chat template to get text
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)

            # extract audios
            audios.append(example["audio"]["array"])

        
        batch = processor(
            text=texts, audio=audios, return_tensors="pt", padding=True
        )

        # the labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        # gemma-3n specific token masking
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if hasattr(processor.tokenizer, 'image_token_id'):
            labels[labels == processor.tokenizer.image_token_id] = -100
        if hasattr(processor.tokenizer, 'audio_token_id'):
            labels[labels == processor.tokenizer.audio_token_id] = -100
        if hasattr(processor.tokenizer, 'boi_token_id'):
            labels[labels == processor.tokenizer.boi_token_id] = -100
        if hasattr(processor.tokenizer, 'eoi_token_id'):
            labels[labels == processor.tokenizer.eoi_token_id] = -100


        batch["labels"] = labels
        return batch

### Train ###

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    processing_class=processor.tokenizer,    
    data_collator=collate_fn,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,
        warmup_ratio = 0.1,
        #max_steps = 60, 
        num_train_epochs = 1,              # Set this instead of max_steps for full training runs
        learning_rate = 5e-5, 
        logging_steps = 10,
        save_strategy="steps",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", 
        seed = 3407,
    output_dir = str(OUTPUTS_DIR),            # directory to save the model
        report_to = ["wandb"],             # For Weights and Biases
        
        # must have for audio finetuning:
        remove_unused_columns = False,
        dataset_text_field = "messages", # field we formatted
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 2,
        max_seq_length = 2048,
    )
)


trainer_stats = trainer.train()


#inference
messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": test_audio['audio']['array']},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            }
        ]

do_gemma_3n_inference(messages, max_new_tokens = 256)

### saving model and processor ###
final_dir = MODELS_DIR / MODEL_NAME
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(final_dir))  
processor.save_pretrained(str(final_dir))

### float16 ###
model.save_pretrained_merged(str(final_dir), processor)

model.save_pretrained_gguf(
    str(final_dir / "gguf"),
    quantization_type = "Q8_0", # for now only Q8_0, BF16, F16 supported
) 