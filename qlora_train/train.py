import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =========================
# PATHS
# =========================
MODEL_PATH = r"C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2"
DATA_PATH = "train.jsonl"
OUTPUT_DIR = "output"

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# DATASET
# =========================
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# =========================
# QLoRA CONFIG
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# =========================
# BASE MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False

# =========================
# LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,      # 🚨 DO NOT INCREASE
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# =========================
# SFT TRAINER (CORRECT FOR NEW TRL)
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE LoRA
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA training complete")
