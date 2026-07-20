import os
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# 🔥 NEW (QLoRA)
from peft import PeftModel

from dotenv import load_dotenv
load_dotenv()


# ==============================
# PATHS
# ==============================
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2")
LORA_PATH  = os.getenv("LORA_PATH", r"C:\Users\ABI-AI\Desktop\DocGPT\qlora_train\output2")


# ==============================
# TOKENIZER (UNCHANGED)
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


# ==============================
# 4-BIT CONFIG (UNCHANGED)
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


# ==============================
# LOAD BASE MODEL (UNCHANGED)
# ==============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# ==============================
# 🔥 APPLY QLoRA ADAPTER (ONLY NEW PART)
# ==============================
model = PeftModel.from_pretrained(
    model,
    LORA_PATH
)

model.eval()


print("✅ Mistral-7B-Instruct-v0.2 + QLoRA Streaming Chat Ready (type 'exit')\n")


# ==============================
# CHAT LOOP (UNCHANGED)
# ==============================
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer briefly and directly. "
                "Do not add extra explanations unless asked."
            )
        },
        {
            "role": "user",
            "content": user_input.strip()
        }
    ]

    # APPLY OFFICIAL CHAT TEMPLATE (UNCHANGED)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # BACKGROUND GENERATION (UNCHANGED)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistant: ", end="", flush=True)

    for token in streamer:
        print(token, end="", flush=True)

    print("\n")
