import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# ==============================
# MODEL PATH
# ==============================
MODEL_PATH = r"C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2"

# ==============================
# LOAD TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ==============================
# LOAD MODEL (4-BIT)
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

print("✅ Mistral-7B-Instruct-v0.2 FINAL Streaming Chat Ready (type 'exit')\n")

# ==============================
# CHAT LOOP
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
        {"role": "user", "content": user_input.strip()}
    ]

    # ==============================
    # APPLY CHAT TEMPLATE
    # (returns Tensor in your version)
    # ==============================
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # ✅ Manually build attention mask (fixes warning)
    attention_mask = torch.ones_like(input_ids).to(model.device)

    # ==============================
    # STREAMER
    # ==============================
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # ==============================
    # GENERATE IN BACKGROUND
    # ==============================
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistant: ", end="", flush=True)

    for token in streamer:
        print(token, end="", flush=True)

    print("\n")
