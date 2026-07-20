from transformers import AutoConfig

MODEL_PATH = r"C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2"

config = AutoConfig.from_pretrained(MODEL_PATH)
print("Model type:", config.model_type)
print("Hidden size:", config.hidden_size)
print("Num layers:", config.num_hidden_layers)
