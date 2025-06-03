# convert_hf_pt_to_tf.py
from transformers import AutoTokenizer, TFAutoModelForCausalLM

MODEL_PATH = "./qwen3-30b-a3b"
OUTPUT_PATH = "./qwen3-tf"

# 加载 PyTorch 模型并转换为 TensorFlow 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForCausalLM.from_pretrained(MODEL_PATH, from_pt=True)
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ 模型已保存为 TensorFlow 格式")
