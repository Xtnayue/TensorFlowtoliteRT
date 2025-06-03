# scripts/pt_to_tf.py
from transformers import AutoTokenizer, TFAutoModelForCausalLM

MODEL_PATH = "../models/qwen3-30b-a3b"
OUTPUT_PATH = "../converted/tf_model"

print("🧠 加载 Qwen3 模型（PyTorch）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForCausalLM.from_pretrained(MODEL_PATH, from_pt=True)

print("💾 保存为 TensorFlow 格式...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ 转换完成！")