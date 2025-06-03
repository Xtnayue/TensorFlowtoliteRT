# scripts/tf_to_tflite.py
import tensorflow as tf
from transformers import AutoTokenizer

MODEL_PATH = "../converted/tf_model"
OUTPUT_TFLITE = "../converted/qwen3.tflite"

print("🧠 加载 TensorFlow 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

print("🔄 开始转换为 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 启用量化压缩
tflite_model = converter.convert()

print(f"💾 保存为 {OUTPUT_TFLITE}...")
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

    print("✅ 转换完成！")