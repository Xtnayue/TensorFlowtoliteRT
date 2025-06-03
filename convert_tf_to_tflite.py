# convert_tf_to_tflite.py
import tensorflow as tf
from transformers import AutoTokenizer

MODEL_PATH = "./qwen3-tf"

# 加载 Tokenizer 和 TF 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# 设置 TFLite 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 启用量化优化
tflite_model = converter.convert()

# 保存 TFLite 模型
with open("qwen3.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ 模型已转换为 TFLite 格式")
