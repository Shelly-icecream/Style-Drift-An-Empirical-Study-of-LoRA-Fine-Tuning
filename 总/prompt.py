import os
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = r"C:\Users\HP\.cache\modelscope\hub\models\Qwen\Qwen2___5-VL-7B-Instruct"
IMAGE_DIR = r"D:\PythonProject2\总\clean_images"
OUT_DIR = "captions3"
os.makedirs(OUT_DIR, exist_ok=True)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("正在初始化 Qwen2.5-VL 模型...")

# 2. 限制分辨率：Qwen2.5-VL 默认像素极高，必须限制才能在 8GB 显存运行
# min_pixels 和 max_pixels 设置为 28 的倍数
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256*28*28,
    max_pixels=512*28*28,
    trust_remote_code=True, use_fast=False
)
print("--- 步骤 2: 正在加载大模型到显存 (可能会持续 1-3 分钟) ---")
# 3. 加载模型：使用 AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    trust_remote_code=True
)
print("✅ 模型加载成功！显卡已准备就绪。")
SYSTEM_PROMPT = ("""
# Role
You are an expert AI image captioning assistant. Your task is to describe the uploaded image for LoRA training.

# Instructions
Describe the image using concise, comma-separated tags. 
The tags must follow this specific order:
1. Start with the trigger phrase: "linkclick_style".
2. Followed by general style tags: "flat color, bold lines, high contrast, anime style".
3. Then describe objective elements: subject (e.g., boy/girl), clothing, hair color, pose, and facial expression.
4. End with background and lighting: (e.g., simple background, blue lighting).

# Constraints
- Use English only.
- Use simple nouns and short phrases.
- Avoid full sentences. 
- Ensure the output starts with "linkclick_style, flat color, bold lines," for every image.
""")
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

print(f"--- 步骤 3: 开始处理图片 (共 {len(image_files)} 张) ---")
for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    image_path = os.path.join(IMAGE_DIR, fname)
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"🚀 正在分析: {fname}")

        # 构建符合 Qwen2.5 标准的对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]

        # 准备输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            # 自动裁剪掉 Prompt 部分
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # 保存
        with open(os.path.join(OUT_DIR, os.path.splitext(fname)[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write(caption.strip())

        print(f"✅ 成功: {caption[:50]}...")

    except Exception as e:
        print(f"❌ 出错: {fname} - {e}")

print("🎉 任务完成！")