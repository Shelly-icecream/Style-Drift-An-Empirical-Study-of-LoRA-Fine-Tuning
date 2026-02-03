import requests
import base64
from io import BytesIO
from PIL import Image

A1111_URL = "http://127.0.0.1:7860"

def txt2img(
    prompt,
    negative_prompt="",
    steps=20,
    cfg_scale=7,
    width=512,
    height=512,
    seed=-1,
    sampler="DPM++ 2M Karras"
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "sampler_name": sampler,
    }

    r = requests.post(
        f"{A1111_URL}/sdapi/v1/txt2img",
        json=payload,
        timeout=300
    )
    r.raise_for_status()

    img_b64 = r.json()["images"][0]
    return Image.open(BytesIO(base64.b64decode(img_b64)))


