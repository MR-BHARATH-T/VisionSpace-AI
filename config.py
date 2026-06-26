import torch

MODELS = {

    "vision": {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct"
    },

    "text": {
        "model": "Qwen/Qwen3-VL-4B-Instruct"
    }

}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16

QUANTIZATION = "AUTO"

import os
HF_CACHE_DIR = "/root/.cache/huggingface/hub" if os.path.exists("/root/.cache/huggingface/hub") else "hf_cache"

# torch.compile is disabled by default. For quantized BitsAndBytes models,
# torch.compile often gives little benefit and can introduce issues depending on the backend.
# It should remain an optional optimization.
ENABLE_TORCH_COMPILE = False

HF_REVISION = "main"


GENERATION_CONFIG = {

    "temperature": 0.2,

    "top_p": 0.95,

    "max_new_tokens": 2048,

    "do_sample": False

}

MAX_IMAGE_SIZE = 1024

SUPPORTED_IMAGE_TYPES = [

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"

]

OUTPUT_DIR = "outputs"

VERBOSE = True

MAX_IMAGES_PER_REQUEST = 6
MAX_VISUAL_TOKENS = 12000

VISION_CONFIG = {
    "max_images": 6,
    "max_visual_tokens": 12000,
    "min_confidence": 0.45,
    "enable_multiview_merge": True
}