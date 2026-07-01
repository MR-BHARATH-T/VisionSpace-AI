import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MODEL = "Qwen/Qwen3-VL-8B-Thinking"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16

QUANTIZATION = "AUTO"

import os
is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
HF_CACHE_DIR = "/root/.cache/huggingface/hub" if is_kaggle else "hf_cache"

# torch.compile is disabled by default. For quantized BitsAndBytes models,
# torch.compile often gives little benefit and can introduce issues depending on the backend.
# It should remain an optional optimization.
ENABLE_TORCH_COMPILE = False

HF_REVISION = "main"


GENERATION_CONFIG = {

    "temperature": 0.1,

    "top_p": 0.9,

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
    "min_confidence": 0.20,
    "enable_multiview_merge": True,
    "enable_tiling": True,
    "enable_adaptive_crops": True
}