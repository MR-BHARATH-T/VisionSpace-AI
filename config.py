import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

VERCEL_API_KEY = os.getenv("VERCEL_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# --- PROVIDER SELECTION ---
# Set to "huggingface" to load models locally (e.g. on a Kaggle T4/L4 GPU).
# Set to "deepinfra" to execute fast cloud completions (requires API key).
PROVIDER = os.getenv("PROVIDER", "deepinfra" if (VERCEL_API_KEY or DEEPINFRA_API_KEY) else "huggingface")

if PROVIDER == "deepinfra":
    MODEL = "deepinfra/alibaba/qwen3-vl-instruct"
else:
    MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    # MODEL = "Qwen/Qwen3-VL-8B-Thinking"
    # MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

SERP_API = SERP_API_KEY or "c4d8337a0efa46f024138f66b99be987882697ecfab07dfb75a893d437a1a803"

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
    "temperature": 0.0,
    "max_new_tokens": 768,
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
    "enable_multiview_merge": True
}