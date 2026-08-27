import base64
import gc
import os
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import set_seed as hf_set_seed


def load_image(path: str):
    """
    Loads the image from given path
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def clear_cuda():
    """
    Clears the cache from CUDA
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def all_seed(seed=42):
    """
    Setting SEED across all random number generators for reproducible experiments
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)


def image_to_base64(image_path_or_pil):
    """
    Converts a PIL Image or image path to a base64 string suitable for VLM input.
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"
