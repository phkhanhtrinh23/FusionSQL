from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / counts


def embed_texts(
    texts: Iterable[str],
    model_name_or_path: str,
    device: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 256,
    torch_dtype: Optional[str] = None,
) -> np.ndarray:
    texts_list: List[str] = [t if isinstance(t, str) else str(t) for t in texts]
    if not texts_list:
        return np.zeros((0, 0), dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = None
    if torch_dtype:
        if torch_dtype.lower() in ("fp16", "float16"):
            dtype = torch.float16
        elif torch_dtype.lower() in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif torch_dtype.lower() in ("fp32", "float32"):
            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    all_vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last_hidden = out.last_hidden_state  # (B, T, H)
            pooled = _mean_pool(last_hidden, enc["attention_mask"])  # (B, H)
            all_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_vecs, axis=0)


